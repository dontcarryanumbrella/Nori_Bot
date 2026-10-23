import asyncio
import os
import random
import re
import threading
import time
from typing import AsyncIterator, Dict, List, Optional

import speech_recognition as sr
from openai import AsyncOpenAI, OpenAI

from assistant_config import (
    AVATAR_ACTIONS,
    AVATAR_EMOTIONS,
    EXIT_WORDS,
    GIVE_BALL_PATTERN,
    LLM_API_KEY,
    LLM_BASE_URL,
    MIN_SENTENCE_CHARS,
    MODIFY_MEMORY_PATTERN,
    NO_WORDS,
    NUMBER_WORDS,
    REMEMBER_COMMAND_PATTERN,
    REPLY_TAG_PATTERN,
    SENTENCE_BREAK_PATTERN,
    SUPPORTED_AUDIO_EXTENSIONS,
    TAKE_BALL_PATTERN,
    TTS_API_KEY,
    TTS_BASE_URL,
    TTS_PROVIDER,
    WAIT_COMMAND_PATTERN,
    YES_WORDS,
)
from assistant_types import AssistantState, AssistantTurn
from audio_workers import (
    BlockingASRWorker,
    EdgeSpeech,
    KeyboardMonitor,
    LLMWorker,
    OpenAISpeech,
    PlaybackController,
    TTSWorker,
)
from avatar_window import NoriBridge, NullAvatar
from conversation_store import ConversationStore



AVATAR_ACTIVITY_BY_STATE = {
    AssistantState.IDLE: "idle",
    AssistantState.LISTENING: "listen",
    AssistantState.WAITING: "sleep",
    AssistantState.THINKING: "think",
    AssistantState.SPEAKING: "talk",
    AssistantState.STOPPING: "idle",
    AssistantState.SHUTDOWN: "sleep",
}


class AssistantCoordinator:
    """
    Simple listen -> think -> speak loop.
    Replies stream sentence by sentence, so speech starts before the whole reply is written.
    Right Shift (or Escape / Stop in Nori's window) interrupts speech.
    """

    def __init__(self, avatar=None, bridge: Optional[NoriBridge] = None):
        if not LLM_API_KEY:
            raise ValueError("No chat API key. Set LLM_API_KEY (or OPENAI_API_KEY) in .env.")
        if TTS_PROVIDER == "edge":
            speech = EdgeSpeech()
        elif TTS_PROVIDER == "openai":
            if not TTS_API_KEY:
                raise ValueError("No text-to-speech API key. Set TTS_API_KEY (or OPENAI_API_KEY) in .env.")
            speech = OpenAISpeech(AsyncOpenAI(api_key=TTS_API_KEY, base_url=TTS_BASE_URL))
        else:
            raise ValueError(f"Unknown TTS_PROVIDER {TTS_PROVIDER!r}. Use 'edge' or 'openai'.")

        # The chat client also writes conversation summaries and long-term memory.
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.bot_dir = os.path.dirname(os.path.abspath(__file__))
        self.music_dir = os.path.join(self.bot_dir, "music")
        self.user_memory_path = os.path.join(self.bot_dir, "user_memory.txt")

        self.avatar = avatar or NullAvatar()
        self.bridge = bridge
        self._state = AssistantState.IDLE
        self.turn_counter = 0

        self.conversation_store = ConversationStore(self.client, self.user_memory_path)
        self.asr = BlockingASRWorker()
        self.llm = LLMWorker(self.async_client)

        self.playback = PlaybackController()
        self.tts = TTSWorker(speech, self.playback)
        self.key_monitor = KeyboardMonitor()

        self.tts_stop_event = threading.Event()
        self.ui_interrupt = threading.Event()
        self.typed_text: Optional[asyncio.Queue] = None
        self.wake_requested = False
        self.ball_active = False
        self._listen_future: Optional[asyncio.Future] = None
        self._listen_result_wanted = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self.music_stop_event = threading.Event()
        self.pending_interrupt_context: Optional[Dict[str, str]] = None
        self.music_task: Optional[asyncio.Task] = None
        self.current_music_file: Optional[str] = None

        self.has_done_first_prompt = False
        self.feedback_phrases = [
            "hmmmm",
            "okay",
            "wait",
            "well",
            "alright",
        ]
        self.waiting_until: Optional[float] = None
        self.waiting_for_audio_offer = False
        self.waiting_audio_mode = False
        self.last_user_timestamp: Optional[float] = None
        self.last_assistant_timestamp: Optional[float] = None
        self.recent_user_timestamps: List[float] = []
        self.idle_timeout_streak = 0
        self.total_user_turns = 0
        self.total_completed_assistant_turns = 0
        self.total_tracked_interruptions = 0
        self.total_ignored_interruptions = 0
        self.pending_system_line: Optional[str] = None

    @property
    def state(self) -> AssistantState:
        return self._state

    @state.setter
    def state(self, value: AssistantState) -> None:
        self._state = value
        self._sync_avatar_activity()

    def _sync_avatar_activity(self) -> None:
        activity = AVATAR_ACTIVITY_BY_STATE[self._state]
        if self._state == AssistantState.WAITING and self.waiting_audio_mode:
            activity = "vibe"
        self.avatar.set_activity(activity)

    def _on_typed_text(self, text: str) -> None:
        if self.typed_text is not None:
            self.typed_text.put_nowait(text)

    def _on_poke(self, part: str) -> None:
        if self.is_waiting_active() and not self.waiting_for_audio_offer:
            self.wake_requested = True

    def request_shutdown(self) -> None:
        """Thread-safe. Used when Nori's window is closed."""
        loop, task = self._loop, self._main_task
        if loop is None or task is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass

    def _record_interruption(self, turn: AssistantTurn) -> AssistantTurn:
        turn.interrupted = True
        turn.delivered = False
        if not turn.track_interruption:
            self.total_ignored_interruptions += 1
            return turn

        self.total_tracked_interruptions += 1
        cut_off_reply = (turn.assistant_text or "").strip()
        self.pending_interrupt_context = {
            "type": "user_interrupted_assistant",
            "cut_off_reply": cut_off_reply[:120],
        }
        return turn

    def _next_turn_id(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    def is_waiting_active(self) -> bool:
        return self.waiting_until is not None and time.monotonic() < self.waiting_until

    def clear_wait_state(self) -> None:
        self.waiting_until = None
        self.waiting_for_audio_offer = False
        self.waiting_audio_mode = False

    def queue_system_line(self, text: str) -> None:
        self.pending_system_line = text.strip()

    def consume_pending_system_line(self) -> Optional[AssistantTurn]:
        if not self.pending_system_line:
            return None

        text = self.pending_system_line
        self.pending_system_line = None
        return AssistantTurn(
            turn_id=self._next_turn_id(),
            user_text="Speak this queued system line in character before resuming listening.",
            assistant_text=text,
            system_generated=True,
            track_interruption=False,
        )

    def record_user_activity(self) -> None:
        now = time.monotonic()
        self.last_user_timestamp = now
        self.idle_timeout_streak = 0
        self.total_user_turns += 1
        self.recent_user_timestamps.append(now)
        cutoff = now - 45.0
        self.recent_user_timestamps = [ts for ts in self.recent_user_timestamps if ts >= cutoff]

    def record_idle_timeout(self) -> None:
        self.idle_timeout_streak += 1

    def get_posture_state(self) -> str:
        if self.is_waiting_active():
            return "cooldown"

        now = time.monotonic()
        if self.last_user_timestamp is None:
            return "cooldown"

        recent_count = len([ts for ts in self.recent_user_timestamps if now - ts <= 30.0])
        silence_seconds = now - self.last_user_timestamp

        if recent_count >= 3 and silence_seconds <= 12.0:
            return "engaged"
        if silence_seconds >= 35.0 or self.idle_timeout_streak >= 2:
            return "quiet_gap"
        return "cooldown"

    def build_posture_note(self, posture_state: str) -> str:
        if posture_state == "engaged":
            return (
                "Current conversation posture: engaged. The pace has been active and recent. "
                "Keep the same persona, but sound a little sharper, quicker, and more naturally locked in."
            )
        if posture_state == "quiet_gap":
            return (
                "Current conversation posture: quiet_gap. There has been a noticeable lull. "
                "Keep the same persona, but sound a little less eager, slightly more self-aware, and more sparse."
            )
        return "Current conversation posture: cooldown. Keep the normal persona and default pacing."

    def parse_wait_command(self, text: str) -> Optional[int]:
        match = WAIT_COMMAND_PATTERN.search(text.strip())
        if not match:
            return None

        raw_amount = match.group("amount").lower()
        unit = match.group("unit").lower()

        if raw_amount.isdigit():
            amount = int(raw_amount)
        else:
            normalized_amount = raw_amount.replace("-", " ")
            parts = [part for part in normalized_amount.split() if part]
            amount = 0
            for part in parts:
                if part not in NUMBER_WORDS:
                    return None
                amount += NUMBER_WORDS[part]
            if amount <= 0:
                return None
        multiplier = 60 if unit.startswith("minute") else 1
        total_seconds = amount * multiplier

        if total_seconds <= 0:
            return None
        return total_seconds

    def format_wait_duration(self, total_seconds: int) -> str:
        if total_seconds % 60 == 0 and total_seconds >= 60:
            minutes = total_seconds // 60
            return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
        return f"{total_seconds} second" if total_seconds == 1 else f"{total_seconds} seconds"

    def classify_audio_offer_reply(self, text: str) -> Dict[str, str]:
        lowered = text.lower().strip()
        tokens = set(re.findall(r"[a-z0-9']+", lowered))

        if tokens & NO_WORDS:
            return {"action": "no", "query": ""}

        if tokens & YES_WORDS or "play" in tokens:
            cleaned = re.sub(
                r"\b(?:yes|yeah|yep|sure|ok|okay|alright|bet|please|play|something|some|music)\b",
                " ",
                lowered,
            )
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return {"action": "yes", "query": cleaned}

        return {"action": "yes", "query": lowered}

    def find_music_file(self, query: str = "") -> Optional[str]:
        if not os.path.isdir(self.music_dir):
            return None

        candidates = []
        for name in os.listdir(self.music_dir):
            full_path = os.path.join(self.music_dir, name)
            if not os.path.isfile(full_path):
                continue
            _, ext = os.path.splitext(name)
            if ext.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                candidates.append(full_path)

        if not candidates:
            return None

        if not query.strip():
            return random.choice(candidates)

        query_tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 2]
        if not query_tokens:
            return random.choice(candidates)

        scored_candidates = []
        for path in candidates:
            filename = os.path.splitext(os.path.basename(path))[0].lower()
            score = sum(1 for token in query_tokens if token in filename)
            if score > 0:
                scored_candidates.append((score, path))

        if not scored_candidates:
            return None

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return scored_candidates[0][1]

    def should_use_feedback(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False

        if len(text.split()) <= 3:
            return False

        return random.random() < 0.45

    def pick_feedback_phrase(self, text: str) -> str:
        lower = text.lower()
        if "?" in text:
            return random.choice(["hm", "wait", "okay"])
        if any(word in lower for word in ["sorry", "sad", "hurt", "pain", "alone", "lost"]):
            return random.choice(["hmm", "yeah", "ah"])
        return random.choice(self.feedback_phrases)

    async def _run_music_playback(self, filename: str) -> None:
        loop = asyncio.get_running_loop()
        finished_normally = False
        try:
            finished_normally = await loop.run_in_executor(
                None,
                self.playback.play_file_blocking,
                filename,
                self.music_stop_event,
            )
        finally:
            if self.current_music_file == filename:
                self.current_music_file = None
                self.music_task = None
                self.waiting_audio_mode = False
                self._sync_avatar_activity()
                if finished_normally:
                    if self.is_waiting_active():
                        self.queue_system_line("ight the song ended. I'm still waiting till your timer cooks through.")
                    else:
                        self.clear_wait_state()
                        self.queue_system_line(self.build_timer_done_line())

    async def start_wait_music(self, filename: str) -> None:
        await self.stop_wait_music()
        self.music_stop_event.clear()
        self.current_music_file = filename
        self.waiting_audio_mode = True
        self._sync_avatar_activity()
        self.music_task = asyncio.create_task(self._run_music_playback(filename))

    async def stop_wait_music(self) -> None:
        self.music_stop_event.set()
        task = self.music_task
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self.music_task = None
        self.current_music_file = None
        self.waiting_audio_mode = False
        self._sync_avatar_activity()

    def build_timer_done_line(self) -> str:
        return random.choice(
            [
                "ight the timer cooked through, I'm back now.",
                "aight that's enough waiting, I'm here again.",
                "timer done. I'm active again, don't waste it.",
            ]
        )

    def build_early_wake_line(self) -> str:
        return random.choice(
            [
                "ight you hit enter so I'm back early. what's the move.",
                "say less I'm back early now. what are we doing.",
                "aight I'm up again before the timer. talk to me nice.",
                "bet I'm back early. don't start chatting nonsense now.",
                "cool I'm active again early. what do you want.",
            ]
        )

    def parse_explicit_memory_request(self, text: str) -> Optional[str]:
        match = REMEMBER_COMMAND_PATTERN.match(text.strip())
        if not match:
            return None

        content = match.group("content").strip()
        if not content:
            return None

        return content

    def parse_modify_memory_request(self, text: str) -> Optional[str]:
        match = MODIFY_MEMORY_PATTERN.match(text.strip())
        if not match:
            return None

        content = match.group("content").strip()
        if not content:
            return None

        return content

    def is_incomplete_memory_note(self, text: str) -> bool:
        cleaned = text.strip().lower().rstrip(".!?")
        if not cleaned:
            return True

        incomplete_phrases = {
            "i am",
            "i'm",
            "im",
            "my",
            "my password",
            "my code",
            "the password is",
            "the code is",
            "it is",
            "its",
            "it's",
        }
        if cleaned in incomplete_phrases:
            return True

        word_count = len(cleaned.split())
        if word_count <= 1:
            return True

        return False

    def build_relationship_metrics(self) -> str:
        interruption_rate = 0.0
        if self.total_completed_assistant_turns:
            interruption_rate = self.total_tracked_interruptions / max(1, self.total_completed_assistant_turns)

        return (
            f"user_turns={self.total_user_turns}; "
            f"completed_assistant_turns={self.total_completed_assistant_turns}; "
            f"tracked_interruptions={self.total_tracked_interruptions}; "
            f"ignored_proactive_interruptions={self.total_ignored_interruptions}; "
            f"idle_timeout_streak={self.idle_timeout_streak}; "
            f"recent_activity_count={len(self.recent_user_timestamps)}; "
            f"interruption_rate={interruption_rate:.2f}"
        )

    async def maybe_handle_special_turn(self, turn: AssistantTurn) -> Optional[AssistantTurn]:
        if turn.system_generated:
            return turn

        modified_memory = self.parse_modify_memory_request(turn.user_text)
        if modified_memory is not None:
            if self.is_incomplete_memory_note(modified_memory):
                turn.assistant_text = "nah finish the memory first. that edit is mad incomplete."
                return turn

            replaced = self.conversation_store.replace_last_explicit_memory(modified_memory)
            if replaced:
                turn.assistant_text = random.choice(
                    [
                        f"bet I updated it. {modified_memory}",
                        f"aight I changed that memory. {modified_memory}",
                        f"say less I replaced it with this. {modified_memory}",
                    ]
                )
            else:
                turn.assistant_text = "there's nothing saved for me to modify yet."
            return turn

        explicit_memory = self.parse_explicit_memory_request(turn.user_text)
        if explicit_memory is not None:
            if self.is_incomplete_memory_note(explicit_memory):
                turn.assistant_text = "that's too half-finished. say the full thing you want me to remember."
                return turn

            self.conversation_store.add_explicit_memory(explicit_memory)
            turn.assistant_text = random.choice(
                [
                    f"say less I got that saved. {explicit_memory}",
                    f"bet I locked that in. {explicit_memory}",
                    f"aight I memorized that. {explicit_memory}",
                ]
            )
            return turn

        if TAKE_BALL_PATTERN.search(turn.user_text):
            if not self.ball_active:
                turn.assistant_text = "i don't even have a ball rn. you're tripping."
                return turn
            self.ball_active = False
            self.avatar.take_ball()
            turn.assistant_text = random.choice(
                [
                    "nah why you taking my ball. that's lowkey evil.",
                    "fine. the ball had no aura anyway.",
                    "ok but i was literally winning.",
                ]
            )
            return turn

        if GIVE_BALL_PATTERN.search(turn.user_text):
            self.ball_active = True
            self.avatar.give_ball()
            turn.assistant_text = random.choice(
                [
                    "ball? BALL. say less.",
                    "oh it's so on. this ball is mine now.",
                    "bet, watch me cook this ball.",
                ]
            )
            return turn

        wait_seconds = self.parse_wait_command(turn.user_text)
        if wait_seconds is not None:
            self.waiting_until = time.monotonic() + wait_seconds
            self.waiting_for_audio_offer = True
            self.waiting_audio_mode = False
            self.wake_requested = False
            self.key_monitor.consume_enter_press()
            duration_text = self.format_wait_duration(wait_seconds)
            turn.assistant_text = (
                f"aight, I'll be quiet for {duration_text}. You want music while I wait. Say yes or no."
            )
            return turn

        if self.waiting_for_audio_offer:
            decision = self.classify_audio_offer_reply(turn.user_text)
            self.waiting_for_audio_offer = False
            self.key_monitor.consume_enter_press()

            if decision["action"] == "no":
                turn.assistant_text = "bet no music then. hit Enter if you want me back early."
                return turn

            if not self.is_waiting_active():
                self.clear_wait_state()
                turn.assistant_text = "nah the timer already ran out, so that window's cooked."
                return turn

            selected_file = self.find_music_file(decision["query"])
            if not selected_file:
                turn.assistant_text = "damn I got nothing local that fits that. hit Enter if you want me back early."
                return turn

            print(f"Selected wait audio: {selected_file}")
            turn.assistant_text = "bet I'll throw this on while the timer runs. hit Enter if you want me back early."
            turn.pending_music_file = selected_file
            return turn

        if self.is_waiting_active():
            await self.stop_wait_music()
            self.clear_wait_state()

        return turn

    async def handle_idle(self) -> Optional[AssistantTurn]:
        pending_turn = self.consume_pending_system_line()
        if pending_turn is not None:
            return pending_turn

        if self.is_waiting_active() and not self.waiting_for_audio_offer:
            self.state = AssistantState.WAITING

            if not self.typed_text.empty():
                # Typing to Nori ends the wait; the text is handled as a normal turn next loop.
                await self.stop_wait_music()
                self.clear_wait_state()
                return None

            if self.key_monitor.consume_enter_press() or self.wake_requested:
                self.wake_requested = False
                await self.stop_wait_music()
                self.clear_wait_state()
                return AssistantTurn(
                    turn_id=self._next_turn_id(),
                    user_text="The user woke you early from wait mode. Re-enter the conversation yourself with one short in-character line.",
                    assistant_text=self.build_early_wake_line(),
                    system_generated=True,
                    track_interruption=False,
                )

            await asyncio.sleep(0.05)
            return None

        self.state = AssistantState.LISTENING

        if self.waiting_until is not None and time.monotonic() >= self.waiting_until:
            self.waiting_for_audio_offer = False
            self.waiting_until = None
            await self.stop_wait_music()
            return AssistantTurn(
                turn_id=self._next_turn_id(),
                user_text="The explicit wait timer ended. Re-enter the conversation yourself with one short in-character line.",
                assistant_text=self.build_timer_done_line(),
                system_generated=True,
                track_interruption=False,
            )

        try:
            user_text = await self.next_user_input()
        except sr.WaitTimeoutError:
            self.record_idle_timeout()

            if self.waiting_for_audio_offer:
                return None

            if self.waiting_until is not None and time.monotonic() >= self.waiting_until:
                self.waiting_for_audio_offer = False
                self.waiting_until = None
                await self.stop_wait_music()
                return AssistantTurn(
                    turn_id=self._next_turn_id(),
                    user_text="The explicit wait timer ended. Re-enter the conversation yourself with one short in-character line.",
                    assistant_text=self.build_timer_done_line(),
                    system_generated=True,
                    track_interruption=False,
                )

            if self.is_waiting_active():
                return None

            if not self.has_done_first_prompt:
                self.has_done_first_prompt = True
                return AssistantTurn(
                    turn_id=self._next_turn_id(),
                    user_text="The user has not spoken since startup. Start the conversation yourself with one short casual line in character.",
                    system_generated=True,
                    track_interruption=False,
                )

            print("Listening...")
            return None

        if not user_text:
            return None

        print(f"You said: {user_text}")
        self.record_user_activity()

        # Whisper adds capitals and punctuation ("Exit."), so normalize before matching.
        normalized_text = re.sub(r"[^a-z' ]", "", user_text.lower()).strip()
        if normalized_text in EXIT_WORDS:
            self.state = AssistantState.SHUTDOWN
            return None

        self.has_done_first_prompt = True

        return AssistantTurn(
            turn_id=self._next_turn_id(),
            user_text=user_text,
        )

    async def next_user_input(self) -> Optional[str]:
        """
        Wait for whichever comes first: speech from the mic or text typed in Nori's window.
        A mic capture can't be cancelled, so if typing wins, that capture keeps running and
        its result is thrown away when it finishes (it may have picked up Nori talking).
        Returns None for a discarded capture.
        """
        loop = asyncio.get_running_loop()
        if self._listen_future is None:
            self._listen_future = loop.run_in_executor(None, self.asr.listen_and_transcribe)
            self._listen_result_wanted = True

        typed_task = asyncio.ensure_future(self.typed_text.get())
        try:
            await asyncio.wait({self._listen_future, typed_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not typed_task.done():
                typed_task.cancel()

        if typed_task.done() and not typed_task.cancelled():
            self._listen_result_wanted = False
            return typed_task.result()

        listen_future, self._listen_future = self._listen_future, None
        if not self._listen_result_wanted:
            listen_future.exception()
            return None

        user_text = listen_future.result()
        if user_text:
            self.avatar.heard(user_text)
        return user_text

    @staticmethod
    def _strip_stage_directions(sentence: str) -> str:
        """Drop things like *backflip* that would otherwise be read out loud."""
        return re.sub(r"\s{2,}", " ", re.sub(r"\*[^*]*\*", "", sentence)).strip()

    def _split_first_sentence(self, buffer: str):
        for match in SENTENCE_BREAK_PATTERN.finditer(buffer):
            if match.end() >= MIN_SENTENCE_CHARS:
                return buffer[: match.end()].strip(), buffer[match.end() :]
        return None, buffer

    def _apply_reply_tag(self, buffer: str, turn: AssistantTurn) -> str:
        """Strip the leading [emotion, action] tag from a reply and send it to the avatar."""
        match = REPLY_TAG_PATTERN.match(buffer)
        if not match:
            return buffer

        emotion = match.group("emotion").lower()
        action = (match.group("action") or "none").lower()
        turn.emotion = emotion if emotion in AVATAR_EMOTIONS else "neutral"
        turn.action = action if action in AVATAR_ACTIONS and action != "none" else None
        self.avatar.set_mood(turn.emotion)
        return buffer[match.end() :]

    async def stream_reply_sentences(self, messages: List[Dict[str, str]], turn: AssistantTurn) -> AsyncIterator[str]:
        """Yield the reply one sentence at a time while the LLM is still writing it."""
        if not turn.system_generated and self.should_use_feedback(turn.user_text):
            # Spoken while the real reply is still being generated; never stored in history.
            yield self.pick_feedback_phrase(turn.user_text)

        buffer = ""
        tag_checked = False
        sentences: List[str] = []

        async for delta in self.llm.stream_reply(messages):
            buffer += delta
            if not tag_checked:
                # Local "thinking" models (e.g. Qwen) write <think>...</think> first. Never speak it.
                head = buffer.lstrip()
                if "<think>".startswith(head) or (head.startswith("<think>") and "</think>" not in head):
                    continue
                if head.startswith("<think>"):
                    buffer = head.split("</think>", 1)[1]
                    head = buffer.lstrip()
                if not head or (head.startswith("[") and "]" not in head and len(head) < 40):
                    continue
                buffer = self._apply_reply_tag(buffer, turn)
                tag_checked = True

            while True:
                sentence, buffer = self._split_first_sentence(buffer)
                if sentence is None:
                    break
                sentence = self._strip_stage_directions(sentence)
                if not sentence:
                    continue
                sentences.append(sentence)
                turn.assistant_text = " ".join(sentences)
                yield sentence

        if not tag_checked:
            head = buffer.lstrip()
            if head.startswith("<think>"):
                buffer = head.split("</think>", 1)[1] if "</think>" in head else ""
            buffer = self._apply_reply_tag(buffer, turn)
        rest = self._strip_stage_directions(buffer)
        if rest:
            sentences.append(rest)
            turn.assistant_text = " ".join(sentences)
            yield rest

    def _on_sentence_started(self, turn: AssistantTurn, sentence: str) -> None:
        if self.state != AssistantState.SPEAKING:
            self.state = AssistantState.SPEAKING
            if turn.action:
                self.avatar.play(turn.action)
        self.avatar.say(sentence)

    async def speak(self, turn: AssistantTurn, sentences: AsyncIterator[str]) -> AssistantTurn:
        self.tts_stop_event.clear()
        self.ui_interrupt.clear()
        await self.stop_wait_music()
        watch_task = asyncio.create_task(self._watch_for_manual_interrupt())

        try:
            finished_normally = await self.tts.speak_sentences(
                sentences,
                self.tts_stop_event,
                on_sentence=lambda sentence: self._on_sentence_started(turn, sentence),
            )
        finally:
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass

        if not turn.assistant_text:
            return turn

        if finished_normally:
            turn.delivered = True
            self.state = AssistantState.IDLE
            self.last_assistant_timestamp = time.monotonic()
            self.total_completed_assistant_turns += 1
            if turn.pending_music_file and self.is_waiting_active():
                await self.start_wait_music(turn.pending_music_file)
        else:
            turn = self._record_interruption(turn)
            print("Assistant speech interrupted.")

        return turn

    async def handle_speaking(self, turn: AssistantTurn) -> AssistantTurn:
        """Speak a reply that is already written (commands, timer lines)."""

        async def single_line():
            yield turn.assistant_text

        print(f"Assistant: {turn.assistant_text}")
        return await self.speak(turn, single_line())

    async def handle_reply(self, turn: AssistantTurn) -> AssistantTurn:
        """Generate a reply with the LLM and speak it as it streams in."""
        self.state = AssistantState.THINKING
        if turn.posture_state is None:
            turn.posture_state = self.get_posture_state()

        posture_note = ""
        if not self.is_waiting_active():
            posture_note = self.build_posture_note(turn.posture_state)

        messages = self.conversation_store.build_messages(turn.user_text, posture_note=posture_note)

        if self.pending_interrupt_context:
            cut_off_reply = self.pending_interrupt_context.get("cut_off_reply", "")
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": (
                        "The user interrupted your previous reply while you were speaking. "
                        "At the very start of this response, briefly acknowledge being cut off "
                        "in character with attitude and personality. Do not use brackets or "
                        "stage directions. Then continue with the actual answer. "
                        f"Your cut-off reply had started like this: {cut_off_reply!r}"
                    ),
                },
            )
            self.pending_interrupt_context = None

        turn = await self.speak(turn, self.stream_reply_sentences(messages, turn))
        print(f"Assistant: {turn.assistant_text}")
        return turn

    async def _watch_for_manual_interrupt(self) -> None:
        while not self.tts_stop_event.is_set():
            if self.key_monitor.right_shift_held or self.ui_interrupt.is_set():
                print("Manual interruption requested.")
                self.state = AssistantState.STOPPING
                self.tts_stop_event.set()
                return
            await asyncio.sleep(0.02)

    async def finalize_turn(self, turn: AssistantTurn) -> None:
        self.conversation_store.commit_turn(turn)
        await self.conversation_store.maybe_summarize()

        if self.state != AssistantState.SHUTDOWN:
            self.state = AssistantState.IDLE

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        self.typed_text = asyncio.Queue()
        if self.bridge is not None:
            self.bridge.attach(
                self._loop,
                on_text=self._on_typed_text,
                on_interrupt=self.ui_interrupt.set,
                on_poke=self._on_poke,
            )

        if LLM_BASE_URL:
            # Local models can take a while to load; start that now instead of on the first reply.
            self._warm_up_task = asyncio.create_task(self.llm.warm_up())

        print("Nori started. Say 'exit', 'quit', or 'stop' to end.")
        print("Hold Right Shift while Nori is speaking to interrupt (or press Escape / Stop in the window).")
        print("While wait mode is active, press Enter or click Nori to wake early.")

        self.key_monitor.start()

        try:
            while self.state != AssistantState.SHUTDOWN:
                try:
                    turn = await self.handle_idle()
                    if self.state == AssistantState.SHUTDOWN:
                        break
                    if turn is None:
                        continue

                    turn = await self.maybe_handle_special_turn(turn)
                    if turn is None:
                        continue

                    if turn.assistant_text:
                        turn = await self.handle_speaking(turn)
                    else:
                        turn = await self.handle_reply(turn)
                    await self.finalize_turn(turn)

                except KeyboardInterrupt:
                    print("Program terminated by user.")
                    self.state = AssistantState.SHUTDOWN
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None:
                        current.uncancel()
                    self.state = AssistantState.SHUTDOWN
                    break
                except Exception as error:
                    print(f"Unexpected error: {error}")
                    self.state = AssistantState.IDLE
        finally:
            self.state = AssistantState.SHUTDOWN
            try:
                await self.stop_wait_music()
                self.conversation_store.set_relationship_metrics(self.build_relationship_metrics())
                await self.conversation_store.persist_long_term_memory()
            except Exception as error:
                print(f"Warning: could not persist long-term memory: {error}")
            self.key_monitor.stop()
            self.playback.stop()
            print("Exiting assistant.")
