import asyncio
import io
import os
import tempfile
import threading
import time
from typing import AsyncIterator, Callable, Dict, List

import edge_tts
import pygame
import soundfile as sf
import speech_recognition as sr
from faster_whisper import WhisperModel
from keyboard import is_pressed
from openai import AsyncOpenAI

from assistant_config import (
    EDGE_TTS_PITCH,
    EDGE_TTS_RATE,
    EDGE_TTS_VOICE,
    LISTEN_TIMEOUT,
    LLM_EXTRA_BODY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    PAUSE_THRESHOLD,
    PHRASE_TIME_LIMIT,
    TTS_INSTRUCTIONS,
    TTS_MODEL,
    TTS_VOICE,
)
from conversation_store import fold_system_messages


class BlockingASRWorker:
    """
    Simple blocking microphone capture.
    No queue, no streaming VAD, no barge-in.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self._calibrated = False
        self.whisper = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

    def listen_and_transcribe(self) -> str:
        with sr.Microphone(sample_rate=16000) as source:
            # Calibrate once; the recognizer keeps adjusting its energy threshold on its own after that.
            if not self._calibrated:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                self._calibrated = True
            print("Listening...")
            audio = self.recognizer.listen(
                source,
                timeout=LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT,
            )

        return self.transcribe_audio_data(audio)

    def transcribe_audio_data(self, audio: sr.AudioData) -> str:
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        audio_array, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        segments, _ = self.whisper.transcribe(
            audio_array,
            language="en",
            beam_size=1,
            vad_filter=False,
        )

        text = "".join(segment.text for segment in segments).strip()
        return text


class LLMWorker:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def stream_reply(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=fold_system_messages(messages),
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
            extra_body=LLM_EXTRA_BODY,
        )
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await stream.close()

    async def warm_up(self) -> None:
        """Send a tiny request so a local model loads while everything else starts up."""
        try:
            await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                extra_body=LLM_EXTRA_BODY,
            )
        except Exception as error:
            print(f"Warning: could not reach the chat model ({error}). Is LM Studio's server running?")


class PlaybackController:
    def __init__(self):
        pygame.mixer.init()
        self._lock = threading.Lock()
        self._is_playing = False

    def play_file_blocking(self, filename: str, stop_event: threading.Event) -> bool:
        with self._lock:
            self._is_playing = True

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if stop_event.is_set():
                    pygame.mixer.music.stop()
                    time.sleep(0.05)
                    try:
                        pygame.mixer.music.unload()
                    except pygame.error:
                        pass
                    return False
                time.sleep(0.03)

            time.sleep(0.05)
            try:
                pygame.mixer.music.unload()
            except pygame.error:
                pass
            return True

        finally:
            with self._lock:
                self._is_playing = False

    def stop(self) -> None:
        with self._lock:
            if self._is_playing:
                pygame.mixer.music.stop()
                time.sleep(0.05)
                try:
                    pygame.mixer.music.unload()
                except pygame.error:
                    pass
                self._is_playing = False


class EdgeSpeech:
    """Free Microsoft Edge read-aloud voices. No API key needed."""

    async def synthesize(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH)
        chunks = [chunk["data"] async for chunk in communicate.stream() if chunk["type"] == "audio"]
        return b"".join(chunks)


class OpenAISpeech:
    """OpenAI text-to-speech, or any server with the same /audio/speech API."""

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def synthesize(self, text: str) -> bytes:
        audio = await self.client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            instructions=TTS_INSTRUCTIONS,
            input=text,
        )
        return await audio.aread()


class TTSWorker:
    def __init__(self, speech, playback_controller: PlaybackController):
        self.speech = speech
        self.playback = playback_controller

    async def _generate_tts_file(self, text: str) -> str:
        data = await self.speech.synthesize(text)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as handle:
            handle.write(data)
            return handle.name

    def _safe_remove(self, filename: str, retries: int = 10, delay: float = 0.1) -> None:
        for _ in range(retries):
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                return
            except PermissionError:
                time.sleep(delay)
            except OSError:
                time.sleep(delay)

        print(f"Warning: could not delete temp audio file yet: {filename}")

    def _discard_when_done(self, task: "asyncio.Task[str]") -> None:
        def cleanup(done: "asyncio.Task[str]") -> None:
            if not done.cancelled() and done.exception() is None:
                self._safe_remove(done.result())

        task.add_done_callback(cleanup)

    @staticmethod
    async def _wait_unless_stopped(future: asyncio.Future, stop_event: threading.Event) -> bool:
        while not future.done():
            if stop_event.is_set():
                return False
            await asyncio.wait({future}, timeout=0.03)
        return True

    async def speak_sentences(
        self,
        sentences: AsyncIterator[str],
        stop_event: threading.Event,
        on_sentence: Callable[[str], None],
    ) -> bool:
        """
        Speak sentences as they arrive.
        Each sentence is sent to TTS as soon as it exists, so the next one is usually
        ready by the time the current one finishes playing.
        Returns False if stop_event interrupted playback.
        """
        loop = asyncio.get_running_loop()
        pending: "asyncio.Queue" = asyncio.Queue()

        async def produce() -> None:
            try:
                async for sentence in sentences:
                    if stop_event.is_set():
                        break
                    await pending.put((sentence, asyncio.create_task(self._generate_tts_file(sentence))))
            finally:
                pending.put_nowait(None)

        producer = asyncio.create_task(produce())
        finished_normally = True
        try:
            while True:
                next_item = asyncio.ensure_future(pending.get())
                if not await self._wait_unless_stopped(next_item, stop_event):
                    next_item.cancel()
                    finished_normally = False
                    break

                item = next_item.result()
                if item is None:
                    if producer.done() and not producer.cancelled() and producer.exception():
                        raise producer.exception()
                    break

                sentence, tts_task = item
                if not await self._wait_unless_stopped(tts_task, stop_event):
                    self._discard_when_done(tts_task)
                    finished_normally = False
                    break

                filename = tts_task.result()
                on_sentence(sentence)
                print("Speaking...")
                try:
                    played = await loop.run_in_executor(
                        None,
                        self.playback.play_file_blocking,
                        filename,
                        stop_event,
                    )
                finally:
                    await loop.run_in_executor(None, self._safe_remove, filename)

                if not played:
                    finished_normally = False
                    break
        finally:
            producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):
                pass
            while not pending.empty():
                item = pending.get_nowait()
                if item is not None:
                    self._discard_when_done(item[1])
                    item[1].cancel()

        return finished_normally


class KeyboardMonitor:
    def __init__(self, poll_interval: float = 0.02):
        self.poll_interval = poll_interval
        self.running = False
        self.right_shift_held = False
        self.enter_pending = False
        self._enter_was_held = False
        self._thread = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def consume_enter_press(self) -> bool:
        if not self.enter_pending:
            return False
        self.enter_pending = False
        return True

    def _run(self) -> None:
        while self.running:
            try:
                self.right_shift_held = is_pressed("right shift")
                enter_held = is_pressed("enter")
                if enter_held and not self._enter_was_held:
                    self.enter_pending = True
                self._enter_was_held = enter_held
            except Exception:
                self.right_shift_held = False
                self._enter_was_held = False
            time.sleep(self.poll_interval)
