import asyncio
import os
from typing import Dict, List

from openai import OpenAI

from assistant_config import (
    AVATAR_PROMPT,
    LLM_EXTRA_BODY,
    LLM_MODEL,
    MAX_RECENT_MESSAGES,
    SUMMARIZE_AFTER_TOTAL_MESSAGES,
    SYSTEM_PROMPT,
)
from assistant_types import AssistantTurn


def fold_system_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge all system messages into one leading system message.
    Many local chat templates (Qwen, Llama) reject system messages anywhere but first.
    A system note that appears mid-conversation is attached to the next user message instead.
    """
    leading: List[str] = []
    pending: List[str] = []
    dialogue: List[Dict[str, str]] = []

    for message in messages:
        if message["role"] == "system":
            (pending if dialogue else leading).append(message["content"].strip())
            continue
        if pending and message["role"] == "user":
            note = "[Note: " + " ".join(pending) + "]\n"
            message = {"role": "user", "content": note + message["content"]}
            pending = []
        dialogue.append(message)

    leading.extend(pending)
    folded = [{"role": "system", "content": "\n\n".join(leading)}] if leading else []
    return folded + dialogue


class ConversationStore:
    def __init__(self, client: OpenAI, memory_path: str):
        self.client = client
        self.memory_path = memory_path
        self.conversation: List[Dict[str, str]] = []
        self.conversation_summary: str = ""
        self.long_term_memory: str = self._read_memory_file()
        self.relationship_metrics: str = ""
        self.explicit_memory_notes: List[str] = []

    def _read_memory_file(self) -> str:
        try:
            with open(self.memory_path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except FileNotFoundError:
            return ""
        except OSError:
            return ""

    def _write_memory_file(self, content: str) -> None:
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            handle.write(content.strip())

    def set_relationship_metrics(self, metrics_text: str) -> None:
        self.relationship_metrics = metrics_text.strip()

    def add_explicit_memory(self, memory_text: str) -> None:
        cleaned = memory_text.strip()
        if not cleaned:
            return
        if cleaned not in self.explicit_memory_notes:
            self.explicit_memory_notes.append(cleaned)

    def replace_last_explicit_memory(self, memory_text: str) -> bool:
        cleaned = memory_text.strip()
        if not cleaned or not self.explicit_memory_notes:
            return False
        self.explicit_memory_notes[-1] = cleaned
        return True

    def build_messages(self, user_input: str, posture_note: str = "") -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": AVATAR_PROMPT},
        ]

        if self.long_term_memory.strip():
            messages.append(
                {
                    "role": "system",
                    "content": f"Long-term memory about the user: {self.long_term_memory}",
                }
            )

        if self.explicit_memory_notes:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The user explicitly asked you to remember these notes this session: "
                        + " | ".join(self.explicit_memory_notes)
                    ),
                }
            )

        if self.conversation_summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation: {self.conversation_summary}",
                }
            )

        if posture_note.strip():
            messages.append({"role": "system", "content": posture_note})

        messages.extend(self.conversation[-MAX_RECENT_MESSAGES:])
        messages.append({"role": "user", "content": user_input})
        return messages

    def commit_turn(self, turn: AssistantTurn) -> None:
        if not turn.assistant_text:
            return

        if turn.system_generated and turn.interrupted and not turn.track_interruption:
            return

        if not turn.system_generated:
            self.conversation.append({"role": "user", "content": turn.user_text})

        if turn.interrupted:
            self.conversation.append({"role": "assistant", "content": turn.assistant_text})
            self.conversation.append(
                {
                    "role": "system",
                    "content": (
                        "The assistant's previous reply was cut off by the user before it "
                        "finished speaking. Treat that reply as interrupted, not fully delivered."
                    ),
                }
            )
        elif turn.delivered:
            self.conversation.append({"role": "assistant", "content": turn.assistant_text})

    async def maybe_summarize(self) -> None:
        if len(self.conversation) <= SUMMARIZE_AFTER_TOTAL_MESSAGES:
            return

        old_messages = self.conversation[:-MAX_RECENT_MESSAGES]
        recent_messages = self.conversation[-MAX_RECENT_MESSAGES:]

        if not old_messages:
            return

        summary_input = [
            {
                "role": "system",
                "content": """
Summarize the conversation so far for future assistant memory.
Preserve:
- user preferences
- ongoing goals
- important facts

Be compact but specific.
""".strip(),
            }
        ]

        if self.conversation_summary.strip():
            summary_input.append(
                {
                    "role": "system",
                    "content": f"Existing summary: {self.conversation_summary}",
                }
            )

        summary_input.extend(old_messages)

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=fold_system_messages(summary_input),
                temperature=0.2,
                extra_body=LLM_EXTRA_BODY,
            ),
        )

        self.conversation_summary = response.choices[0].message.content.strip()
        self.conversation = recent_messages

    async def persist_long_term_memory(self) -> None:
        if not self.conversation and not self.conversation_summary.strip():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_memory_file, self.long_term_memory)
            return

        memory_messages = [
            {
                "role": "system",
                "content": """
Rewrite the assistant's long-term memory for the user.
Keep only durable, high-value facts that should survive across sessions.

Include a short affinity section that reflects how close the assistant feels to the user.
Use the session metrics as signal, but also use the assistant's own in-character evaluation.
Keep it stable and readable, not dramatic.

Preserve:
- name or nickname
- recurring preferences
- important dislikes
- ongoing goals
- stable constraints
- affinity level and short reason
- any explicit "remember this" notes the user directly asked you to keep, even if they are niche game codes or temporary-use info

Exclude:
- insults
- transient arguments
- one-off jokes
- raw transcript quotes
- temporary moods

Return plain text only, compact and specific.
Use simple labeled sections.
""".strip(),
            }
        ]

        if self.long_term_memory.strip():
            memory_messages.append(
                {
                    "role": "system",
                    "content": f"Existing long-term memory: {self.long_term_memory}",
                }
            )

        if self.conversation_summary.strip():
            memory_messages.append(
                {
                    "role": "system",
                    "content": f"Session summary: {self.conversation_summary}",
                }
            )

        if self.explicit_memory_notes:
            memory_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Explicit remember requests from this session: "
                        + " | ".join(self.explicit_memory_notes)
                    ),
                }
            )

        if self.relationship_metrics:
            memory_messages.append(
                {
                    "role": "system",
                    "content": f"Session relationship metrics: {self.relationship_metrics}",
                }
            )

        memory_messages.extend(self.conversation[-MAX_RECENT_MESSAGES:])

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=fold_system_messages(memory_messages),
                temperature=0.2,
                extra_body=LLM_EXTRA_BODY,
            ),
        )

        self.long_term_memory = response.choices[0].message.content.strip()
        await loop.run_in_executor(None, self._write_memory_file, self.long_term_memory)
        print(f"Long-term memory saved to {self.memory_path}")
