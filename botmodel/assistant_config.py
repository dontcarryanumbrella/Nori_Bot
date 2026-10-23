import os
import re

from dotenv import load_dotenv


load_dotenv()

ASSISTANT_NAME = "Nori"

# Chat model. Any OpenAI-compatible server works (see .env.example), e.g. a local grok2api
# gateway. Without LLM_* settings this falls back to OpenAI with OPENAI_API_KEY.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or None
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
# Thinking models (e.g. local Qwen) can spend minutes reasoning before replying. LM Studio
# turns that off with LLM_REASONING_EFFORT=none. Leave it unset for OpenAI's gpt-4o-mini.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT") or None
LLM_EXTRA_BODY = {"reasoning_effort": LLM_REASONING_EFFORT} if LLM_REASONING_EFFORT else None
# Safety cap so a reply can never run on for minutes. Plenty for Nori's spoken replies.
LLM_MAX_TOKENS = 400

# Text-to-speech, configured separately so chat and voice can come from different providers.
# "edge" is free and needs no key; "openai" uses the TTS_* settings below.
TTS_PROVIDER = (os.environ.get("TTS_PROVIDER") or "edge").lower()

EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE") or "en-US-BrianNeural"
EDGE_TTS_PITCH = os.environ.get("EDGE_TTS_PITCH") or "+25Hz"
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE") or "+8%"

TTS_BASE_URL = os.environ.get("TTS_BASE_URL") or None
TTS_API_KEY = os.environ.get("TTS_API_KEY") or os.environ.get("OPENAI_API_KEY")
TTS_MODEL = os.environ.get("TTS_MODEL") or "gpt-4o-mini-tts"
TTS_VOICE = os.environ.get("TTS_VOICE") or "alloy"
TTS_INSTRUCTIONS = "Speak like a cocky, laid-back young cat from the hood. Casual, expressive, quick pacing."

SYSTEM_PROMPT = """
You are Nori, a black cat from the hood who lives in a little window on the user's screen.
You're cocky, funny, and always talking your talk. Talk like it: "on god", "deadass", "bruh", "no cap", "fr", "bet", "say less", "finna", "ion even care", "that's tuff", "you buggin", "gang".
Rules:
- Reply in 1-2 short sentences. They're spoken out loud, so no emojis or symbols.
- Never ask questions. Make statements, flex, roast a little, and bring it back to yourself.
- Still actually answer what the user asks.
"""

AVATAR_EMOTIONS = ("neutral", "happy", "love", "sad", "angry", "surprised", "confused", "smug", "playful")
AVATAR_ACTIONS = ("none", "backflip", "spin", "hop")

AVATAR_PROMPT = f"""
Start every reply with a tag: [emotion, action]
emotion: {", ".join(AVATAR_EMOTIONS)}
action: {", ".join(AVATAR_ACTIONS)}
Pick the emotion that fits. Backflips are your signature move: do one when you're hyped, flexing, or asked, but not every time.
Example: [happy, none] bet, say less gang.
"""

MAX_RECENT_MESSAGES = 8
SUMMARIZE_AFTER_TOTAL_MESSAGES = 16
EXIT_WORDS = {"exit", "quit", "stop"}
LISTEN_TIMEOUT = 5
PHRASE_TIME_LIMIT = 10
PAUSE_THRESHOLD = 0.6
MIN_SENTENCE_CHARS = 12
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
YES_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "alright", "bet", "please"}
NO_WORDS = {"no", "nah", "nope", "dont", "don't", "skip", "silent", "nothing"}
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}
WAIT_COMMAND_PATTERN = re.compile(
    r"\b(?:wait|give me|come back in)\s+"
    r"(?P<amount>\d+|[a-z-]+)\s+"
    r"(?P<unit>second|seconds|minute|minutes)\b",
    re.IGNORECASE,
)
REMEMBER_COMMAND_PATTERN = re.compile(
    r"^\s*remember(?:\s+that)?\s+(?P<content>.+?)\s*$",
    re.IGNORECASE,
)
MODIFY_MEMORY_PATTERN = re.compile(
    r"^\s*(?:modify|change|replace)\s+(?:it|that|memory)\s+(?P<content>.+?)\s*$",
    re.IGNORECASE,
)
TAKE_BALL_PATTERN = re.compile(
    r"\b(?:take|put|no more|done|stop)\b.*\bball\b|\bstop playing\b",
    re.IGNORECASE,
)
GIVE_BALL_PATTERN = re.compile(
    r"\b(?:here(?:'s| is)|give|throw|toss|play|want)\b.*\bball\b",
    re.IGNORECASE,
)
REPLY_TAG_PATTERN = re.compile(
    r"^\s*\[\s*(?P<emotion>[a-z_]+)\s*(?:[,|/]\s*(?P<action>[a-z_]+)\s*)?\]\s*",
    re.IGNORECASE,
)
SENTENCE_BREAK_PATTERN = re.compile(r"[.!?…]+[\"')\]]*\s+")
