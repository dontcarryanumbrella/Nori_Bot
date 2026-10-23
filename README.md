# Nori

A voice assistant that lives in a small window on your screen as a sketchy black cat. You talk to him out loud (or type), he answers out loud in slang, and his face, ears, and tail react to what he's saying.

## Features

- **Voice conversation**: local speech-to-text with faster-whisper, replies spoken with text-to-speech.
- **Fast replies**: the reply streams in sentence by sentence, so Nori starts talking before the full answer is written.
- **Animated avatar**: a hand-drawn-style cat rendered in SVG, with spring-based transitions between 12 emotions and tricks like backflips. The language model picks his emotion and action for each reply.
- **Click interactions**: pat his head, poke him, grab his tail, flick his ears.
- **Voice commands**: give him a ball to play with, set a wait timer with music, ask him to remember things.
- **Memory**: summarizes long conversations and keeps long-term notes about you between sessions.
- **Swappable models**: works with OpenAI or any OpenAI-compatible server, including a local model in LM Studio. Voice can be a free Microsoft Edge voice or OpenAI TTS.

## Setup

Requires Windows and Python 3.13.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Copy `botmodel/.env.example` to `botmodel/.env` and fill in either:

- `OPENAI_API_KEY` to use OpenAI, or
- the `LLM_*` lines to use a local model in LM Studio (start its Local API server first).

The default voice is a free Edge voice and needs no key.

## Run

```bash
cd botmodel
../.venv/Scripts/python.exe base2.py
```

Add `--no-avatar` to run in the terminal without the window.

## Talking to Nori

| Say or type | What happens |
|---|---|
| "exit", "quit", "stop" | Saves memory and closes |
| "wait 5 minutes" | Goes quiet, offers music from `botmodel/music/` (Enter or a click wakes him early) |
| "remember that ..." / "change it to ..." | Saves or edits a note about you |
| "here's a ball" / "take the ball" | Starts or ends a game of ball |
| "do a backflip" | He might do it, if he feels like it |

Hold Right Shift, press Escape, or click Stop to interrupt him.

## Project layout

| File | Purpose |
|---|---|
| `botmodel/base2.py` | Entry point; runs the window on the main thread and the bot on a worker thread |
| `botmodel/assistant_coordinator.py` | The listen, think, speak loop, commands, and avatar state |
| `botmodel/audio_workers.py` | Microphone and Whisper, streaming LLM, TTS, playback |
| `botmodel/conversation_store.py` | Prompt building, summaries, long-term memory |
| `botmodel/assistant_config.py` | Personality prompt, models, and settings |
| `botmodel/avatar_window.py` | pywebview window and the Python/JavaScript bridge |
| `botmodel/avatar/nori.html` | The cat: SVG drawing, spring animation rig, clicks, ball physics |
