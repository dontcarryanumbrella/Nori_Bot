import asyncio
import json
import os
import queue
import threading
from typing import Callable, Optional

from assistant_config import ASSISTANT_NAME


AVATAR_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatar", "nori.html")


class NullAvatar:
    """Stand-in used when the bot runs without a window (--no-avatar)."""

    def set_activity(self, name: str) -> None: ...
    def set_mood(self, name: str) -> None: ...
    def play(self, name: str) -> None: ...
    def say(self, text: str) -> None: ...
    def heard(self, text: str) -> None: ...
    def give_ball(self) -> None: ...
    def take_ball(self) -> None: ...
    def close(self) -> None: ...


class NoriBridge:
    """
    Exposed to the page as window.pywebview.api.
    Calls arrive on pywebview's threads, so each one is handed to the bot's event loop.
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_text: Optional[Callable[[str], None]] = None
        self._on_interrupt: Optional[Callable[[], None]] = None
        self._on_poke: Optional[Callable[[str], None]] = None

    def attach(
        self,
        loop: asyncio.AbstractEventLoop,
        on_text: Callable[[str], None],
        on_interrupt: Callable[[], None],
        on_poke: Callable[[str], None],
    ) -> None:
        self._loop = loop
        self._on_text = on_text
        self._on_interrupt = on_interrupt
        self._on_poke = on_poke

    def submit_text(self, text: str) -> None:
        if isinstance(text, str) and text.strip():
            self._post(self._on_text, text.strip())

    def interrupt(self) -> None:
        self._post(self._on_interrupt)

    def poked(self, part: str) -> None:
        self._post(self._on_poke, str(part))

    def _post(self, callback, *args) -> None:
        loop = self._loop
        if loop is None or callback is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            pass


class AvatarWindow:
    """
    Small always-on-top window that shows Nori.
    Python -> JS calls go through a queue on their own thread, so the bot loop never waits on the GUI.
    """

    def __init__(self, bridge: NoriBridge):
        import webview

        self.window = webview.create_window(
            ASSISTANT_NAME,
            url=AVATAR_HTML,
            js_api=bridge,
            width=320,
            height=400,
            min_size=(220, 260),
            on_top=True,
            background_color="#DCDBD6",
        )
        self._calls: "queue.Queue[Optional[str]]" = queue.Queue()
        self._ready = threading.Event()
        self._activity: Optional[str] = None
        self.window.events.loaded += self._ready.set
        threading.Thread(target=self._dispatch, daemon=True).start()

    def _dispatch(self) -> None:
        self._ready.wait()
        while True:
            script = self._calls.get()
            if script is None:
                return
            try:
                self.window.run_js(script)
            except Exception:
                pass

    def _call(self, function: str, *args) -> None:
        arguments = ", ".join(json.dumps(arg) for arg in args)
        self._calls.put(f"window.nori && window.nori.{function}({arguments})")

    def set_activity(self, name: str) -> None:
        if name != self._activity:
            self._activity = name
            self._call("setActivity", name)

    def set_mood(self, name: str) -> None:
        self._call("setMood", name)

    def play(self, name: str) -> None:
        self._call("play", name)

    def say(self, text: str) -> None:
        self._call("say", text)

    def heard(self, text: str) -> None:
        self._call("heard", text)

    def give_ball(self) -> None:
        self._call("giveBall")

    def take_ball(self) -> None:
        self._call("takeBall")

    def close(self) -> None:
        self._calls.put(None)
        try:
            self.window.destroy()
        except Exception:
            pass
