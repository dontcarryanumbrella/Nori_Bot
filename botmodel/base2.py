import argparse
import asyncio
import threading

from assistant_coordinator import AssistantCoordinator


async def main():
    coordinator = AssistantCoordinator()
    await coordinator.run()


def run_with_avatar() -> None:
    """
    The window has to own the main thread, so the bot's event loop runs on a worker thread.
    Closing the window stops the bot; saying "exit" closes the window.
    """
    import webview

    from avatar_window import AvatarWindow, NoriBridge

    bridge = NoriBridge()
    avatar = AvatarWindow(bridge)
    window_closed = threading.Event()
    holder = {}

    async def run_bot():
        coordinator = AssistantCoordinator(avatar=avatar, bridge=bridge)
        holder["coordinator"] = coordinator
        if not window_closed.is_set():
            await coordinator.run()

    def bot_thread_main():
        try:
            asyncio.run(run_bot())
        except Exception as error:
            print(f"Assistant crashed: {error}")
        finally:
            avatar.close()

    bot_thread = threading.Thread(target=bot_thread_main, daemon=True)
    webview.start(bot_thread.start)

    window_closed.set()
    coordinator = holder.get("coordinator")
    if coordinator is not None:
        coordinator.request_shutdown()
    # Give the bot time to save long-term memory before the process exits.
    bot_thread.join(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nori, a voice assistant cat.")
    parser.add_argument("--no-avatar", action="store_true", help="run in the terminal without Nori's window")
    args = parser.parse_args()

    try:
        if args.no_avatar:
            asyncio.run(main())
        else:
            run_with_avatar()
    except KeyboardInterrupt:
        pass
