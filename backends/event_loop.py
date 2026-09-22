from __future__ import annotations

import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="engine-loop", daemon=True).start()
        return _loop


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result()
