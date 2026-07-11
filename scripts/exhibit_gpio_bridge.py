#!/usr/bin/env python3
"""전시 물리 전화기 ↔ callone 이벤트 브리지 (라즈베리파이, EXHIBIT_PLAN §5).

역할 두 가지:
  1) 후크 스위치(GPIO 입력) → POST /api/exhibit/event {hook_up|hook_down}
     — 수화기 들면 통화 시작/수신, 내려놓으면 즉시 소멸(물리적 통제권)
  2) WS /ws/exhibit/events 구독 → ring_start/ring_stop → 벨 릴레이(GPIO 출력) 구동
     — 키오스크가 벨 단계에 들어가면 진짜 벨(솔레노이드)이 울린다

하드웨어 없이도 시험 가능: gpiozero 미설치면 **키보드 시뮬레이션 모드**
(u=hook_up, d=hook_down, q=종료 / 벨 이벤트는 콘솔 출력).

사용:
  python scripts/exhibit_gpio_bridge.py --server http://127.0.0.1:8000 \
      --hook-pin 17 --bell-pin 27
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

def post_event(server: str, event: str) -> None:
    req = urllib.request.Request(
        f"{server.rstrip('/')}/api/exhibit/event",
        data=json.dumps({"event": event}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            r.read()
        print(f"[bridge] → {event}")
    except Exception as e:  # noqa: BLE001
        print(f"[bridge] 이벤트 전송 실패({event}): {e}", file=sys.stderr)


class Bell:
    """벨 구동 — GPIO 릴레이 or 콘솔 폴백. 1s 울림 / 2s 쉼 반복."""

    def __init__(self, pin: int | None):
        self.dev = None
        if pin is not None:
            try:
                from gpiozero import OutputDevice  # type: ignore
                self.dev = OutputDevice(pin)
            except Exception:  # noqa: BLE001
                print("[bridge] gpiozero 없음 — 벨은 콘솔 출력으로 대체")
        self._task: asyncio.Task | None = None

    async def _ring_loop(self):
        try:
            while True:
                if self.dev:
                    self.dev.on()
                else:
                    print("[bridge] 따르릉—")
                await asyncio.sleep(1.0)
                if self.dev:
                    self.dev.off()
                await asyncio.sleep(2.0)
        finally:
            if self.dev:
                self.dev.off()

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._ring_loop())

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        if self.dev:
            self.dev.off()


async def listen_events(server: str, bell: Bell):
    """서버 이벤트 구독(ring_start/ring_stop) — 끊기면 3s 후 재접속."""
    try:
        import websockets  # type: ignore  # uvicorn 의존으로 대개 설치돼 있음
    except ImportError:
        print("[bridge] websockets 미설치 — 벨 구독 생략(pip install websockets)")
        return
    url = server.rstrip("/").replace("http", "ws", 1) + "/ws/exhibit/events"
    while True:
        try:
            async with websockets.connect(url) as ws:
                print(f"[bridge] 이벤트 구독: {url}")
                async for raw in ws:
                    try:
                        ev = json.loads(raw).get("event")
                    except Exception:  # noqa: BLE001
                        continue
                    if ev == "ring_start":
                        bell.start()
                    elif ev == "ring_stop":
                        bell.stop()
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] 구독 끊김({e}) — 3s 후 재접속")
            await asyncio.sleep(3)


async def watch_hook(server: str, pin: int | None):
    """후크 스위치 감시 — GPIO or 키보드 시뮬레이션."""
    try:
        from gpiozero import Button  # type: ignore
        if pin is None:
            raise ImportError
        hook = Button(pin, bounce_time=0.05)
        loop = asyncio.get_running_loop()
        hook.when_pressed = lambda: loop.call_soon_threadsafe(post_event, server, "hook_up")
        hook.when_released = lambda: loop.call_soon_threadsafe(post_event, server, "hook_down")
        print(f"[bridge] 후크 GPIO{pin} 감시 중")
        while True:
            await asyncio.sleep(3600)
    except ImportError:
        print("[bridge] GPIO 없음 — 키보드 시뮬레이션: u=hook_up, d=hook_down, q=종료")
        loop = asyncio.get_running_loop()
        while True:
            line = (await loop.run_in_executor(None, sys.stdin.readline)).strip().lower()
            if line == "u":
                post_event(server, "hook_up")
            elif line == "d":
                post_event(server, "hook_down")
            elif line == "q":
                return


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--hook-pin", type=int, default=None)
    ap.add_argument("--bell-pin", type=int, default=None)
    a = ap.parse_args()
    bell = Bell(a.bell_pin)
    await asyncio.gather(listen_events(a.server, bell), watch_hook(a.server, a.hook_pin))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
