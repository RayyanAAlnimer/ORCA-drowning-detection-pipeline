from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import json
import queue
import threading
import urllib.error
import urllib.request


@dataclass(frozen=True)
class AlertEvent:
    track_id: int
    probability: float
    timestamp: float


class AlertController:
    """
    Lightweight alert dispatcher for confirmed Stage 6 events.

    Design goals:
      - fast/non-blocking handle_alert() path
      - optional console logging
      - optional sound alarm
      - optional webhook delivery
      - future extension via custom handlers for GPIO/Arduino/Jetson
      - local duplicate suppression as a second safety layer
    """

    def __init__(
        self,
        enable_console: bool = True,
        enable_sound: bool = False,
        webhook_url: Optional[str] = None,
        min_repeat_interval_seconds: float = 5.0,
        queue_size: int = 32,
        request_timeout_seconds: float = 1.0,
        custom_handlers: Optional[Iterable[Callable[[AlertEvent], None]]] = None,
    ):
        if min_repeat_interval_seconds < 0.0:
            raise ValueError("min_repeat_interval_seconds must be >= 0")
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        if request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be > 0")

        self.enable_console = bool(enable_console)
        self.enable_sound = bool(enable_sound)
        self.webhook_url = webhook_url
        self.min_repeat_interval_seconds = float(min_repeat_interval_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.custom_handlers = list(custom_handlers or [])

        self._queue: "queue.Queue[Optional[AlertEvent]]" = queue.Queue(maxsize=queue_size)
        self._last_alert_ts_by_track: dict[int, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def handle_alert(self, track_id: int, probability: float, timestamp: float) -> bool:
        """
        Queue an alert for asynchronous delivery.
        Returns True if accepted, False if suppressed or dropped.
        """
        event = AlertEvent(
            track_id=int(track_id),
            probability=float(probability),
            timestamp=float(timestamp),
        )

        with self._lock:
            last_ts = self._last_alert_ts_by_track.get(event.track_id)
            if last_ts is not None and (event.timestamp - last_ts) < self.min_repeat_interval_seconds:
                return False
            self._last_alert_ts_by_track[event.track_id] = event.timestamp

        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            if self.enable_console:
                print(
                    f"[AlertController] queue full, dropped alert for track={event.track_id}",
                    flush=True,
                )
            return False

    def shutdown(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=max(0.0, timeout))

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if event is None:
                    continue
                self._dispatch(event)
            finally:
                self._queue.task_done()

    def _dispatch(self, event: AlertEvent) -> None:
        if self.enable_console:
            self._emit_console(event)

        if self.enable_sound:
            self._emit_sound()

        if self.webhook_url:
            self._emit_webhook(event)

        for handler in self.custom_handlers:
            try:
                handler(event)
            except Exception as exc:
                if self.enable_console:
                    print(f"[AlertController] custom handler failed: {exc}", flush=True)

    def _emit_console(self, event: AlertEvent) -> None:
        print(
            f"[ALERT] track={event.track_id} smooth={event.probability:.3f} ts={event.timestamp:.3f}",
            flush=True,
        )

    def _emit_sound(self) -> None:
        print("\a", end="", flush=True)

    def _emit_webhook(self, event: AlertEvent) -> None:
        payload = json.dumps(
            {
                "track_id": event.track_id,
                "probability": event.probability,
                "timestamp": event.timestamp,
                "source": "orca-drowning-detection-pipeline",
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_seconds):
                pass
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            if self.enable_console:
                print(f"[AlertController] webhook failed: {exc}", flush=True)
