import math
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

PER_IP_MESSAGE = "Too many questions in the last minute. Please wait a moment and try again."
DAILY_MESSAGE = "The demo has reached its daily limit. Please try again tomorrow."

WINDOW_SECONDS = 60


class RateLimited(Exception):
    """Raised when a question is over a limit. `limit` says which one ("per_ip" or
    "daily") so the frontend can tell them apart, and `retry_after` is how many
    seconds until a question would be accepted again."""

    def __init__(self, limit: str, message: str, retry_after: int):
        super().__init__(message)
        self.limit = limit
        self.message = message
        self.retry_after = retry_after


class RateLimiter:
    """Two limits for the public demo: a sliding one-minute window per client IP,
    and one count of questions per UTC day across everyone, which is the hard
    backstop on spend. State lives in memory, which is enough because the backend
    runs as a single container; a restart resets both counts."""

    def __init__(self, per_minute: int, per_day: int, clock=time.time):
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()  # /ask runs in a thread pool, so guard the counts
        self._recent: dict[str, deque[float]] = {}
        self._day: str | None = None
        self._day_count = 0

    def check(self, client: str) -> None:
        """Count one question from `client`, or raise RateLimited without counting it."""
        now = self._clock()
        with self._lock:
            today = datetime.fromtimestamp(now, timezone.utc).date()
            if today.isoformat() != self._day:
                self._day, self._day_count = today.isoformat(), 0
            if self._day_count >= self.per_day:
                midnight = datetime.combine(today + timedelta(days=1), datetime.min.time(), timezone.utc)
                raise RateLimited("daily", DAILY_MESSAGE, math.ceil(midnight.timestamp() - now))

            recent = self._recent.setdefault(client, deque())
            while recent and now - recent[0] >= WINDOW_SECONDS:
                recent.popleft()
            if len(recent) >= self.per_minute:
                raise RateLimited("per_ip", PER_IP_MESSAGE, math.ceil(WINDOW_SECONDS - (now - recent[0])))

            recent.append(now)
            self._day_count += 1
            self._forget_idle_clients(now)

    def _forget_idle_clients(self, now: float) -> None:
        # drop clients with nothing in the window, so the map doesn't grow forever
        if len(self._recent) > 1000:
            self._recent = {
                ip: times for ip, times in self._recent.items()
                if times and now - times[-1] < WINDOW_SECONDS
            }
