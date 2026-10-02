"""Host clock -> Opal (sensor) clock.

Events we generate on the host (audio cues, key presses) must be stamped in
the Opals' clock — the epoch-microsecond ``Time`` of every sample — or they
cannot be lined up with the data afterwards. Every received sample gives one
observation ``host_arrival - sample_time`` = transport latency + clock offset.
The MINIMUM over a sliding window is the least-delayed packet, i.e. the best
estimate of the pure clock offset; ``sensor_now()`` subtracts it from the
host clock. Error ≈ the fastest packet's latency (a few ms-tens of ms with
the AP; ~one poll interval for a replay).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class SensorClock:
    def __init__(self, window_s: float = 10.0):
        self.window_s = window_s
        self._obs: deque = deque()          # (host_s, offset_us)
        self._min: Optional[float] = None

    def observe(self, newest_t_us: int, host_s: Optional[float] = None):
        host_s = time.time() if host_s is None else host_s
        off = host_s * 1e6 - float(newest_t_us)
        self._obs.append((host_s, off))
        while self._obs and self._obs[0][0] < host_s - self.window_s:
            self._obs.popleft()
        self._min = min(o for _, o in self._obs)

    @property
    def ready(self) -> bool:
        return self._min is not None

    def sensor_now(self, host_s: Optional[float] = None) -> int:
        """Current time in the Opal clock [epoch µs]."""
        host_s = time.time() if host_s is None else host_s
        if self._min is None:
            return int(host_s * 1e6)
        return int(host_s * 1e6 - self._min)

    def latency_ms(self) -> float:
        """Spread between newest and fastest observation (≈ current extra delay)."""
        if not self._obs:
            return float('nan')
        return (self._obs[-1][1] - self._min) / 1e3
