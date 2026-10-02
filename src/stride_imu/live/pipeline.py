"""Source -> (recorder, one LiveFoot per foot) -> display buffers.

``LivePipeline.step()`` is the whole live loop body: poll the source, append
raw samples to the host recording, mechanize every foot, and keep rolling
buffers the viewer draws from. It has no plotting in it, so it runs headless
(tests, a feedback controller) exactly as it runs under the viewer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..inertial import GRAVITY
from .clock import SensorClock
from .mechanize import LiveFoot
from .sources import Event, to_body


class Ring:
    """Fixed-length rolling buffer addressed by ABSOLUTE sample index."""

    def __init__(self, n: int, width: int = 1):
        self.n = n
        self.a = np.full((n, width), np.nan)

    def put(self, i0: int, x: np.ndarray):
        x = np.asarray(x, float).reshape(len(x), -1)
        if len(x) > self.n:
            i0, x = i0 + len(x) - self.n, x[-self.n:]
        idx = (i0 + np.arange(len(x))) % self.n
        self.a[idx] = x

    def last(self, end: int) -> np.ndarray:
        """Values for absolute samples [end - n, end), oldest first."""
        idx = (end - self.n + np.arange(self.n)) % self.n
        return self.a[idx]


@dataclass
class StrideStat:
    stop: int                 # footfall sample (absolute, per-foot index)
    t: float                  # session seconds of the footfall
    time_s: float
    length_m: float
    speed_ms: float


@dataclass
class FootState:
    label: str
    foot: LiveFoot
    n: int = 0
    wm: Ring = None           # |gyro| deg/s
    am: Ring = None           # |accel| - g
    speed: Ring = None        # horizontal foot speed (final where known)
    final: Ring = None        # 1 where speed is final, 0 provisional
    stance: Ring = None
    path: List[np.ndarray] = field(default_factory=list)   # finalized P (downsampled)
    ff_pos: List[np.ndarray] = field(default_factory=list) # P [m] at each finalized footfall
    strides: List[StrideStat] = field(default_factory=list)
    t0_us: Optional[int] = None
    # link health
    last_t_us: Optional[int] = None
    last_host: float = 0.0
    gaps: int = 0                 # sample-time jumps > 1.5 periods
    missing: int = 0              # samples those jumps skipped
    recv: List[tuple] = field(default_factory=list)   # (host_s, n) of recent blocks


class LivePipeline:
    def __init__(self, source, recorder=None, window_s: float = 10.0,
                 orientation: Optional[int] = None, path_every: int = 4):
        self.source = source
        self.recorder = recorder
        self.window_s = window_s
        self.orientation = orientation
        self.path_every = path_every
        self.events: List[Event] = []
        self.feet: Dict[int, FootState] = {}
        self.t0_us: Optional[int] = None
        self.compute_s = 0.0          # time spent mechanizing (for the report)
        self.samples = 0
        self.clock = SensorClock()    # host -> Opal clock, for host-generated events

    def setup(self):
        p = self.source.period
        L = int(self.window_s / p)
        for dev, label in self.source.devices.items():
            self.feet[dev] = FootState(label, LiveFoot(p), wm=Ring(L), am=Ring(L),
                                       speed=Ring(L), final=Ring(L), stance=Ring(L))

    def session_s(self, t_us: int) -> float:
        return (t_us - self.t0_us) / 1e6 if self.t0_us is not None else 0.0

    def add_event(self, ev: Event):
        self.events.append(ev)
        if self.recorder is not None:
            self.recorder.annotate(ev)

    def step(self) -> int:
        blocks, events = self.source.poll()
        p = self.source.period
        host = time.time()
        newest = None
        for b in blocks:
            if self.recorder is not None:
                self.recorder.write(b)
            fs = self.feet.get(b.device_id)
            if fs is None or len(b.t_us) == 0:
                continue
            self._link_health(fs, b, host, p)
            newest = int(b.t_us[-1]) if newest is None else max(newest, int(b.t_us[-1]))
            if self.t0_us is None:
                self.t0_us = int(b.t_us[0])
            W, A = to_body(b, p, self.orientation)
            i0 = fs.n
            tic = time.perf_counter()
            stance = np.zeros(len(W))
            for k in range(len(W)):
                fs.foot.push(W[k], A[k])
                stance[k] = fs.foot.ff.stationary
            self.compute_s += time.perf_counter() - tic
            self.samples += len(W)
            fs.n += len(W)
            fs.wm.put(i0, np.linalg.norm(W, axis=1) * 180 / np.pi / p)
            fs.am.put(i0, np.linalg.norm(A, axis=1) - GRAVITY)
            fs.stance.put(i0, stance)
            if fs.t0_us is None:
                fs.t0_us = int(b.t_us[0])
            for blk in fs.foot.pop_blocks():
                fs.speed.put(blk.start, np.hypot(blk.V[:, 0], blk.V[:, 1]))
                fs.final.put(blk.start, np.ones(len(blk.V)))
                fs.ff_pos.append(blk.P[-1].copy())
                if blk.start > 0:     # the first block starts un-anchored
                    fs.path.append(blk.P[::self.path_every, :2])
                self._stride_stat(fs, blk)
            # provisional speed for the open (not yet ZUPTed) samples
            v = fs.foot.provisional_velocity()
            if len(v):
                fs.speed.put(fs.foot.open_start, np.hypot(v[:, 0], v[:, 1]))
                fs.final.put(fs.foot.open_start, np.zeros(len(v)))
        if newest is not None:
            self.clock.observe(newest, host)
        for ev in events:
            self.add_event(ev)
        return sum(len(b.t_us) for b in blocks)

    @staticmethod
    def _link_health(fs: FootState, b, host: float, p: float):
        t = np.asarray(b.t_us, dtype=np.int64)
        if fs.last_t_us is not None:
            t = np.concatenate([[fs.last_t_us], t])
        dt = np.diff(t) / 1e6
        jumps = dt > 1.5 * p
        fs.gaps += int(np.sum(jumps))
        fs.missing += int(np.sum(np.round(dt[jumps] / p) - 1)) if jumps.any() else 0
        fs.last_t_us = int(b.t_us[-1])
        fs.last_host = host
        fs.recv.append((host, len(b.t_us)))
        while fs.recv and fs.recv[0][0] < host - 2.0:
            fs.recv.pop(0)

    def rate_hz(self, fs: FootState) -> float:
        """Received samples per second over the last ~2 s."""
        if len(fs.recv) < 2:
            return 0.0
        span = fs.recv[-1][0] - fs.recv[0][0]
        return sum(n for _, n in fs.recv[1:]) / span if span > 0 else 0.0

    def _stride_stat(self, fs: FootState, blk):
        """Footfall-to-footfall stats. A block spans (previous footfall,
        this footfall]; the first block starts at the stream start, not at a
        footfall, so it has no stride."""
        t = self._foot_t(fs, blk.stop)
        if blk.start == 0:
            fs.strides.append(StrideStat(blk.stop, t, np.nan, np.nan, np.nan))
            return
        dt = (blk.stop - blk.start + 1) * self.source.period
        # P at the previous footfall: P[start] = P[start-1] + V[start]*period
        p_prev = blk.P[0, :2] - blk.V[0, :2] * self.source.period
        length = float(np.hypot(*(blk.P[-1, :2] - p_prev)))
        fs.strides.append(StrideStat(blk.stop, t, dt, length, length / dt))

    def _foot_t(self, fs: FootState, i: int) -> float:
        return self.session_s(fs.t0_us) + i * self.source.period

    def now_s(self) -> float:
        n = max((f.n for f in self.feet.values()), default=0)
        fs = next(iter(self.feet.values()), None)
        return (self._foot_t(fs, 0) if fs and fs.t0_us else 0.0) + n * self.source.period

    def compute_load(self) -> float:
        """Mechanization CPU seconds per second of data (all feet)."""
        n = max((f.n for f in self.feet.values()), default=0)
        data_s = n * self.source.period
        return self.compute_s / data_s if data_s else 0.0
