"""Causal (sample-by-sample) inertial mechanization for live foot IMUs.

``LiveFoot`` is the streaming twin of ``inertial.compute_position``: push one
sample at a time and it produces the SAME numbers the offline function
produces on the full recording (tests/test_live_mechanize.py checks this on
real Brock data), just later:

* orientation (quaternion), navigation-frame acceleration ``An`` and the
  stance flag are available immediately — they only depend on the past;
* a footfall is only known once its stance plateau has CLOSED, i.e. ``T_FF``
  (0.4 s) after the last low-motion sample, because ``foot_fall`` merges
  plateaus separated by < T_FF and puts the footfall at the arg-min of |gyro|
  over the whole plateau. At that moment the stride that ended at the
  footfall gets its zero-velocity update and its final ``Anz``/``V``/``P``
  are emitted as a ``StrideBlock``.

Between footfalls ``provisional_velocity()`` dead-reckons the open stride
with the previous stride's ZUPT bias (display only — it is replaced by the
exact values when the footfall is decided).

Latency is therefore one stance + T_FF (~0.6-0.8 s while walking): fine for
live plotting and per-stride feedback, not for within-stride control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..inertial import (GRAVITY, FootTrajectory, kalman_filter_tilt, qua2eul,
                        qua2rot, qua_est)


def _acc_tilt_1(ax: float, ay: float):
    """Per-sample ``inertial.acc_tilt`` (same quirks: out-of-range ratios are
    passed through un-arcsined, exactly like the vectorised version)."""
    theta = ax / GRAVITY
    if abs(theta) <= 1:
        theta = np.arcsin(theta)
    phi = ay / (np.cos(theta) * GRAVITY)
    if abs(phi) <= 1:
        phi = -np.arcsin(phi)
    return phi, theta


class LiveFootfall:
    """Causal ``inertial.foot_fall``: identical footfall indices, decided late.

    ``push(i, w)`` takes the absolute sample index and the rad/sample gyro
    vector; it returns the list of footfall indices that became certain at
    this sample (usually empty, occasionally one). ``stationary`` (the
    low-motion flag of the sample just pushed) is exposed for the tilt filter.
    """

    def __init__(self, period: float, W_FF: float = 30, A_FF: float = 1,
                 T_FF: float = 0.4, MAX_T_FF: Optional[float] = None):
        self.period = period
        self.W_FF = W_FF
        self.A_FF = A_FF
        self.T_FF = int(T_FF / period)
        self.MAX_T_FF = self.T_FF * 10 if MAX_T_FF is None else int(MAX_T_FF / period)
        self.seg_start: Optional[int] = None   # first low-motion sample of the open plateau
        self.last_low: Optional[int] = None
        self.n_chunks_done = 0                 # full MAX_T_FF chunks already emitted
        self._wm: List[float] = []             # |gyro| (deg/s) from seg_start on
        self.stationary = False

    def _chunk_argmins(self, upto: int) -> List[int]:
        """Emit every full MAX_T_FF chunk of the open plateau ending <= upto."""
        out = []
        M = self.MAX_T_FF
        while M > 0 and self.seg_start + (self.n_chunks_done + 1) * M - 1 <= upto:
            lo = self.n_chunks_done * M
            out.append(self.seg_start + lo + int(np.argmin(self._wm[lo:lo + M])))
            self.n_chunks_done += 1
        return out

    def _close(self) -> List[int]:
        e = self.last_low
        out = self._chunk_argmins(e)
        lo = self.n_chunks_done * self.MAX_T_FF
        n_cut = e - self.seg_start + 1
        if lo < n_cut:
            out.append(self.seg_start + lo + int(np.argmin(self._wm[lo:n_cut])))
        self.seg_start = None
        self.n_chunks_done = 0
        self._wm = []
        return out

    def push(self, i: int, w: np.ndarray, a: np.ndarray) -> List[int]:
        wm = float(np.sqrt(np.sum(w ** 2))) * 180 / np.pi / self.period
        am = float(np.sqrt(np.sum(a ** 2))) - GRAVITY
        low = (wm < self.W_FF) and (abs(am) < self.A_FF)
        self.stationary = low
        out: List[int] = []
        if self.seg_start is not None and i - self.last_low > self.T_FF:
            out += self._close()
        if self.seg_start is not None:
            self._wm.append(wm)
        if low:
            if self.seg_start is None:
                self.seg_start = i
                self._wm = [wm]
            self.last_low = i
            out += self._chunk_argmins(self.last_low)
        return out

    def finish(self) -> List[int]:
        """End of stream: close the open plateau (offline treats the data end
        as the plateau end)."""
        return self._close() if self.seg_start is not None else []


@dataclass
class StrideBlock:
    """Final (ZUPT-corrected) samples [start, stop] of one foot, emitted when
    the footfall at ``stop`` is decided. Indices are absolute sample numbers.
    A footfall that ``zero_velocity_updates`` skips (range < 2 samples) makes
    no block; its samples join the next one."""
    start: int
    stop: int
    Anz: np.ndarray
    V: np.ndarray
    P: np.ndarray


class LiveFoot:
    """Streaming ``compute_position`` for one foot (USE_KF, default thresholds).

    Feed body-frame samples with ``push(w, a)`` — w in rad/sample and a in
    m/s², i.e. exactly the ``Wb``/``Ab`` of an ``ImuRecording``. Read
    ``pop_blocks()`` for finalized strides and ``provisional_velocity()`` for
    the live (not yet ZUPTed) state.
    """

    def __init__(self, period: float, W_FF: float = 30, A_FF: float = 1,
                 T_FF: float = 0.4, MAX_T_FF: Optional[float] = None,
                 keep_history: bool = False):
        self.period = period
        self.ff = LiveFootfall(period, W_FF, A_FF, T_FF, MAX_T_FF)
        self.n = 0                        # samples pushed
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        _, self.P_kf, self.Q_kf, self.R_kf = kalman_filter_tilt(period, None, None, None, None)
        # Samples not yet covered by a ZUPT: An from self.base on.
        self.base = 0                     # absolute index of self._an[0]
        self._an: List[np.ndarray] = []
        self.last_footfall = -1           # as zero_velocity_updates after i==1
        self.v_carry = np.zeros(3)        # cumsum(Anz) up to base-1 (unscaled)
        self.p_carry = np.zeros(3)        # cumsum(V) up to base-1 (unscaled by 2nd period)
        self.bias: Optional[np.ndarray] = None   # last stride's ZUPT correction
        self._stance_sum = np.zeros(3)             # pre-first-ZUPT gravity estimate
        self._stance_n = 0
        self._blocks: List[StrideBlock] = []
        self.footfalls: List[int] = []    # decided footfall indices, in order
        self.keep_history = keep_history
        self.hist_q: List[np.ndarray] = []
        self.hist_an: List[np.ndarray] = []
        self.hist_stat: List[bool] = []

    # --- per-sample -------------------------------------------------------
    def push(self, w: np.ndarray, a: np.ndarray) -> None:
        i = self.n
        new_ff = self.ff.push(i, w, a)
        stationary = self.ff.stationary
        if i == 0:
            # compute_position: q0 from the KF init, An[0] = R(q0) A[0]
            an = (qua2rot(self.q) @ a.reshape(3, 1)).ravel()
        else:
            self.q = qua_est(w, self.q)
            an = (qua2rot(self.q) @ a.reshape(3, 1)).ravel()
            phi, theta = _acc_tilt_1(a[0], a[1])
            self.q, self.P_kf, self.Q_kf, self.R_kf = kalman_filter_tilt(
                self.period, self.q, self.P_kf, self.Q_kf, self.R_kf, theta, phi, stationary)
        self._an.append(an)
        if stationary and self.bias is None:
            self._stance_sum += an
            self._stance_n += 1
        if self.keep_history:
            self.hist_q.append(self.q.copy())
            self.hist_an.append(an)
            self.hist_stat.append(stationary)
        self.n += 1
        for f in new_ff:
            self._footfall(f)

    def push_block(self, W: np.ndarray, A: np.ndarray) -> None:
        for w, a in zip(W, A):
            self.push(w, a)

    def finish(self) -> None:
        """End of stream, mirroring compute_position on the data so far:
        close the open plateau and treat the last sample as a footfall."""
        for f in self.ff.finish():
            self._footfall(f)
        if self.n > 0:
            self._footfall(self.n - 1)

    # --- ZUPT --------------------------------------------------------------
    def _footfall(self, f: int) -> None:
        if self.footfalls and f <= self.footfalls[-1]:
            return                        # e.g. finish()'s N-1 already decided
        self.footfalls.append(f)
        if f < 1:
            return                        # compute_position's loop starts at i=1
        lo = self.last_footfall + 1
        if f - lo + 1 < 2:
            return                        # zero_velocity_updates skips; range grows
        # _an starts at self.base; ZUPT covers [lo, f], and lo == self.base.
        An = np.asarray(self._an[lo - self.base: f - self.base + 1])
        err = np.sum(An, axis=0) / An.shape[0]
        Anz = An - err
        # bit-identical to np.cumsum over the whole record (sequential sums)
        vs = np.cumsum(np.vstack([self.v_carry, Anz]), axis=0)[1:]
        V = vs * self.period
        ps = np.cumsum(np.vstack([self.p_carry, V]), axis=0)[1:]
        self.v_carry = vs[-1]
        self.p_carry = ps[-1]
        self._blocks.append(StrideBlock(lo, f, Anz, V, ps * self.period))
        self.bias = err
        del self._an[: f - self.base + 1]
        self.base = f + 1
        self.last_footfall = f

    # --- consumers ----------------------------------------------------------
    def pop_blocks(self) -> List[StrideBlock]:
        out, self._blocks = self._blocks, []
        return out

    def provisional_velocity(self) -> np.ndarray:
        """Velocity [m/s] of the samples after the last footfall (n_open x 3),
        dead-reckoned with the previous stride's bias. Display only."""
        if not self._an:
            return np.zeros((0, 3))
        An = np.asarray(self._an)
        if self.bias is not None:
            bias = self.bias
        elif self._stance_n:
            bias = self._stance_sum / self._stance_n     # gravity seen while still
        else:
            bias = An.mean(axis=0)
        return (self.v_carry + np.cumsum(An - bias, axis=0)) * self.period

    def last_position(self) -> np.ndarray:
        """Final position [m] at the last ZUPTed sample (zeros before the first)."""
        return self.p_carry * self.period

    def provisional_position(self) -> np.ndarray:
        """Dead-reckoned current position [m] (display only)."""
        v = self.provisional_velocity()
        if not len(v):
            return self.last_position()
        return self.last_position() + np.sum(v, axis=0) * self.period

    @property
    def open_start(self) -> int:
        """Absolute index of the first not-yet-final sample."""
        return self.base


def run_live(W: np.ndarray, A: np.ndarray, period: float, **kw) -> FootTrajectory:
    """Replay a whole recording through LiveFoot and assemble the result into
    a FootTrajectory — for testing equivalence with compute_position."""
    foot = LiveFoot(period, keep_history=True, **kw)
    N = len(W)
    Anz = np.zeros((N, 3))
    V = np.zeros((N, 3))
    P = np.zeros((N, 3))
    for i in range(N):
        foot.push(W[i], A[i])
    foot.finish()
    for b in foot.pop_blocks():
        Anz[b.start:b.stop + 1] = b.Anz
        V[b.start:b.stop + 1] = b.V
        P[b.start:b.stop + 1] = b.P
    # compute_position runs cumsum over all N, so samples no ZUPT covered
    # (Anz == 0) still carry the running V / P; fill them the same way.
    Vfull = np.cumsum(Anz, axis=0) * period
    Pfull = np.cumsum(Vfull, axis=0) * period
    FF = np.zeros(N, bool)
    FF[foot.footfalls] = True
    q = np.asarray(foot.hist_q)
    return FootTrajectory(
        FF=FF, FF_walking=FF, stationary_periods=np.asarray(foot.hist_stat),
        P=Pfull, V=Vfull, Vm=np.sqrt(np.sum(Vfull[:, :2] ** 2, axis=1)),
        euler=qua2eul(q), quaternion=q, An=np.asarray(foot.hist_an), Anz=Anz,
        A=A, W=W)
