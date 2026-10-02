"""Matplotlib live view of two walkers' feet.

One row per subject (labels '<side>_foot_<subject>', e.g. left_foot_a):
  foot speed (solid = final/ZUPTed, faint = provisional) | |gyro| with the
  stance flag shaded | overhead path of the finalized strides.
Events (Opal buttons, sync-box inputs, keys) are vertical lines on the strips
and listed at the bottom with the latest stride stats per foot.

Keys: s = 'Start', e = 'Stop', m = 'mark', o = pulse the AP output line
(hardware only), q = quit.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .pipeline import LivePipeline
from .sources import Event

SIDE_COLOR = {'left': '#1f5fbf', 'right': '#c8372d', '?': '#444444'}
EVENT_COLOR = {'button': '#2a9d3f', 'key': '#7b3fb0', 'sync_in': '#d48a00', 'status': '#888888'}


def ensure_gui_backend():
    """Switch to a windowed backend if we were left on a non-interactive one
    (e.g. MPLBACKEND=Agg). Call BEFORE creating the LiveViewer."""
    if matplotlib.get_backend().lower() in ('agg', 'svg', 'pdf', 'ps', 'template'):
        for b in ('macosx', 'qtagg', 'tkagg'):
            try:
                plt.switch_backend(b)
                return b
            except Exception:
                continue
    return matplotlib.get_backend()


def _side_subject(label: str):
    lab = label.lower()
    side = 'left' if 'left' in lab else 'right' if 'right' in lab else '?'
    subject = lab.rsplit('_', 1)[-1] if '_' in lab else lab
    return side, subject


def _heading_up(P: np.ndarray) -> np.ndarray:
    """Each foot's nav frame has its own arbitrary yaw: rotate the path so
    its start->end displacement points +Y, origin at the start (the same idea
    as plotting.draw_overhead, applied to the visible path only)."""
    P = P - P[0]
    d = P[-1]
    if np.hypot(*d) < 0.5:              # standing / shuffling: leave as is
        return P
    ang = np.pi / 2 - np.arctan2(d[1], d[0])
    c, s = np.cos(ang), np.sin(ang)
    return P @ np.array([[c, s], [-s, c]])


class LiveViewer:
    def __init__(self, pipe: LivePipeline, fps: float = 15, speed_max: float = 6.0,
                 gyro_max: float = 800.0, output_pin: int = 0):
        self.pipe = pipe
        self.fps = fps
        self.output_pin = output_pin
        self.subjects: Dict[str, List[int]] = OrderedDict()
        for dev, fs in pipe.feet.items():
            self.subjects.setdefault(_side_subject(fs.label)[1], []).append(dev)
        nrow = len(self.subjects)
        self.fig = plt.figure(figsize=(13, 3.3 * nrow + 1.2))
        gs = self.fig.add_gridspec(nrow + 1, 3, width_ratios=[3, 3, 1.6],
                                   height_ratios=[3] * nrow + [1.1], hspace=0.45, wspace=0.25)
        self.lines = {}
        self.axes = []
        W = pipe.window_s
        for r, (subj, devs) in enumerate(self.subjects.items()):
            ax_v = self.fig.add_subplot(gs[r, 0])
            ax_w = self.fig.add_subplot(gs[r, 1], sharex=ax_v)
            ax_p = self.fig.add_subplot(gs[r, 2])
            ax_v.set_ylim(0, speed_max)
            ax_v.set_ylabel(f'subject {subj}\nfoot speed [m/s]')
            ax_w.set_ylim(0, gyro_max)
            ax_w.set_ylabel('|gyro| [deg/s]')
            for ax in (ax_v, ax_w):
                ax.set_xlim(-W, 0)
                ax.grid(alpha=0.3)
            ax_p.set_aspect('equal', adjustable='datalim')
            ax_p.set_title('overhead (finalized)', fontsize=9)
            ax_p.grid(alpha=0.3)
            if r == len(self.subjects) - 1:
                ax_v.set_xlabel('t - now [s]')
                ax_w.set_xlabel('t - now [s]')
            for dev in devs:
                side, _ = _side_subject(pipe.feet[dev].label)
                c = SIDE_COLOR[side]
                self.lines[dev] = dict(
                    vf=ax_v.plot([], [], color=c, lw=1.6, label=side)[0],
                    vp=ax_v.plot([], [], color=c, lw=1.0, alpha=0.35)[0],
                    w=ax_w.plot([], [], color=c, lw=1.0)[0],
                    st=ax_w.plot([], [], color=c, lw=4, alpha=0.5, solid_capstyle='butt')[0],
                    p=ax_p.plot([], [], color=c, lw=1.2)[0])
            ax_v.legend(loc='upper left', fontsize=8, ncol=2)
            self.axes.append((ax_v, ax_w, ax_p))
        self.ax_txt = self.fig.add_subplot(gs[nrow, :])
        self.ax_txt.axis('off')
        self.txt = self.ax_txt.text(0, 1, '', va='top', family='monospace', fontsize=8.5,
                                    transform=self.ax_txt.transAxes)
        self.ev_lines = []
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # --- keys ----------------------------------------------------------------
    def _latest_t_us(self) -> int:
        """Sensor-clock time of the newest sample (what 'now' shows)."""
        fs = next(iter(self.pipe.feet.values()))
        if fs.t0_us is None:
            return int(time.time() * 1e6)
        return int(fs.t0_us + fs.n * self.pipe.source.period * 1e6)

    def _on_key(self, event):
        names = {'s': 'Start', 'e': 'Stop', 'm': 'mark'}
        if event.key in names:
            self.pipe.add_event(Event(self._latest_t_us(), 0, 'key', names[event.key]))
        elif event.key == 'o' and hasattr(self.pipe.source, 'set_output'):
            self.pipe.source.set_output(self.output_pin, 1)
            self.pipe.add_event(Event(self._latest_t_us(), 0, 'key', f'output pin {self.output_pin} pulse'))
            self.fig.canvas.new_timer(interval=100, callbacks=[(self.pipe.source.set_output,
                                                                (self.output_pin, 0), {})]).start()

    # --- drawing ---------------------------------------------------------------
    def update(self, _frame=None):
        self.pipe.step()
        p = self.pipe.source.period
        now = self.pipe.now_s()
        for dev, L in self.lines.items():
            fs = self.pipe.feet[dev]
            if fs.n == 0:
                continue
            n = fs.speed.n
            t = (np.arange(fs.n - n, fs.n) - fs.n) * p
            v = fs.speed.last(fs.n)[:, 0]
            fin = fs.final.last(fs.n)[:, 0] == 1
            L['vf'].set_data(t, np.where(fin, v, np.nan))
            L['vp'].set_data(t, np.where(fin, np.nan, v))
            L['w'].set_data(t, fs.wm.last(fs.n)[:, 0])
            st = fs.stance.last(fs.n)[:, 0] == 1
            L['st'].set_data(t, np.where(st, 10.0, np.nan))
            if fs.path:
                P = _heading_up(np.vstack(fs.path[-400:]))
                L['p'].set_data(P[:, 0], P[:, 1])
        for _, _, ax_p in self.axes:
            ax_p.relim()
            ax_p.autoscale_view()
        for ln in self.ev_lines:
            ln.remove()
        self.ev_lines = []
        recent = [e for e in self.pipe.events if e.kind != 'status'
                  and now - self.pipe.window_s <= self.pipe.session_s(e.t_us) <= now + 1]
        for ev in recent:
            x = self.pipe.session_s(ev.t_us) - now
            for ax_v, ax_w, _ in self.axes:
                for ax in (ax_v, ax_w):
                    self.ev_lines.append(ax.axvline(x, color=EVENT_COLOR.get(ev.kind, 'k'), lw=1.2, ls='--'))
            self.ev_lines.append(self.axes[0][0].text(x, self.axes[0][0].get_ylim()[1], f' {ev.text}',
                                                      fontsize=8, va='top', color=EVENT_COLOR.get(ev.kind, 'k')))
        self.txt.set_text(self._status_text(now))
        return []

    def _status_text(self, now: float) -> str:
        rows = [f't = {now:8.2f} s   mechanization load {100 * self.pipe.compute_load():4.1f}% of one core'
                '   keys: s=Start e=Stop m=mark o=output q=quit']
        for dev, fs in self.pipe.feet.items():
            s = next((s for s in reversed(fs.strides) if np.isfinite(s.length_m)), None)
            if s is None:
                rows.append(f'{fs.label:>14}: no stride yet')
            elif s.length_m < 0.2:
                rows.append(f'{fs.label:>14}: standing (last contact {now - s.t:4.1f} s ago)')
            else:
                rows.append(f'{fs.label:>14}: last stride {s.length_m:4.2f} m  {s.time_s:4.2f} s  '
                            f'{s.speed_ms:4.2f} m/s   ({now - s.t:4.1f} s ago)')
        evs = [e for e in self.pipe.events][-4:]
        rows.append('events: ' + '   '.join(f'[{self.pipe.session_s(e.t_us):.2f} s {e.kind}] {e.text}'
                                             for e in evs))
        return '\n'.join(rows)

    def run(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=1000 / self.fps,
                                  cache_frame_data=False)
        plt.show()
