"""Live experiment console (PySide6 + pyqtgraph).

Layout, top to bottom:
  transport  | play/pause, previous / restart / next trial, mute, current trial,
               next cue countdown, zero buttons, assign Opals, recording status
  cue track  | DAW-style lanes (Announce / Person A / Person B / Events) on
               experiment time; played cues dimmed, playhead fixed-follow,
               double-click to seek
  person A   | live |accel| of both feet  | overhead footfalls (zeroable)
  person B   | same
  bottom     | note entry + Mark, event log, Opal link health

Schedule cues with an ``event`` ('Start'/'Stop') are written — stamped in the
Opal clock — to the host recording's Annotations and to events.csv, and can
pulse the access point's output line. Zeroing only moves the overhead
display; nothing recorded changes.

Keys: Space play/pause, Left previous trial, R restart trial, Right next
trial, Z zero both, M mark.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .audio import CuePlayer
from .pipeline import LivePipeline
from .recorder import EventLog
from .schedule import KIND_COLORS, TRACKS, Conductor, Schedule
from .sources import Event

SIDE_COLOR = {'left': '#1f5fbf', 'right': '#c8372d'}
ROLES = ['A left', 'A right', 'B left', 'B right', 'event', 'box', 'ignore']
LANES = list(TRACKS) + ['Events']
LANE_Y = {name: len(LANES) - 1 - k for k, name in enumerate(LANES)}   # top lane highest
EVENT_COLORS = {'Start': '#2a9d3f', 'Stop': '#c8372d', 'button': '#1b7f8c',
                'key': '#7b3fb0', 'note': '#7b3fb0', 'sync_in': '#d48a00'}


def guess_role(label: str) -> str:
    lab = label.lower()
    if 'event' in lab:
        return 'event'
    if 'box' in lab:
        return 'box'
    side = 'left' if 'left' in lab else 'right' if 'right' in lab else None
    person = lab.rsplit('_', 1)[-1] if '_' in lab else ''
    if side and person in ('a', 'b', '1', '2', 's1', 's2'):
        p = 'A' if person in ('a', '1', 's1') else 'B'
        return f'{p} {side}'
    return 'ignore'


# ---------------------------------------------------------------------------
# Opal -> role assignment
# ---------------------------------------------------------------------------
class AssignDialog(QtWidgets.QDialog):
    """Pick each Opal's role. The activity bars are live |gyro| — shake a
    sensor and watch which row lights up."""

    def __init__(self, pipe: LivePipeline, roles: Dict[int, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Assign Opals to people')
        self.pipe = pipe
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel('Shake an Opal to see which row it is. Roles only affect the '
                                       'display; every Opal is recorded regardless.'))
        self.table = QtWidgets.QTableWidget(len(pipe.feet), 4)
        self.table.setHorizontalHeaderLabels(['Opal', 'label', 'activity', 'role'])
        self.bars, self.combos = {}, {}
        for r, (dev, fs) in enumerate(pipe.feet.items()):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f'XI-{dev:06d}'))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(fs.label))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 400)
            bar.setTextVisible(False)
            self.table.setCellWidget(r, 2, bar)
            cb = QtWidgets.QComboBox()
            cb.addItems(ROLES)
            cb.setCurrentText(roles.get(dev, guess_role(fs.label)))
            self.table.setCellWidget(r, 3, cb)
            self.bars[dev], self.combos[dev] = bar, cb
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumWidth(620)
        lay.addWidget(self.table)
        self.warn = QtWidgets.QLabel('')
        self.warn.setStyleSheet('color: #c8372d')
        lay.addWidget(self.warn)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(100)

    def _update(self):
        for dev, bar in self.bars.items():
            fs = self.pipe.feet[dev]
            if fs.n:
                w = fs.wm.last(fs.n)[-int(0.5 / self.pipe.source.period):, 0]
                bar.setValue(int(np.nanmax(w)) if np.isfinite(w).any() else 0)

    def roles(self) -> Dict[int, str]:
        return {dev: cb.currentText() for dev, cb in self.combos.items()}

    def _accept(self):
        used = [r for r in self.roles().values() if r not in ('ignore', 'box')]
        dup = {r for r in used if used.count(r) > 1}
        if dup:
            self.warn.setText('Each role can only be used once: ' + ', '.join(sorted(dup)))
            return
        self.accept()


# ---------------------------------------------------------------------------
# DAW-style cue track
# ---------------------------------------------------------------------------
class CueTrack(pg.PlotWidget):
    seekRequested = QtCore.Signal(float)

    def __init__(self, schedule: Schedule, past_s: float = 40, ahead_s: float = 110):
        super().__init__()
        self.schedule = schedule
        self.past_s, self.ahead_s = past_s, ahead_s
        self.setBackground('w')
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setFixedHeight(190)
        ax = self.getAxis('left')
        ax.setTicks([[(LANE_Y[n], n) for n in LANES]])
        ax.setWidth(70)
        self.getAxis('bottom').setLabel('experiment time [s]')
        self.setYRange(-0.6, len(LANES) - 0.4, padding=0)
        vb = self.getViewBox()
        # trial blocks: alternating shading + label
        for k, b in enumerate(schedule.blocks):
            if k % 2 == 0:
                r = QtWidgets.QGraphicsRectItem(b.t_start - 0.5, -0.5, b.t_end - b.t_start + 1.0, len(LANES))
                r.setBrush(pg.mkBrush(0, 0, 0, 12))
                r.setPen(pg.mkPen(None))
                vb.addItem(r)
        # cues
        self.rects, self.labels = [], []
        for c in schedule.cues:
            y = LANE_Y.get(c.track, 0)
            r = QtWidgets.QGraphicsRectItem(c.t_s, y - 0.32, max(c.duration_s, 0.3), 0.64)
            col = QtGui.QColor(KIND_COLORS.get(c.kind, '#888'))
            r.setBrush(pg.mkBrush(col))
            r.setPen(pg.mkPen(col.darker(130)))
            r.setToolTip(f'{c.t_s:.1f} s  {c.text}' + (f'  -> {c.event}' if c.event else ''))
            vb.addItem(r)
            # short clips are narrow at this zoom: label beside the block, in its colour
            short = {'announce': f'T{c.trial} r{c.rep}', 'go': f'GO {c.walker.upper()}', 'stop': 'STOP'}.get(c.kind)
            lbl = None
            if short:
                lbl = pg.TextItem(short, color=col.darker(140), anchor=(0, 0.5))
                lbl.setPos(c.t_s + max(c.duration_s, 0.3) + 0.15, y)
                vb.addItem(lbl)
            self.labels.append(lbl)
            self.rects.append(r)
        self.playhead = pg.InfiniteLine(0, angle=90, pen=pg.mkPen('#111', width=2))
        vb.addItem(self.playhead)
        self.event_items: List = []
        self._dimmed = -1
        key = '   '.join(f'<span style="color:{KIND_COLORS[k]}">■</span> {n}' for k, n in
                         (('announce', 'announce'), ('ready', 'ready'), ('go', 'go → Start'),
                          ('stop', 'stop → Stop')))
        self.legend = QtWidgets.QLabel(key, self)
        self.legend.setStyleSheet('background: rgba(255,255,255,210); font-size: 11px; padding: 1px 4px')
        self.legend.move(78, 2)

    def mark_played(self, upto_index: int):
        """Dim cues already played (index into schedule.cues)."""
        if upto_index == self._dimmed:
            return
        for k, (r, lbl) in enumerate(zip(self.rects, self.labels)):
            op = 0.35 if k < upto_index else 1.0
            r.setOpacity(op)
            if lbl is not None:
                lbl.setOpacity(op)
        self._dimmed = upto_index

    def add_event_marker(self, exp_t: float, text: str, color: str):
        y = LANE_Y['Events']
        ln = pg.PlotDataItem([exp_t, exp_t], [y - 0.35, y + 0.35], pen=pg.mkPen(color, width=3))
        tx = pg.TextItem(text, color=color, anchor=(0, 0.5))
        tx.setPos(exp_t + 0.2, y)
        self.addItem(ln)
        self.addItem(tx)
        self.event_items += [ln, tx]

    def follow(self, exp_t: float):
        self.playhead.setValue(exp_t)
        self.setXRange(exp_t - self.past_s, exp_t + self.ahead_s, padding=0)

    def mouseDoubleClickEvent(self, ev):
        pt = self.getViewBox().mapSceneToView(self.mapToScene(ev.position().toPoint()))
        self.seekRequested.emit(float(pt.x()))
        ev.accept()


# ---------------------------------------------------------------------------
# One person's row: live accel + overhead footfalls
# ---------------------------------------------------------------------------
class PersonView:
    def __init__(self, name: str, window_s: float, accel_max: float = 50.0,
                 separation: float = 0.2):
        self.name = name
        self.separation = separation
        self.accel = pg.PlotWidget(title=f'Person {name}: |accel|')
        self.accel.setBackground('w')
        self.accel.setLabel('left', '|A| [m/s²]')
        self.accel.setLabel('bottom', 't − now [s]')
        self.accel.setXRange(-window_s, 0, padding=0)
        self.accel.setYRange(0, accel_max, padding=0)
        self.accel.showGrid(x=True, y=True, alpha=0.25)
        self.accel.addLegend(offset=(5, 5))
        self.over = pg.PlotWidget(title=f'Person {name}: footfalls (overhead)')
        self.over.setBackground('w')
        self.over.setAspectLocked(True)
        self.over.showGrid(x=True, y=True, alpha=0.25)
        self.over.setLabel('left', 'forward [m]')
        self.over.setLabel('bottom', 'lateral [m]')
        self.curves, self.ff_scatter, self.dot, self.trail = {}, {}, {}, {}
        for side in ('left', 'right'):
            c = SIDE_COLOR[side]
            self.curves[side] = self.accel.plot([], [], pen=pg.mkPen(c, width=1.3), name=side)
            self.trail[side] = self.over.plot([], [], pen=pg.mkPen(c, width=1, style=QtCore.Qt.DotLine))
            self.ff_scatter[side] = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(c), pen=pg.mkPen('w'))
            self.over.addItem(self.ff_scatter[side])
            self.dot[side] = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(None), pen=pg.mkPen(c, width=2))
            self.over.addItem(self.dot[side])
        self.info = pg.TextItem('', color='#222', anchor=(0, 0), fill=pg.mkBrush(255, 255, 255, 220))
        self.over.addItem(self.info)
        self.event_lines: List = []
        self.zero_state: Dict[str, dict] = {}
        self.zero_time_s: Optional[float] = None

    # --- zeroing (display only) ------------------------------------------------
    def zero(self, feet: Dict[str, object], now_s: float):
        """Re-origin the overhead at the feet's current positions; heading is
        re-locked on each foot's next ~0.6 m of travel."""
        self.zero_time_s = now_s
        self.zero_state = {}
        for side, fs in feet.items():
            if fs is None:
                continue
            # origin = last FINAL (ZUPTed) position: the provisional one can be
            # mid-swing dead reckoning
            self.zero_state[side] = dict(p0=fs.foot.last_position()[:2].copy(),
                                         n_ff0=len(fs.ff_pos), rot=None)

    def _lock_heading(self, side: str, ffs: np.ndarray):
        """Each foot's nav frame has its own yaw. Rotate the foot so the line
        from the zero point to its FARTHEST finalized footfall points +Y (like
        the offline overhead's start->contact alignment, but stable when the
        walker comes back); only once that is >= 0.6 m away, and never from
        provisional motion."""
        z = self.zero_state[side]
        if not len(ffs):
            return
        dd = ffs - z['p0']
        d = dd[np.argmax(np.hypot(dd[:, 0], dd[:, 1]))]
        if np.hypot(*d) >= 0.6:
            ang = np.pi / 2 - np.arctan2(d[1], d[0])
            c, s = np.cos(ang), np.sin(ang)
            z['rot'] = np.array([[c, s], [-s, c]])

    def _display(self, side: str, p: np.ndarray) -> np.ndarray:
        z = self.zero_state[side]
        d = np.atleast_2d(p) - z['p0']
        if z['rot'] is not None:
            d = d @ z['rot']
        off = -self.separation / 2 if side == 'left' else self.separation / 2
        return d + np.array([off, 0.0])

    def update(self, pipe: LivePipeline, feet: Dict[str, object], expected_m: Optional[float]):
        p = pipe.source.period
        dist = []
        for side in ('left', 'right'):
            fs = feet.get(side)
            if fs is None or fs.n == 0:
                self.curves[side].setData([], [])
                continue
            a = fs.am.last(fs.n)[:, 0] + 9.80297286843
            t = (np.arange(len(a)) - len(a)) * p
            self.curves[side].setData(t, a, connect='finite')
            if side not in self.zero_state:
                self.zero(feet, pipe.now_s())
            z = self.zero_state[side]
            # finalized footfalls since zero
            ffs = np.array([q[:2] for q in fs.ff_pos[z['n_ff0']:]]).reshape(-1, 2)
            cur = fs.foot.provisional_position()[:2]
            self._lock_heading(side, ffs)
            pts = self._display(side, np.vstack([ffs, cur]) if len(ffs) else cur)
            self.ff_scatter[side].setData(pts[:-1, 0], pts[:-1, 1])
            self.trail[side].setData(pts[:, 0], pts[:, 1])
            self.dot[side].setData([pts[-1, 0]], [pts[-1, 1]])
            dist.append(float(np.hypot(*(pts[-1] - self._display(side, z['p0'])[0]))))
        txt = f'{max(dist):.2f} m since zero' if dist else ''
        if expected_m is not None and dist:
            txt += f'  (trial: {expected_m:g} m)'
        self.info.setText(txt)
        vb = self.over.getViewBox()
        allp = [s.getData() for s in self.dot.values()] + [s.getData() for s in self.ff_scatter.values()]
        ys = np.concatenate([np.asarray(y) for _, y in allp if len(y)]) if any(len(y) for _, y in allp) else np.array([0.0])
        top = max(2.0, float(np.nanmax(ys)) + 0.5)
        bot = min(-0.5, float(np.nanmin(ys)) - 0.5)
        vb.setRange(xRange=(-1.2, 1.2), yRange=(bot, top), padding=0)
        self.info.setPos(-1.15, top - 0.05)

    def set_event_lines(self, xs_colors):
        for ln in self.event_lines:
            self.accel.removeItem(ln)
        self.event_lines = []
        for x, col, txt in xs_colors:
            ln = pg.InfiniteLine(x, angle=90, pen=pg.mkPen(col, width=1.5, style=QtCore.Qt.DashLine),
                                 label=txt, labelOpts=dict(position=0.92, color=col))
            self.accel.addItem(ln)
            self.event_lines.append(ln)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class ExperimentWindow(QtWidgets.QMainWindow):
    def __init__(self, pipe: LivePipeline, schedule: Schedule, conditions=None,
                 out_dir: str = '.', roles: Optional[Dict[int, str]] = None,
                 mute: bool = False, gpio_pulse: Optional[int] = None,
                 auto_zero: bool = True, video=None):
        super().__init__()
        self.pipe = pipe
        self.schedule = schedule
        self.conditions = conditions
        self.out_dir = out_dir
        self.gpio_pulse = gpio_pulse
        self.video = video                  # live.video.VideoRecorder or None
        self.conductor = Conductor(schedule)
        self.player = CuePlayer(schedule)
        self.player.muted = mute
        self.events_csv = EventLog(os.path.join(out_dir, 'events.csv'))
        self.roles = roles or {dev: guess_role(fs.label) for dev, fs in pipe.feet.items()}
        self.setWindowTitle('Live gait experiment')
        self._build()
        self._save_roles()
        self._n_events_seen = 0
        # timers: cues (100 Hz), data (50 Hz), drawing (20 Hz)
        for ms, fn in ((10, self._tick_cues), (20, self._tick_data), (50, self._tick_draw)):
            t = QtCore.QTimer(self)
            t.setTimerType(QtCore.Qt.PreciseTimer)
            t.timeout.connect(fn)
            t.start(ms)
        self.auto_zero.setChecked(auto_zero)

    # --- layout --------------------------------------------------------------
    def _button(self, text, fn, tip=''):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(fn)
        b.setToolTip(tip)
        return b

    def _build(self):
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        # transport
        bar = QtWidgets.QHBoxLayout()
        self.play_btn = self._button('▶  Play', self._toggle, 'Space')
        self.play_btn.setMinimumWidth(110)
        bar.addWidget(self.play_btn)
        bar.addWidget(self._button('⏮ Prev trial', self._back, 'Left arrow'))
        bar.addWidget(self._button('↺ Restart trial', self._restart, 'R'))
        bar.addWidget(self._button('Next trial ⏭', self._next, 'Right arrow'))
        self.mute_box = QtWidgets.QCheckBox('mute')
        self.mute_box.setChecked(self.player.muted)
        self.mute_box.toggled.connect(lambda m: setattr(self.player, 'muted', m))
        bar.addWidget(self.mute_box)
        self.trial_lbl = QtWidgets.QLabel('')
        self.trial_lbl.setStyleSheet('font-weight: bold; font-size: 14px; padding-left: 12px')
        bar.addWidget(self.trial_lbl, 1)
        self.next_lbl = QtWidgets.QLabel('')
        self.next_lbl.setStyleSheet('font-size: 13px; padding-right: 12px')
        bar.addWidget(self.next_lbl)
        bar.addWidget(self._button('Zero A', lambda: self._zero('A')))
        bar.addWidget(self._button('Zero B', lambda: self._zero('B')))
        self.auto_zero = QtWidgets.QCheckBox('auto-zero walker on Go')
        bar.addWidget(self.auto_zero)
        bar.addWidget(self._button('Assign Opals…', self._assign))
        self.rec_lbl = QtWidgets.QLabel('')
        bar.addWidget(self.rec_lbl)
        v.addLayout(bar)
        # cue track
        self.track = CueTrack(self.schedule)
        self.track.seekRequested.connect(self._seek)
        v.addWidget(self.track)
        # people
        grid = QtWidgets.QGridLayout()
        self.people = {}
        for r, name in enumerate(('A', 'B')):
            pv = PersonView(name, self.pipe.window_s)
            grid.addWidget(pv.accel, r, 0)
            grid.addWidget(pv.over, r, 1)
            self.people[name] = pv
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        v.addLayout(grid, 1)
        # bottom: notes, log, link health
        bottom = QtWidgets.QHBoxLayout()
        notes = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText('note for the current trial (e.g. "B started before command"), Enter to log')
        self.note.returnPressed.connect(self._log_note)
        row.addWidget(self.note, 1)
        row.addWidget(self._button('Mark', lambda: self._manual_event('mark'), 'M'))
        row.addWidget(self._button('Start', lambda: self._manual_event('Start')))
        row.addWidget(self._button('Stop', lambda: self._manual_event('Stop')))
        notes.addLayout(row)
        self.log = QtWidgets.QListWidget()
        self.log.setMaximumHeight(120)
        notes.addWidget(self.log)
        bottom.addLayout(notes, 3)
        if self.video is not None:
            cam = QtWidgets.QVBoxLayout()
            self.cam_view = QtWidgets.QLabel('camera starting…')
            self.cam_view.setFixedSize(256, 144)
            self.cam_view.setAlignment(QtCore.Qt.AlignCenter)
            self.cam_view.setStyleSheet('background: #111; color: #aaa')
            self.cam_lbl = QtWidgets.QLabel('')
            self.cam_lbl.setStyleSheet('font-size: 11px')
            cam.addWidget(self.cam_view)
            cam.addWidget(self.cam_lbl)
            bottom.addLayout(cam)
        self.health = QtWidgets.QTableWidget(0, 5)
        self.health.setHorizontalHeaderLabels(['Opal', 'role', 'Hz', 'age ms', 'missing'])
        self.health.setMaximumHeight(150)
        self.health.verticalHeader().setVisible(False)
        self.health.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        bottom.addWidget(self.health, 2)
        v.addLayout(bottom)
        self.setCentralWidget(central)
        for key, fn in ((QtCore.Qt.Key_Space, self._toggle), (QtCore.Qt.Key_Left, self._back),
                        (QtCore.Qt.Key_Right, self._next), (QtCore.Qt.Key_R, self._restart),
                        (QtCore.Qt.Key_Z, lambda: (self._zero('A'), self._zero('B'))),
                        (QtCore.Qt.Key_M, lambda: self._manual_event('mark'))):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.WindowShortcut)
            sc.activated.connect(fn)
        self.resize(1500, 980)

    # --- roles ------------------------------------------------------------------
    def person_feet(self, name: str):
        out = {'left': None, 'right': None}
        for dev, role in self.roles.items():
            if role.startswith(name + ' '):
                out[role.split()[1]] = self.pipe.feet.get(dev)
        return out

    def _assign(self):
        dlg = AssignDialog(self.pipe, self.roles, self)
        if dlg.exec():
            self.roles = dlg.roles()
            self._save_roles()
            for name in self.people:
                self._zero(name)
            self._add_log(f'roles: {self._roles_text()}')

    def _roles_text(self):
        return ', '.join(f'{self.pipe.feet[d].label}={r}' for d, r in self.roles.items())

    def _save_roles(self):
        with open(os.path.join(self.out_dir, 'roles.json'), 'w') as f:
            json.dump({f'XI-{d:06d}': dict(label=self.pipe.feet[d].label, role=r)
                       for d, r in self.roles.items()}, f, indent=1)

    # --- transport --------------------------------------------------------------
    def _toggle(self):
        self.conductor.toggle()
        self._log_transport('play' if self.conductor.playing else 'pause')

    def _back(self):
        self.conductor.back_trial()
        self._log_transport('back to previous trial')

    def _restart(self):
        self.conductor.restart_trial()
        self._log_transport('restart trial')

    def _next(self):
        self.conductor.next_trial()
        self._log_transport('skip to next trial')

    def _seek(self, t):
        self.conductor.seek(t)
        self._log_transport(f'seek {t:.1f} s')

    def _log_transport(self, what):
        self.events_csv.write(sensor_us=self.pipe.clock.sensor_now(), exp_t=round(self.conductor.exp_t, 3),
                              kind='transport', text=what)
        self._add_log(f'[{self.conductor.exp_t:7.1f}] {what}')

    # --- events -----------------------------------------------------------------
    def _current_context(self):
        k = self.conductor.current_block_index()
        if k < 0:
            return {}
        b = self.schedule.blocks[k]
        return dict(trial=b.trial, rep=b.rep, attempt=self.conductor.attempts.get((b.trial, b.rep), ''))

    def _manual_event(self, text):
        ev = Event(self.pipe.clock.sensor_now(), 0, 'cue' if text in ('Start', 'Stop') else 'key', text)
        self.pipe.add_event(ev)
        self.events_csv.write(sensor_us=ev.t_us, exp_t=round(self.conductor.exp_t, 3), kind='manual',
                              text=text, event=text if text in ('Start', 'Stop') else '',
                              **self._current_context())
        self.track.add_event_marker(self.conductor.exp_t, text, EVENT_COLORS.get(text, '#7b3fb0'))
        self._add_log(f'[{self.conductor.exp_t:7.1f}] manual {text}')

    def _log_note(self):
        text = self.note.text().strip()
        if not text:
            return
        ev = Event(self.pipe.clock.sensor_now(), 0, 'note', text)
        self.pipe.add_event(ev)
        self.events_csv.write(sensor_us=ev.t_us, exp_t=round(self.conductor.exp_t, 3), kind='note',
                              text=text, **self._current_context())
        self.track.add_event_marker(self.conductor.exp_t, '✎', EVENT_COLORS['note'])
        self._add_log(f'[{self.conductor.exp_t:7.1f}] note: {text}')
        self.note.clear()

    def _add_log(self, s):
        self.log.insertItem(0, s)
        while self.log.count() > 300:
            self.log.takeItem(self.log.count() - 1)

    def _zero(self, name):
        self.people[name].zero(self.person_feet(name), self.pipe.now_s())

    # --- timers -----------------------------------------------------------------
    def _tick_cues(self):
        for em in self.conductor.tick():
            c = em.cue
            self.player.play(c.clip)
            sensor_us = self.pipe.clock.sensor_now()
            if c.event:
                self.pipe.add_event(Event(sensor_us, 0, 'cue', c.event))
                self.track.add_event_marker(em.exp_t, c.event, EVENT_COLORS[c.event])
                if self.gpio_pulse is not None and hasattr(self.pipe.source, 'set_output'):
                    self.pipe.source.set_output(self.gpio_pulse, 1)
                    QtCore.QTimer.singleShot(100, lambda: self.pipe.source.set_output(self.gpio_pulse, 0))
            self.events_csv.write(host_s=em.host_time, sensor_us=sensor_us, exp_t=round(em.exp_t, 3),
                                  kind='cue', text=c.text, event=c.event, trial=c.trial, rep=c.rep,
                                  bout=c.bout, walker=c.walker, attempt=em.attempt)
            if c.kind == 'go' and self.auto_zero.isChecked() and c.walker:
                self._zero(c.walker.upper())
            self._add_log(f'[{em.exp_t:7.1f}] {c.track}: {c.text}' + (f'  → {c.event}' if c.event else '')
                          + (f'  (attempt {em.attempt})' if em.attempt > 1 else ''))

    def _tick_data(self):
        self.pipe.step()
        # hardware / replay events (Opal buttons, sync box) onto the Events lane
        evs = self.pipe.events
        for ev in evs[self._n_events_seen:]:
            if ev.kind in ('button', 'sync_in'):
                self.track.add_event_marker(self.conductor.exp_t, ev.text, EVENT_COLORS[ev.kind])
                self.events_csv.write(sensor_us=ev.t_us, exp_t=round(self.conductor.exp_t, 3), kind=ev.kind,
                                      text=ev.text, device_id=ev.device_id, **self._current_context())
                self._add_log(f'[{self.conductor.exp_t:7.1f}] Opal {ev.kind}: {ev.text}')
        self._n_events_seen = len(evs)

    def _tick_draw(self):
        c = self.conductor
        self.track.follow(c.exp_t)
        self.track.mark_played(c._next)
        self.play_btn.setText('⏸  Pause' if c.playing else '▶  Play')
        k = c.current_block_index()
        expected = None
        walker_now = None
        if k >= 0:
            b = self.schedule.blocks[k]
            txt = f'Trial {b.trial}  rep {b.rep}'
            if self.conditions is not None:
                row = self.conditions[self.conditions['trial'] == b.trial]
                if len(row):
                    r = row.iloc[0]
                    expected = float(r['distance_m'])
                    txt += f'  ·  {expected:g} m, {r["package"]}, {r["pose"]}'
            att = c.attempts.get((b.trial, b.rep), 0)
            if att > 1:
                txt += f'  ·  attempt {att}'
            self.trial_lbl.setText(txt)
            gos = [q for q in self.schedule.cues if (q.trial, q.rep) == (b.trial, b.rep) and q.kind == 'go']
            walker_now = next((q.walker for q in reversed(gos) if q.t_s <= c.exp_t), None)
        else:
            self.trial_lbl.setText('before the first trial')
        nxt = c.next_cue()
        state = '' if c.playing else '   ⏸ PAUSED'
        self.next_lbl.setText((f'next: {nxt.text[:34]} in {nxt.t_s - c.exp_t:5.1f} s' if nxt else 'end of track') + state)
        # people
        now = self.pipe.now_s()
        lines = []
        for ev in self.pipe.events:
            x = self.pipe.session_s(ev.t_us) - now
            if -self.pipe.window_s <= x <= 0.5 and ev.kind in ('cue', 'button', 'key', 'note'):
                lines.append((x, EVENT_COLORS.get(ev.text, EVENT_COLORS.get(ev.kind, '#555')), ev.text[:12]))
        for name, pv in self.people.items():
            pv.update(self.pipe, self.person_feet(name),
                      expected if walker_now and walker_now.upper() == name else None)
            pv.set_event_lines(lines)
        self._update_health()
        self._update_camera()
        rec = self.pipe.recorder
        self.rec_lbl.setText(('● REC ' + os.path.basename(rec.path)) if rec else 'not recording')
        self.rec_lbl.setStyleSheet('color: #c8372d; font-weight: bold' if rec else 'color: #888')

    def _update_health(self):
        feet = list(self.pipe.feet.items())
        self.health.setRowCount(len(feet))
        now = time.time()
        for r, (dev, fs) in enumerate(feet):
            age = (now - fs.last_host) * 1e3 if fs.last_host else float('nan')
            vals = [fs.label, self.roles.get(dev, ''), f'{self.pipe.rate_hz(fs):.0f}',
                    f'{age:.0f}', str(fs.missing)]
            for col, val in enumerate(vals):
                it = self.health.item(r, col)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    self.health.setItem(r, col, it)
                it.setText(val)
                bad = (col == 3 and age > 500) or (col == 4 and fs.missing > 0)
                it.setForeground(QtGui.QColor('#c8372d' if bad else '#222'))

    def _update_camera(self):
        v = self.video
        if v is None:
            return
        f = v.latest_preview
        if f is not None:
            rgb = np.ascontiguousarray(f[:, :, ::-1])
            h, w = rgb.shape[:2]
            img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
            self.cam_view.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
                self.cam_view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        bad = v.dropped > 0 or v.errors or v.measured_fps() < 0.8 * v.fps
        self.cam_lbl.setText(('● CAM  ' if not bad else '⚠ CAM  ') + v.status()
                             + (f'\n{v.errors[-1][:60]}' if v.errors else ''))
        self.cam_lbl.setStyleSheet('font-size: 11px; color: ' + ('#c8372d' if bad else '#222'))

    def closeEvent(self, ev):
        self.conductor.pause()
        self.events_csv.close()
        super().closeEvent(ev)
