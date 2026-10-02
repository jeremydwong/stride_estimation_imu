"""Open-loop cue schedule ("the play track") and its conductor.

A ``Schedule`` is a list of ``Cue``s on EXPERIMENT time (seconds from the
start of the session's track, which stops while paused). Each cue belongs to
a track (lane in the GUI), is spoken as a short audio clip, and may carry an
``event`` ('Start' / 'Stop') that is written to the recording the moment the
cue plays — the automatic replacement for the experimenter's button presses.

``build_brock_schedule`` turns the Brock trial table into a track:
per trial-rep an announcement, then for each bout "Person X, ready" and
"Go" (-> Start) and, after the expected walk + hand-off time, "Stop"
(-> Stop). Walkers follow the protocol (rep 1: ``first_walker`` walks bout 1,
rep 2 flipped). All timing lives in ``BrockTiming`` — these defaults are a
first guess, edit them or hand-edit the saved schedule CSV.

``Conductor`` plays a schedule: ``tick()`` returns the cues that became due,
``pause``/``play``, ``back_trial`` (to the previous trial), ``restart_trial``,
``next_trial``. Re-running a trial increments its ``attempt`` so the log
tells the takes apart.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional, Tuple

TRACKS = ('Announce', 'Person A', 'Person B')
KIND_COLORS = {                     # also used by the GUI lanes
    'announce': '#6c7a89',
    'ready': '#e0a526',
    'go': '#2a9d3f',
    'stop': '#c8372d',
    'rest': '#9aa5b1',
    'note': '#7b3fb0',
}


@dataclass
class Cue:
    t_s: float                      # experiment time the clip starts
    track: str                      # lane: 'Announce' | 'Person A' | 'Person B'
    kind: str                       # 'announce' | 'ready' | 'go' | 'stop' | ...
    text: str                       # what is spoken
    event: str = ''                 # '' | 'Start' | 'Stop'  (written to the recording)
    trial: int = 0
    rep: int = 0
    bout: int = 0
    walker: str = ''                # 'a' | 'b'
    duration_s: float = 1.0         # clip length (filled in by audio synthesis)
    clip: str = ''                  # path of the audio file


@dataclass
class TrialBlock:
    trial: int
    rep: int
    t_start: float
    t_end: float
    label: str = ''


@dataclass
class BrockTiming:
    announce_to_ready_s: float = 5.0   # announcement -> first "ready"
    ready_to_go_s: float = 2.5
    walk_speed_ms: float = 1.0         # used to time "Stop" after "Go"
    handoff_s: float = 3.0             # time allowed for the hand-off
    stop_to_next_ready_s: float = 4.0  # between bout 1's Stop and bout 2's ready
    between_trials_s: float = 8.0      # after the last Stop before the next announcement
    start_offset_s: float = 3.0        # first announcement


POSE_TEXT = {'ground': 'hand off at the ground', 'held out': 'hand off held out',
             'waist': 'hand off at the waist'}


def _num(x: float) -> str:
    return f'{x:g}'


class Schedule:
    def __init__(self, cues: List[Cue], blocks: Optional[List[TrialBlock]] = None):
        self.cues = sorted(cues, key=lambda c: c.t_s)
        self.blocks = blocks if blocks is not None else self._infer_blocks()

    def _infer_blocks(self) -> List[TrialBlock]:
        by: Dict[Tuple[int, int], List[Cue]] = {}
        for c in self.cues:
            if c.trial:
                by.setdefault((c.trial, c.rep), []).append(c)
        blocks = [TrialBlock(t, r, min(c.t_s for c in cs), max(c.t_s + c.duration_s for c in cs),
                             f'T{t} r{r}') for (t, r), cs in by.items()]
        return sorted(blocks, key=lambda b: b.t_start)

    @property
    def end_s(self) -> float:
        return max((c.t_s + c.duration_s for c in self.cues), default=0.0)

    def retime_durations(self, durations: Dict[str, float]):
        """After synthesis: set each cue's duration from its clip."""
        for c in self.cues:
            if c.clip in durations:
                c.duration_s = durations[c.clip]
        self.blocks = self._infer_blocks()

    # --- CSV round trip (hand-editable) ---------------------------------------
    def save_csv(self, path: str):
        names = [f.name for f in fields(Cue)]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=names)
            w.writeheader()
            for c in self.cues:
                w.writerow(asdict(c))

    @classmethod
    def load_csv(cls, path: str) -> 'Schedule':
        types = {f.name: f.type for f in fields(Cue)}
        cues = []
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                kw = {}
                for k, v in row.items():
                    t = types.get(k)
                    if t in ('float', float):
                        kw[k] = float(v)
                    elif t in ('int', int):
                        kw[k] = int(float(v)) if v != '' else 0
                    else:
                        kw[k] = v
                cues.append(Cue(**kw))
        return cls(cues)


def build_brock_schedule(conditions, reps=(1, 2), first_walker: str = 'a',
                         timing: BrockTiming = BrockTiming(),
                         trials: Optional[List[int]] = None) -> Schedule:
    """Cue track for the Brock hand-off protocol from ``load_condition_table``
    output (columns trial, distance_m, package, pose)."""
    T = timing
    other = {'a': 'b', 'b': 'a'}
    rows = conditions if trials is None else conditions[conditions['trial'].isin(trials)]
    cues: List[Cue] = []
    t = T.start_offset_s
    for rep in reps:
        for _, r in rows.iterrows():
            trial = int(r['trial'])
            d = float(r['distance_m'])
            pose = POSE_TEXT.get(str(r['pose']).strip().lower(), str(r['pose']))
            cues.append(Cue(t, 'Announce', 'announce',
                            f'Trial {trial}. {_num(d)} meters. {r["package"]} box. {pose}.',
                            trial=trial, rep=rep, duration_s=4.0))
            t += T.announce_to_ready_s
            w1 = first_walker if rep == 1 else other[first_walker]
            for bout, walker in ((1, w1), (2, other[w1])):
                track = f'Person {walker.upper()}'
                kw = dict(trial=trial, rep=rep, bout=bout, walker=walker)
                cues.append(Cue(t, track, 'ready', f'Person {walker.upper()}, ready.', **kw))
                t += T.ready_to_go_s
                cues.append(Cue(t, track, 'go', 'Go.', event='Start', **kw))
                t += d / T.walk_speed_ms + T.handoff_s
                cues.append(Cue(t, track, 'stop', 'Stop.', event='Stop', **kw))
                t += T.stop_to_next_ready_s if bout == 1 else T.between_trials_s
    return Schedule(cues)


@dataclass
class Emission:
    cue: Cue
    exp_t: float
    host_time: float
    attempt: int


class Conductor:
    """Plays a Schedule on a pausable experiment clock.

    Call ``tick()`` often (the GUI does every 10 ms); it returns the cues
    whose start time was crossed since the last tick, in order. Seeking never
    emits the skipped cues."""

    def __init__(self, schedule: Schedule, clock: Callable[[], float] = time.monotonic,
                 preroll_s: float = 1.0):
        self.schedule = schedule
        self.clock = clock
        self.preroll_s = preroll_s
        self.exp_t = 0.0
        self.playing = False
        self._last_wall: Optional[float] = None
        self._next = 0
        self.attempts: Dict[Tuple[int, int], int] = {}
        self.log: List[Emission] = []
        self.listeners: List[Callable[[Emission], None]] = []

    # --- transport --------------------------------------------------------------
    def play(self):
        if not self.playing:
            self.playing = True
            self._last_wall = self.clock()

    def pause(self):
        self._advance()
        self.playing = False

    def toggle(self):
        self.pause() if self.playing else self.play()

    def seek(self, t: float):
        self._advance()
        self.exp_t = max(0.0, t)
        cues = self.schedule.cues
        self._next = next((k for k, c in enumerate(cues) if c.t_s >= self.exp_t), len(cues))

    def current_block_index(self) -> int:
        """Index of the trial block the playhead is in (or last started)."""
        blocks = self.schedule.blocks
        idx = -1
        for k, b in enumerate(blocks):
            if b.t_start - self.preroll_s <= self.exp_t + 1e-9:
                idx = k
        return idx

    def _seek_block(self, k: int):
        blocks = self.schedule.blocks
        if not blocks:
            return
        k = min(max(k, 0), len(blocks) - 1)
        self.seek(blocks[k].t_start - self.preroll_s)

    def back_trial(self):
        """To the start of the PREVIOUS trial block."""
        self._seek_block(self.current_block_index() - 1)

    def restart_trial(self):
        self._seek_block(max(self.current_block_index(), 0))

    def next_trial(self):
        self._seek_block(self.current_block_index() + 1)

    # --- clock ---------------------------------------------------------------------
    def _advance(self):
        if self.playing and self._last_wall is not None:
            now = self.clock()
            self.exp_t += now - self._last_wall
            self._last_wall = now

    def tick(self) -> List[Emission]:
        self._advance()
        out = []
        cues = self.schedule.cues
        while self._next < len(cues) and cues[self._next].t_s <= self.exp_t:
            c = cues[self._next]
            self._next += 1
            key = (c.trial, c.rep)
            if c.kind == 'announce' or key not in self.attempts:
                self.attempts[key] = self.attempts.get(key, 0) + 1
            em = Emission(c, self.exp_t, time.time(), self.attempts.get(key, 1))
            self.log.append(em)
            out.append(em)
            for fn in self.listeners:
                fn(em)
        return out

    def next_cue(self) -> Optional[Cue]:
        cues = self.schedule.cues
        return cues[self._next] if self._next < len(cues) else None
