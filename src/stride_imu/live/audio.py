"""Spoken cue clips: synthesis (macOS `say`) and low-latency playback.

``synthesize_clips(schedule, cache_dir)`` renders every distinct cue text to
a WAV once (cached by voice+rate+text hash), stores the path in ``cue.clip``
and sets ``cue.duration_s`` from the file, so the GUI lanes draw real clip
lengths. Replace any WAV in the cache with a recorded human voice and it is
used as is (same file name).

``CuePlayer`` preloads every clip as a QSoundEffect (in-process, ~10-30 ms
start latency) and falls back to `afplay` if Qt Multimedia is unavailable.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import wave
from typing import Dict

from .schedule import Schedule

DEFAULT_VOICE = 'Samantha'


def clip_path(cache_dir: str, text: str, voice: str, rate: int) -> str:
    h = hashlib.sha1(f'{voice}|{rate}|{text}'.encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f'cue_{h}.wav')


def wav_duration(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


def synthesize_clips(schedule: Schedule, cache_dir: str, voice: str = DEFAULT_VOICE,
                     rate: int = 190) -> Dict[str, float]:
    os.makedirs(cache_dir, exist_ok=True)
    durations: Dict[str, float] = {}
    for c in schedule.cues:
        p = clip_path(cache_dir, c.text, voice, rate)
        if not os.path.exists(p):
            cmd = ['say', '-v', voice, '-r', str(rate), '-o', p, '--data-format=LEI16@22050', c.text]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:       # voice missing -> system default
                subprocess.run([a for a in cmd if a not in ('-v', voice)], check=True, capture_output=True)
        c.clip = p
        if p not in durations:
            durations[p] = wav_duration(p)
    schedule.retime_durations(durations)
    return durations


class CuePlayer:
    def __init__(self, schedule: Schedule, volume: float = 1.0):
        self.effects = {}
        self.muted = False
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect
            for c in schedule.cues:
                if c.clip and c.clip not in self.effects:
                    e = QSoundEffect()
                    e.setSource(QUrl.fromLocalFile(c.clip))
                    e.setVolume(volume)
                    self.effects[c.clip] = e
            self.backend = 'qt'
        except Exception:
            self.backend = 'afplay'

    def play(self, clip: str):
        if self.muted or not clip:
            return
        e = self.effects.get(clip)
        if e is not None:
            e.play()
        else:
            subprocess.Popen(['afplay', clip])
