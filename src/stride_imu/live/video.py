"""Webcam recording for multi-hour sessions.

Design (why it survives a 2-hour experiment):

  camera --capture thread--> bounded queue --writer thread--> ffmpeg (hardware H.264)
             |  timestamps each frame            |  rotates to a new file
             |  never blocks on disk/encoder     |  every `segment_s` (30 min)
             v                                   v
        latest frame (GUI preview)      video_000.mp4, video_001.mp4, ... + frames.csv

* The capture thread only grabs + timestamps. If the writer falls behind the
  queue fills and NEW frames are dropped (counted, and logged in frames.csv),
  so capture timing is never disturbed.
* Encoding runs in an ffmpeg child process on Apple's VideoToolbox encoder
  (~1-3% CPU for 720p30); libx264 is the fallback. 720p30 at 3 Mbit/s is
  ~1.35 GB/hour.
* Segments are rotated by FRAME COUNT in Python (not by ffmpeg's segment
  muxer), so frames.csv maps every frame to its exact file and frame number.
* Files are fragmented MP4: a crash or power loss leaves a playable file up
  to the last fragment (~1 s), not an unreadable one.
* frames.csv: frame, segment, frame_in_segment, host_s, sensor_us, dropped.
  host_s is the same clock as events.csv, so video lines up with cues
  directly; sensor_us maps to the Opal data via the live SensorClock.
  The camera's own exposure->delivery latency (typically 30-100 ms) is NOT
  in the timestamps; measure it once with a tap test (stomp a foot in view,
  compare the accel spike with the frame).
"""
from __future__ import annotations

import csv
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np


def ffmpeg_exe() -> str:
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def pick_encoder(exe: Optional[str] = None) -> str:
    exe = exe or ffmpeg_exe()
    out = subprocess.run([exe, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    return 'h264_videotoolbox' if 'h264_videotoolbox' in out else 'libx264'


def list_cameras() -> str:
    """Camera names/indices as AVFoundation sees them (does not open a camera)."""
    r = subprocess.run([ffmpeg_exe(), '-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                       capture_output=True, text=True)
    lines = [ln.split('] ', 1)[-1] for ln in r.stderr.splitlines() if 'AVFoundation' in ln]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class CameraSource:
    """OpenCV webcam (AVFoundation on macOS). ``read()`` blocks until the
    next frame and returns (frame BGR uint8, host time [s])."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: float = 30):
        import cv2
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(
                f'could not open camera {index}. On macOS, allow camera access for the app running '
                'Python (System Settings > Privacy & Security > Camera: Terminal / VS Code), '
                'and check --list-cameras.')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        ok, frame = self.cap.read()
        return (frame if ok else None), time.time()

    def close(self):
        self.cap.release()


class SyntheticCamera:
    """Moving test pattern at a fixed rate, paced to the wall clock (tests,
    and rehearsing without a camera)."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30):
        self.w, self.h, self.fps = width, height, fps
        self._t0 = None              # starts at the first read (no catch-up burst)
        self._n = 0
        yy, xx = np.mgrid[0:height, 0:width]
        self._base = ((xx // 40 + yy // 40) % 2 * 60 + 60).astype(np.uint8)

    def read(self):
        if self._t0 is None:
            self._t0 = time.monotonic()
        due = self._t0 + self._n / self.fps
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        f = np.repeat(self._base[:, :, None], 3, axis=2)
        x = int((self._n * 8) % self.w)
        f[:, max(0, x - 20):x + 20] = (40, 180, 255)
        self._n += 1
        return f, time.time()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class VideoRecorder:
    def __init__(self, out_dir: str, source, sensor_now: Optional[Callable[[float], int]] = None,
                 segment_s: float = 1800.0, bitrate: str = '3M', encoder: str = 'auto',
                 overlay: bool = True, queue_s: float = 4.0, preview_width: int = 480,
                 prefix: str = 'video'):
        self.out_dir = out_dir
        self.source = source
        self.sensor_now = sensor_now
        self.fps = float(source.fps)
        self.segment_frames = int(round(segment_s * self.fps))
        self.bitrate = bitrate
        self.exe = ffmpeg_exe()
        self.encoder = pick_encoder(self.exe) if encoder == 'auto' else encoder
        self.overlay = overlay
        self.prefix = prefix
        self.preview_width = preview_width
        self.q: 'queue.Queue' = queue.Queue(maxsize=max(2, int(queue_s * self.fps)))
        self.latest_preview: Optional[np.ndarray] = None
        self.captured = 0
        self.dropped = 0
        self.written = 0
        self.segment = -1
        self.errors: List[str] = []
        self._proc = None
        self._closing: List[threading.Thread] = []
        self._stop = threading.Event()
        self._recent: List[float] = []
        os.makedirs(out_dir, exist_ok=True)
        self._csv_f = open(os.path.join(out_dir, f'{prefix}_frames.csv'), 'w', newline='')
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(['frame', 'segment', 'frame_in_segment', 'host_s', 'sensor_us', 'dropped'])
        self._csv_lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------------
    def start(self):
        self._t_cap = threading.Thread(target=self._capture_loop, name='video-capture', daemon=True)
        self._t_wr = threading.Thread(target=self._writer_loop, name='video-writer', daemon=True)
        self._t_cap.start()
        self._t_wr.start()

    def stop(self, timeout: float = 30.0):
        self._stop.set()
        self._t_cap.join(timeout)
        self.q.put(None)
        self._t_wr.join(timeout)
        self._close_segment()
        for t in self._closing:
            t.join(timeout)
        self.source.close()
        with self._csv_lock:
            self._csv_f.close()

    # --- capture thread (never blocks on I/O) -----------------------------------
    def _capture_loop(self):
        while not self._stop.is_set():
            frame, host = self.source.read()
            if frame is None:
                self.errors.append(f'{time.strftime("%H:%M:%S")} camera read failed')
                time.sleep(0.05)
                continue
            idx = self.captured
            self.captured += 1
            sensor_us = self.sensor_now(host) if self.sensor_now else ''
            self._recent.append(host)
            if len(self._recent) > 60:
                self._recent.pop(0)
            if idx % 3 == 0:
                self.latest_preview = self._shrink(frame)
            try:
                self.q.put_nowait((idx, frame, host, sensor_us))
            except queue.Full:
                self.dropped += 1
                with self._csv_lock:
                    self._csv.writerow([idx, '', '', f'{host:.6f}', sensor_us, 1])

    def _shrink(self, frame):
        h, w = frame.shape[:2]
        step = max(1, w // self.preview_width)
        return frame[::step, ::step].copy()

    # --- writer thread ----------------------------------------------------------------
    def _open_segment(self, w: int, h: int):
        self.segment += 1
        self._seg_count = 0
        path = os.path.join(self.out_dir, f'{self.prefix}_{self.segment:03d}.mp4')
        enc = ['-c:v', self.encoder, '-b:v', self.bitrate]
        if self.encoder == 'libx264':
            enc += ['-preset', 'veryfast']
        cmd = [self.exe, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-framerate', f'{self.fps:g}',
               '-i', '-', *enc, '-g', str(int(round(self.fps))), '-pix_fmt', 'yuv420p',
               '-movflags', '+frag_keyframe+empty_moov+default_base_moof', path]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.segment_path = path

    def _close_segment(self):
        """Finish the current file in the background (ffmpeg flushes after
        stdin closes) so the next segment starts without a gap."""
        proc, self._proc = self._proc, None
        if proc is None:
            return

        def finish(p=proc, seg=self.segment):
            try:
                p.stdin.close()
            except Exception:
                pass
            err = p.stderr.read().decode(errors='replace').strip()
            if p.wait() != 0 or err:
                self.errors.append(f'segment {seg}: ffmpeg exit {p.returncode} {err[-300:]}')
        t = threading.Thread(target=finish, daemon=True)
        t.start()
        self._closing.append(t)

    def _stamp(self, frame, idx, host, sensor_us):
        import cv2
        txt = f'#{idx}  {time.strftime("%H:%M:%S", time.localtime(host))}.{int(host % 1 * 1000):03d}'
        if sensor_us != '':
            txt += f'  opal {sensor_us}'
        cv2.putText(frame, txt, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def _writer_loop(self):
        last_flush = time.monotonic()
        while True:
            item = self.q.get()
            if item is None:
                break
            idx, frame, host, sensor_us = item
            h, w = frame.shape[:2]
            if self._proc is None or self._seg_count >= self.segment_frames:
                self._close_segment()
                self._open_segment(w, h)
            if self.overlay:
                self._stamp(frame, idx, host, sensor_us)
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError) as e:
                # encoder died: this frame is lost; the next one opens a new
                # segment. (Frames already sent but not yet in a finished
                # fragment - up to ~1 s - are lost from the old file too.)
                self.errors.append(f'segment {self.segment}: write failed ({e}); reopening')
                self._proc = None
                self.dropped += 1
                with self._csv_lock:
                    self._csv.writerow([idx, '', '', f'{host:.6f}', sensor_us, 1])
                continue
            with self._csv_lock:
                self._csv.writerow([idx, self.segment, self._seg_count, f'{host:.6f}', sensor_us, 0])
            self._seg_count += 1
            self.written += 1
            if time.monotonic() - last_flush > 1.0:
                with self._csv_lock:
                    self._csv_f.flush()
                last_flush = time.monotonic()

    # --- status -------------------------------------------------------------------------
    def measured_fps(self) -> float:
        r = self._recent
        return (len(r) - 1) / (r[-1] - r[0]) if len(r) > 2 and r[-1] > r[0] else 0.0

    def bytes_written(self) -> int:
        total = 0
        for k in range(self.segment + 1):
            p = os.path.join(self.out_dir, f'{self.prefix}_{k:03d}.mp4')
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

    def status(self) -> str:
        return (f'seg {max(self.segment, 0)} · {self.measured_fps():4.1f} fps · '
                f'{self.dropped} dropped · queue {self.q.qsize()}/{self.q.maxsize} · '
                f'{_size(self.bytes_written())}')


def _size(n: int) -> str:
    return f'{n / 1e9:.2f} GB' if n >= 1e9 else f'{n / 1e6:.0f} MB'
