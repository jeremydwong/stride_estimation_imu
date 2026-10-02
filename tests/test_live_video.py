"""Segmented webcam recorder with a synthetic camera (needs ffmpeg + opencv)."""
import os
import subprocess
import sys
import tempfile
import time
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import cv2  # noqa: F401
    from stride_imu.live.video import SyntheticCamera, VideoRecorder, ffmpeg_exe
    FFMPEG = ffmpeg_exe()
except Exception:                                    # pragma: no cover
    FFMPEG = None


def n_frames(path):
    probe = os.path.join(os.path.dirname(FFMPEG), 'ffprobe')
    if not os.path.exists(probe):
        probe = 'ffprobe'
    r = subprocess.run([probe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                        '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', path],
                       capture_output=True, text=True)
    return int(r.stdout.strip())


@unittest.skipUnless(FFMPEG, 'ffmpeg/opencv not available (uv sync --group live)')
class VideoRecorderTests(unittest.TestCase):
    def test_segments_match_frame_log(self):
        with tempfile.TemporaryDirectory() as d:
            rec = VideoRecorder(d, SyntheticCamera(320, 240, 20), sensor_now=lambda h: int(h * 1e6),
                                segment_s=1.0)
            rec.start()
            time.sleep(3.6)
            rec.stop()
            self.assertEqual(rec.errors, [])
            f = pd.read_csv(os.path.join(d, 'video_frames.csv'))
            kept = f[f.dropped == 0]
            self.assertGreaterEqual(kept.segment.nunique(), 3)
            for seg, g in kept.groupby('segment'):
                path = os.path.join(d, f'video_{int(seg):03d}.mp4')
                self.assertEqual(n_frames(path), len(g))
                self.assertEqual(list(g.frame_in_segment), list(range(len(g))))
            self.assertTrue((f.host_s.diff().dropna() > 0).all())

    def test_capture_drops_instead_of_blocking(self):
        """With the writer stalled, capture keeps its rate and counts drops."""
        with tempfile.TemporaryDirectory() as d:
            rec = VideoRecorder(d, SyntheticCamera(160, 120, 50), queue_s=0.1)
            import threading
            t = threading.Thread(target=rec._capture_loop, daemon=True)   # no writer
            t0 = time.time()
            t.start()
            time.sleep(1.0)
            rec._stop.set()
            t.join(2)
            rec._csv_f.close()
            self.assertGreater(rec.dropped, 30)
            self.assertAlmostEqual(rec.captured / (time.time() - t0), 50, delta=10)


if __name__ == '__main__':
    unittest.main()
