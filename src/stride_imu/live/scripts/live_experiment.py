"""Live experiment console: cue track + live feet for two walkers.

Needs the GUI deps once:  uv sync --group live

Rehearse with a recorded session (no hardware):
    uv run python src/stride_imu/live/scripts/live_experiment.py \\
        --replay "$HOME/Dropbox/Treadmill Brock 2025/imu data/imuData_s07_s08_20260708.h5" --start-s 1000 \\
        --conditions "$HOME/Dropbox/Treadmill Brock 2025/imu data/trialtable_20260708.csv"

Live (Motion Studio closed, Opals configured with src/stride_imu/live/scripts/live_feet.py --apdm --configure):
    uv run python src/stride_imu/live/scripts/live_experiment.py --apdm --conditions <trialtable.csv> --out sessions/s09_s10

Add a webcam with --camera 0 (--list-cameras to see indices; --synthetic-camera
to rehearse). Video goes to video_000.mp4, video_001.mp4, ... (30-min segments)
plus video_frames.csv (per-frame host + Opal-clock time).

Outputs in --out: session.h5 (all Opals + Start/Stop/notes as Annotations,
APDM layout), events.csv (every cue/event with trial context, host + Opal
clock), schedule.csv (the exact track used — edit and pass back with
--schedule), roles.json (who wore what).
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))   # src/


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--replay', metavar='H5')
    src.add_argument('--apdm', action='store_true')
    ap.add_argument('--start-s', type=float, default=0.0, help='replay: session second to start at')
    ap.add_argument('--duration-s', type=float, default=3600.0)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--latency-ms', type=int, default=250)
    ap.add_argument('--conditions', help='trialtable CSV (load_condition_table format)')
    ap.add_argument('--schedule', help='use this schedule CSV instead of building one')
    ap.add_argument('--trials', help='subset, e.g. 1-10 or 3,5,9')
    ap.add_argument('--reps', default='1,2')
    ap.add_argument('--first-walker', default='a', choices=['a', 'b'])
    ap.add_argument('--walk-speed', type=float, default=1.0, help='m/s used to time Stop after Go')
    ap.add_argument('--handoff-s', type=float, default=3.0)
    ap.add_argument('--between-trials-s', type=float, default=8.0)
    ap.add_argument('--voice', default='Samantha')
    ap.add_argument('--out', help='session output folder (default live_sessions/<timestamp>)')
    ap.add_argument('--no-record', action='store_true', help='do not write session.h5')
    ap.add_argument('--mute', action='store_true')
    ap.add_argument('--gpio-pulse', type=int, metavar='AP',
                    help='pulse the access point output on every Start/Stop cue (hardware)')
    ap.add_argument('--no-assign', action='store_true', help='skip the Opal assignment dialog at start')
    ap.add_argument('--camera', type=int, metavar='INDEX', help='record this webcam (see --list-cameras)')
    ap.add_argument('--synthetic-camera', action='store_true', help='test pattern instead of a webcam')
    ap.add_argument('--list-cameras', action='store_true')
    ap.add_argument('--cam-size', default='1280x720')
    ap.add_argument('--cam-fps', type=float, default=30)
    ap.add_argument('--cam-bitrate', default='3M', help='3M ~ 1.35 GB/hour')
    ap.add_argument('--segment-min', type=float, default=30, help='video file length [min]')
    ap.add_argument('--autoplay', action='store_true')
    ap.add_argument('--screenshot', help='(testing) save a window grab here, then quit')
    ap.add_argument('--quit-after', type=float, default=8.0)
    if argv is None and '--list-cameras' in sys.argv:
        from stride_imu.live.video import list_cameras
        print(list_cameras())
        return 0
    a = ap.parse_args(argv)

    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    from brock_functions import load_condition_table
    from stride_imu.live import BridgeSource, HdfRecorder, LivePipeline, ReplaySource
    from stride_imu.live.app import ExperimentWindow
    from stride_imu.live.audio import synthesize_clips
    from stride_imu.live.schedule import BrockTiming, Schedule, build_brock_schedule

    out = a.out or os.path.join('live_sessions', f'{datetime.datetime.now():%Y%m%d-%H%M%S}')
    os.makedirs(out, exist_ok=True)

    conditions = load_condition_table(a.conditions) if a.conditions else None
    if a.schedule:
        schedule = Schedule.load_csv(a.schedule)
    elif conditions is not None:
        trials = None
        if a.trials:
            trials = []
            for part in a.trials.split(','):
                lo, _, hi = part.partition('-')
                trials += list(range(int(lo), int(hi or lo) + 1))
        timing = BrockTiming(walk_speed_ms=a.walk_speed, handoff_s=a.handoff_s,
                             between_trials_s=a.between_trials_s)
        schedule = build_brock_schedule(conditions, reps=[int(r) for r in a.reps.split(',')],
                                        first_walker=a.first_walker, timing=timing, trials=trials)
    else:
        ap.error('give --conditions (trial table) or --schedule')
    cache = os.path.join(os.path.expanduser('~'), '.cache', 'stride_imu_cues')
    synthesize_clips(schedule, cache, voice=a.voice)
    schedule.save_csv(os.path.join(out, 'schedule.csv'))

    if a.replay:
        source = ReplaySource(a.replay, a.start_s, a.duration_s, patterns=('',), speed=a.speed)
    else:
        source = BridgeSource(latency_ms=a.latency_ms)
    source.start()
    recorder = None if a.no_record else HdfRecorder(os.path.join(out, 'session.h5'),
                                                    source.devices, source.period)
    pipe = LivePipeline(source, recorder)
    pipe.setup()

    video = None
    if a.camera is not None or a.synthetic_camera:
        from stride_imu.live.video import CameraSource, SyntheticCamera, VideoRecorder
        w, h = (int(x) for x in a.cam_size.lower().split('x'))
        cam = (SyntheticCamera(w, h, a.cam_fps) if a.synthetic_camera
               else CameraSource(a.camera, w, h, a.cam_fps))
        video = VideoRecorder(out, cam, sensor_now=lambda host: pipe.clock.sensor_now(host) if pipe.clock.ready else '',
                              segment_s=a.segment_min * 60, bitrate=a.cam_bitrate)
        video.start()
        print(f'video: {video.encoder}, {cam.fps:g} fps, {a.segment_min:g}-min segments, '
              f'~{float(a.cam_bitrate.rstrip("Mm")) * 3600 / 8 / 1000:.2f} GB/hour at {a.cam_bitrate}')

    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ExperimentWindow(pipe, schedule, conditions=conditions, out_dir=out, mute=a.mute,
                           gpio_pulse=a.gpio_pulse, video=video)
    win.show()
    if not a.no_assign and not a.screenshot:
        QtCore.QTimer.singleShot(300, win._assign)
    if a.autoplay:
        win.conductor.play()
    if a.screenshot:
        def snap():
            win.grab().save(a.screenshot)
            app.quit()
        QtCore.QTimer.singleShot(int(a.quit_after * 1000), snap)
    try:
        rc = app.exec()
    finally:
        win.events_csv.close()
        if video is not None:
            video.stop()
            if video.errors:
                print('video warnings:', *video.errors[-5:], sep='\n  ')
        source.stop()
        if recorder:
            recorder.close()
        print('session saved in', out)
    return rc


if __name__ == '__main__':
    sys.exit(main())
