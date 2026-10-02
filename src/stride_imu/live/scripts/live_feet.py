"""Live foot-IMU viewer / recorder.

Replay a recorded session (no hardware needed):
    uv run python src/stride_imu/live/scripts/live_feet.py --replay "<imu data>/imuData_s07_s08_20260708.h5" --start-s 1010 --duration-s 60

Live Opals through the access point (after `src/stride_imu/live/scripts/build_apdm_bridge.sh` and
`--configure` with the Opals docked):
    uv run python src/stride_imu/live/scripts/live_feet.py --apdm --configure     # docked: configure for streaming
    uv run python src/stride_imu/live/scripts/live_feet.py --apdm --record session.h5

Headless (no window): add --headless SECONDS; prints stride stats at the end.
Save a snapshot of the view: --svg out.svg (with --headless).
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))   # src/

from stride_imu.live import (BridgeSource, HdfRecorder, LivePipeline,  # noqa: E402
                             ReplaySource, bridge_command)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--replay', metavar='H5', help='session .h5 to replay in real time')
    src.add_argument('--apdm', action='store_true', help='live Opals via the access point')
    ap.add_argument('--start-s', type=float, default=0.0)
    ap.add_argument('--duration-s', type=float, default=120.0)
    ap.add_argument('--speed', type=float, default=1.0, help='replay speed factor')
    ap.add_argument('--patterns', default='foot', help='comma-separated label substrings (replay)')
    ap.add_argument('--configure', action='store_true',
                    help='(with --apdm) configure docked Opals + AP for streaming, then exit')
    ap.add_argument('--rate', type=int, default=128)
    ap.add_argument('--latency-ms', type=int, default=250,
                    help='max wireless latency the AP waits for retries (lower = fresher, more drops)')
    ap.add_argument('--no-sd', action='store_true', help='(configure) do NOT also log to the Opal SD card')
    ap.add_argument('--record', metavar='H5', help='host-side recording path')
    ap.add_argument('--window-s', type=float, default=10.0)
    ap.add_argument('--headless', type=float, metavar='SECONDS')
    ap.add_argument('--svg', help='(headless) save a snapshot of the viewer figure')
    a = ap.parse_args(argv)

    if a.apdm and a.configure:
        cmd = bridge_command('configure', str(a.rate), '0' if a.no_sd else '1')
        return subprocess.call(cmd)

    if a.replay:
        source = ReplaySource(a.replay, a.start_s, a.duration_s,
                              patterns=a.patterns.split(','), speed=a.speed)
    else:
        source = BridgeSource(latency_ms=a.latency_ms)
        if not a.record:
            a.record = f'live_{datetime.datetime.now():%Y%m%d-%H%M%S}.h5'
    source.start()
    print('devices:', source.devices, ' period:', source.period)

    recorder = HdfRecorder(a.record, source.devices, source.period) if a.record else None
    pipe = LivePipeline(source, recorder, window_s=a.window_s)
    pipe.setup()
    try:
        if a.headless is not None:
            t_end = time.monotonic() + a.headless
            while time.monotonic() < t_end and not getattr(source, 'finished', False):
                pipe.step()
                time.sleep(0.01)
            pipe.step()
            if a.svg:
                import matplotlib
                matplotlib.use('Agg')
                from stride_imu.live.viewer import LiveViewer
                v = LiveViewer(pipe)
                v.update()
                v.fig.savefig(a.svg)
                print('saved', a.svg)
            for fs in pipe.feet.values():
                st = [s for s in fs.strides if s.length_m == s.length_m]
                print(f'{fs.label}: {len(fs.foot.footfalls)} footfalls, {len(st)} strides')
            for e in pipe.events:
                print(f'  event {pipe.session_s(e.t_us):7.2f} s  {e.kind:8s} {e.text}')
            print(f'mechanization load: {100 * pipe.compute_load():.1f}% of one core')
        else:
            from stride_imu.live.viewer import LiveViewer, ensure_gui_backend
            ensure_gui_backend()
            LiveViewer(pipe).run()
    finally:
        source.stop()
        if recorder:
            recorder.close()
            print('recorded to', a.record)


if __name__ == '__main__':
    sys.exit(main())
