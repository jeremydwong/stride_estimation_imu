# APDM Opal live streaming: design report

*2026-09-29. Covers the APDM SDK research, what your hardware and recordings
show, and the live pipeline built in `src/stride_imu/live/`. That pipeline has
been tested on replayed Brock data but not yet on live hardware.*

## TL;DR

- **Live access exists and is already on this Mac.** Motion Studio bundles
  APDM's SDK: a C library `libapdm.dylib`, the Java binding `apdm.jar`, and
  firmware. The SDK can stream every Opal through the access point (AP),
  deliver **Opal button events live**, read **sync-box input edges**, and
  **drive the AP's output line**.
- **The catch:** the official Python binding is Python 2.7 only, and
  `libapdm.dylib` is x86_64 only. I sidestepped both with a **small Java
  bridge**. It runs on Motion Studio's own x86_64 JRE under Rosetta, uses
  APDM's documented Java API, and streams text lines to Python 3 (native
  arm64). It compiles, loads `libapdm`, and scans USB for an AP. It stops
  there because no AP is plugged in.
- **Logging can stay on the devices.** Streaming mode can also log to the
  Opal's flash ("robust streaming", `enable_sd_card`), so dropped radio
  packets are recoverable after docking. I recommend logging in **both**
  places: on the device as the ground truth, and on the host with the new
  `HdfRecorder`, which writes APDM's own .h5 layout so all existing loaders
  read it.
- **Live mechanization works, and no C is needed.** `LiveFoot` is a causal
  re-implementation of `compute_position`. It is **bit-identical** to the
  offline result: same footfalls, quaternions, `An`, `Anz`, `V` and `P`,
  verified on pilot data and synthetic gait. It costs ~35 µs per sample, so
  four feet at 128 Hz use about **2% of one core** in pure Python.
  **Latency** is one stance plus 0.4 s: a stride is final ~0.6–0.8 s after its
  footfall. A provisional (dead-reckoned) speed fills the gap on screen.
- **Try it now, without hardware:**
  `uv run python src/stride_imu/live/scripts/live_feet.py --replay "<imu data>/imuData_s07_s08_20260708.h5" --start-s 1010 --duration-s 60`
  This replays a real Brock session in real time: two walkers, four feet,
  live foot speed, stride length and speed, and the Start/Stop button events.

---

## 1. What your recordings say about the current setup

Read from the `Configuration` attributes of the Brock session files
(s01_s02, s03_s04 and s07_s08) and the pilot files:

| Item | Value | Meaning |
|---|---|---|
| Hardware | Opal **v2**, firmware `V2-STM-20190315` | Current generation (v2 has the LCD); SDK's `apdm_ctx_v2_*` path |
| `Wireless Protocol` | **1 = synchronized logging** | Data goes to each Opal's flash and is imported after docking. **Nothing is streamed today.** |
| `SI Logging Enabled` | 1 | Calibrated SI data logged on the device |
| `Wireless Latency (ms)` | 65535 | Streaming latency unused (logging mode) |
| `Sample Rate` | 128 Hz (s01_s02: 100 Hz) | |
| Sensors per session | 6: `left/right_foot_a`, `left/right_foot_b`, **`event`**, `box` | |
| Annotations | 192 `Start` / 189 `Stop`, all from Sensor ID 21789 = the **`event`** Opal | Button presses on a 6th Opal, used as a clicker |
| AP | `Wireless Target AP ID` 21710 | One access point |

So the "event logging" you use today is **Opal button presses recorded in
logging mode**. It works because each Opal logs its own button transitions
and Motion Studio turns them into annotations on import. In streaming mode
the same presses arrive **live** as button events (section 3).

## 2. The live API: APDM SDK

### What APDM provides

- APDM's SDK is a C library (`libapdm`) with **Java** (`apdm.jar`) and
  **Python** (`apdm.py`, SWIG) bindings. APDM describes it as limited and
  **unsupported**: "sufficient for simple configuration and streaming", but
  not for converting logged recordings or for sync-box work (APDM support
  article; see Sources).
- **The Python binding is Python 2.7 only.** Both public example projects
  (aero-man/apdm-opal-simple-stream and apdm-opal-database) say so, and ship
  macOS setup notes for 2.7.
- **The SDK is already installed here**, inside Motion Studio (build
  2022-06-16):
  `/Applications/MotionStudio.app/Contents/Resources/configuration/org.eclipse.osgi/6/0/.cp/apdm_sdk/`
  - `libs/MacOSX/x64/libapdm.dylib`: **x86_64 only**. It exports the full C
    API (checked with `nm`).
  - `java/apdm.jar`: high-level `com.apdm.Context`, `RecordRaw`,
    `DockingStation`, plus SWIG classes. Also contains APDM's own example,
    `com.apdm.misc.ConfigureAndStreamData`.
  - Motion Studio's bundled JRE 8 is also **x86_64**, so it can load
    `libapdm` (under Rosetta on Apple silicon).

### Streaming sequence

This is the sequence APDM's examples use; I disassembled it from the jar and
the public examples.

```
docked Opals + AP plugged in:
  ctx = apdm_ctx_allocate_new_context(); apdm_ctx_open_all_access_points(ctx)
  apdm_init_streaming_config(cfg); cfg.{enable_accel,gyro,mag, output_rate_hz,
      enable_sd_card, button_enable, wireless_max_latency_ms, ...}
  apdm_ctx_autoconfigure_devices_and_accesspoint_streaming(ctx, cfg)
undock -> wait for AP + Opals to blink green together:
  apdm_ctx_open_all_access_points(ctx); apdm_ctx_sync_record_list_head(ctx)
  loop:
    apdm_ctx_get_next_access_point_record_list(ctx)      # one time-aligned sample set
    for each device: apdm_ctx_extract_data_by_device_id(ctx, id, &rec)
        rec.accl_*_si, gyro_*_si, mag_*_si, v2_sync_val64_us, button_status,
        orientation_quaternion0..3 (on-board estimate), ...
    apdm_ctx_get_next_button_event(ctx, &btn)            # live Opal button presses
    apdm_ctx_get_next_synchronization_event(ctx, &sync)  # sync-box input edges
  apdm_ctx_ap_set_io_value(ctx, ap, APDM_AP_GPIO_0, v)   # drive AP / sync-box output
```

In Java these are `Context.open / autoConfigureDevicesAndAccessPointStreaming /
syncRecordHeadList / getNextRecordList / getButtonEvent /
getSynchronizationEvent / setAPOutputGPIOValue / setMaxLatency`.
`getNextRecordList()` returns an empty list when no data is ready.

### Throughput, latency and range

From the Motion Studio User Guide / Opal spec sheet:

- One AP streams **up to 6 Opals at 128 Hz** (1600 samples/s per AP). Four
  feet (512 samples/s) fits, and so do all six current sensors (768). More
  than six sensors, or higher rates, needs a second AP.
- Latency: **~30 ms typical without the data buffer, ~300 ms with it**. The
  SDK's `wireless_max_latency_ms` / `setMaxLatency` sets how long the AP waits
  for retries, which trades freshness against dropped samples.
- Range: **30 m line of sight, 10 m indoors**. The Brock walks are
  metres-scale, so put the AP centrally and do a range test (section 7).
- Battery: streaming about **8 h**, synchronized logging about 12 h.
- Clock sync between Opals: under 1 ms.

### How to call it from Python 3 on Apple silicon

| Option | Status | Notes |
|---|---|---|
| **Java bridge → Python over a pipe** | **Built** (`bridge/ApdmBridge.java`) | Uses APDM's documented Java API with no struct guessing. It is a child process of the Python app. The Python side stays native arm64 with the normal env. |
| ctypes on `libapdm.dylib` | Possible later | Needs an x86_64 Python (the whole stack under Rosetta) plus reverse-engineered `apdm_record_t` / config struct layouts; the SWIG getters give the field names but not the offsets. Only worth it if the bridge proves too slow, which is unlikely. |
| Official `apdm.py` | No | Python 2.7 only |
| Motion Studio streaming GUI | Exists | Live strip charts plus annotation buttons and a presentation remote, but no programmatic output. It also holds the AP, so **Motion Studio must be closed** while the bridge runs. |
| LSL (Lab Streaming Layer) | No native APDM app found | The bridge could publish an LSL outlet later if you want to sync with other LSL devices. |

**Longevity risk:** the bridge depends on **Rosetta 2**. Apple has signalled
that Rosetta is being phased out for general apps over the next macOS
releases. If APDM never ships an arm64 `libapdm`, a Windows PC or an Intel
machine becomes the long-term fallback. The bridge's Java code runs unchanged
there; only the library path differs.

## 3. Events and logging: what goes where

| Event source | Logging mode (today) | Streaming mode (live) |
|---|---|---|
| **Opal buttons** (the `event` Opal) | Logged on device → annotations on import | `getButtonEvent()` delivers them live (`B` lines); `button_enable` must be set when configuring. Also recorded as annotations in the host .h5. |
| **Host keys** (new) | n/a | `s`=Start, `e`=Stop, `m`=mark in the viewer. Stamped with the sensor clock of the newest sample, so they read **late by the stream latency** (~30–300 ms); prefer the Opal button when timing matters. |
| **Sync box IN** (TTL from mocap, force plate, a foot switch) | Starts/stops recordings (Motion Studio) | `getSynchronizationEvent()` edges (`X` lines). Logic high ≥2.3 V, low ≤0.99 V, isolated. |
| **Sync box / AP OUT** | Output trigger on record start/stop | `setAPOutputGPIOValue` (`o` key pulses it) to **mark our events on other equipment**. |
| Presentation remote | Motion Studio streaming only | Not reachable via the SDK; a USB remote could be read on the host as key events. |

**Where to log: both places.**

1. **On the device.** Configure streaming with `enable_sd_card=1`, which is
   the default in `src/stride_imu/live/scripts/live_feet.py --apdm --configure`. The Opals then
   still log everything; after docking, Motion Studio's Import Manager
   recovers it. This is the complete, drop-free record.
2. **On the host.** `HdfRecorder` writes every received sample and every event
   as it arrives (flushed every 2 s; events immediately) into APDM's v5
   layout: `/Sensors/XI-<id>/{Accelerometer,Gyroscope,Magnetometer,Time}`,
   `Configuration` attrs `Label 0` and `Sample Rate`, and an `Annotations`
   table. `load_imu_recording`, `list_sensors` and `load_available_feet` open
   it unchanged (tested); Opal button text is stored verbatim, so `Start` /
   `Stop` keep working with the Brock event code.

### Can we write events TO the Opals? No.

I searched the whole SDK surface: the `apdm.jar` classes and every exported
`libapdm` symbol. There is **no host-to-Opal command that adds an event or
annotation to an Opal's log** while it streams.
- `apdm_write_annotation` writes to a host-side HDF5 file handle.
- `Device.cmd_*` commands (e.g. `cmd_set_device_button_event_0`, which sets
  the *text* a button press is logged with) are docking-station/USB
  configuration commands.
- `apdm_ap_enqueue_button_event` is the AP's internal queue of Opal→host
  button presses.
- `apdm_ctx_set_requested_device_state` only offers RUN / HALT.

The only things an Opal ever writes to its own flash are its own samples and
its **own physical button presses**.

What the live console does instead, so every Start/Stop still sits on the
data's timeline:
1. Each cue event is stamped in the **Opal clock**. `SensorClock` maps host
   time to Opal time using the least-delayed recent packet, so the error is
   about the fastest packet's latency. The event is written to the host
   `session.h5` Annotations as bare `Start`/`Stop` with **Sensor ID 0**,
   which `brock_functions.load_events` reads like the clicker's presses.
   Opal presses keep their own Sensor ID, so the two sources stay separable.
2. The same event goes to `events.csv` with its full context: host time, Opal
   time, experiment time, trial/rep/bout/walker and attempt number.
3. Optionally (`--gpio-pulse`), each Start/Stop also **pulses the AP output
   line**. This marks the event on external equipment wired to the sync box.
   It is not recorded by the Opals.
4. If you want a copy **on the devices themselves**, the experimenter keeps
   pressing the `event` Opal as today. Its presses land on its flash and
   arrive live as button events, shown on the Events lane beside the cue
   events, so any disagreement is visible.

**Mode switching caveat.** Configuring for streaming overwrites the Opals'
synchronized-logging configuration. Before a normal logging session,
re-configure in Motion Studio.

## 4. Live mechanization

### Why `compute_position` can run live

Offline mechanization is already causal except in one place:

| Step | Depends on | Live? |
|---|---|---|
| Quaternion integration (`qua_est`) | past gyro | immediate |
| `An = R(q)·A` | current q | immediate |
| Tilt KF (`kalman_filter_tilt`) | current accel + stance flag | immediate |
| Stance flag (`stationary_periods`) | current |gyro| and |accel| | immediate |
| **Footfall (`foot_fall`)** | the whole stance plateau: merged when gaps < `T_FF`=0.4 s, footfall = arg-min |gyro| over the plateau | **known 0.4 s after the plateau ends** |
| ZUPT (`zero_velocity_updates`) | An from the previous footfall to this one | when the footfall is known |
| V, P (cumsum) | Anz | per finished stride |

`LiveFootfall` reproduces `foot_fall` exactly, including its 4 s `MAX_T_FF`
chunking of long standing and the end-of-data rule. It just emits each
footfall once it is certain. `LiveFoot` then applies the same ZUPT to the
buffered stride and continues the cumsums with a carried sum, so the result
is **bit-identical**, not merely close: `tests/test_live_mechanize.py`
asserts exact array equality against `compute_position` on real pilot data
and on synthetic gait, including the emitted per-stride blocks.

### Timing and cost (measured)

- Per sample: ~35 µs (numpy per-sample path, reusing the library's own
  `qua_est` / `qua2rot` / `kalman_filter_tilt` for exactness).
- Four feet at 128 Hz: **1.5–2.6% of one core**, measured on replayed s07_s08
  data, viewer included. A **C kernel is not needed**. If you ever go to
  800 Hz × 6 sensors (~13% of a core), a numba or C version of the
  per-sample loop would give ~50× headroom; I'd only do that then.
- Latency to a final stride: the rest of the stance plus 0.4 s. While walking
  that is ~0.6–0.8 s after the footfall, i.e. during the next swing.
  `T_FF` sets it; lowering `T_FF` gives faster strides but no longer matches
  the offline footfalls.
- Between footfalls, `provisional_velocity()` dead-reckons with the previous
  stride's ZUPT bias, or with gravity averaged over stance before the first
  stride. It is display only (faint lines in the viewer) and is replaced by
  the final values.

### Replay check on real data

s07_s08, a hand-off between walkers a and b: walker a's strides came out at
1.51–1.56 m and 1.4–1.6 m/s. Walker b's strides after the hand-off (1.44–1.49 m,
1.4–1.5 m/s) appear within about one stride. The replayed `Start`/`Stop` button
events line up with the gait. These are live per-stride numbers, not the bout
analysis.

### What stays offline

`snug_start` / `snug_end`, `stride_segmentation` and `steps_from_strides` are
**bout-level**: they need the whole bout. The natural live extension is to run
them **at the Stop event** on the buffered bout (sub-second compute), which
would give the full step table and overhead map a moment after each walk.
This is not built yet.

## 5. What was built

Everything is in `src/stride_imu/live/`; the tests stay in the repo's
`tests/` (`test_live_*.py`).

```
README.md                      this report
bridge/ApdmBridge.java         Java bridge: probe | configure <hz> <sd> | stream [latency_ms]
bridge/build/                  compiled bridge (gitignored)
scripts/build_apdm_bridge.sh   javac --release 8 against apdm.jar, then `probe`
scripts/live_feet.py           simple viewer/recorder: --replay H5 | --apdm [--configure]
scripts/live_experiment.py     experiment console: cues, two people, camera
mechanize.py   LiveFootfall, LiveFoot, StrideBlock, run_live (causal compute_position)
sources.py     ReplaySource, BridgeSource, bridge_command, to_body
recorder.py    HdfRecorder (APDM v5 layout), EventLog (events.csv)
pipeline.py    LivePipeline: source -> recorder + LiveFoot per Opal -> rolling buffers,
               stride stats, link health, SensorClock
clock.py       SensorClock: host time -> Opal clock
schedule.py    Cue, Schedule, BrockTiming, build_brock_schedule, Conductor
audio.py       `say` clip synthesis, CuePlayer
video.py       CameraSource, SyntheticCamera, VideoRecorder (segmented, crash-safe)
viewer.py      matplotlib viewer used by live_feet.py
app.py         PySide6 + pyqtgraph console used by live_experiment.py
```

Bridge protocol (stdout, one record per line):
`M {json devices, rate}`, `S dev t_us ax ay az gx gy gz mx my mz button`,
`B dev sync code text`, `X ap sync pin value`, `I/E text`.
Stdin commands: `gpio <ap> <value>`, `quit`. The SDK's own logging goes to
stderr, is set to WARNING, and the last 200 lines are kept for error
messages.

### How to run

```bash
# replay (works now)
uv run python src/stride_imu/live/scripts/live_feet.py --replay "$HOME/Dropbox/Treadmill Brock 2025/imu data/imuData_s07_s08_20260708.h5" \
    --start-s 1010 --duration-s 60 [--speed 2] [--record test.h5]

# hardware (close Motion Studio first)
src/stride_imu/live/scripts/build_apdm_bridge.sh                              # compile + probe
uv run python src/stride_imu/live/scripts/live_feet.py --apdm --configure       # Opals DOCKED, AP plugged in
#   undock, wait for synchronized green blinking
uv run python src/stride_imu/live/scripts/live_feet.py --apdm --record session_live.h5 [--latency-ms 100]
```

## 6. Verified vs. not verified

| Claim | Status |
|---|---|
| `LiveFoot` == `compute_position` (all arrays, bit-exact) | **Verified** (tests, pilot + synthetic) |
| Real-time replay of four feet with viewer, ~2% CPU | **Verified** (headless and a native macOS window) |
| Host .h5 opens with existing loaders; samples identical to source | **Verified** |
| Bridge protocol parsing, events, gpio command echo | **Verified with a fake bridge** |
| Bridge compiles against `apdm.jar`; bundled x86_64 JRE loads `libapdm` under Rosetta and scans USB | **Verified** (fails cleanly at "no access point") |
| Configure / stream / button / sync / gpio against a real AP | **Not tested: needs hardware** |
| `v2_sync_val64_us` = epoch µs (same clock as .h5 `Time`) | From a public example; **check on day 1** |
| Units of the button event's `event_sync_time` | **Unknown** (may be AP sync ticks, 1/2560 s): check against `S` timestamps |
| `setAPOutputGPIOValue(first arg)` is AP id vs AP index | **Unknown**: the bridge passes it through (`gpio <n> <v>`); try 0 and the AP id |

## 7. Live experiment console (`src/stride_imu/live/scripts/live_experiment.py`)

`uv sync --group live` (PySide6 + pyqtgraph), then for a rehearsal on
recorded data:

```bash
uv run python src/stride_imu/live/scripts/live_experiment.py \
  --replay "$HOME/Dropbox/Treadmill Brock 2025/imu data/imuData_s07_s08_20260708.h5" --start-s 1003 \
  --conditions "$HOME/Dropbox/Treadmill Brock 2025/imu data/trialtable_20260708.csv" --trials 1-6
```

For a live session, replace `--replay ... --start-s ...` with `--apdm` and
add `--out <session folder>`.

- **Opal → person assignment:** a dialog lists every Opal with a live
  activity bar ("shake it to find it") and a role menu (A/B left/right,
  event, box, ignore). It is pre-filled from the Opal labels, rejects
  duplicate roles, can be reopened at any time, and is saved to `roles.json`.
  Roles only change the display; every Opal is always recorded.
- **Per person:** live |accel| of both feet (10 s window, with cue and button
  events as dashed lines) and an **overhead footfall map**.
  - **Zero A / Zero B** re-origin the map. Display only; nothing saved
    changes. "Auto-zero walker on Go" re-zeros whoever is cued.
  - Each foot is rotated so its farthest finalized footfall points +Y, since
    every foot has its own heading frame. Feet are drawn ±0.1 m apart.
  - The readout shows distance since zero next to the trial's expected
    distance.
- **Cue track along the top (DAW style):** lanes Announce / Person A /
  Person B / Events on experiment time.
  - Colour-coded blocks sized to the real audio clip: announce, ready,
    go → Start, stop → Stop.
  - Played cues dim, the playhead follows, and upcoming trials stay visible.
  - Double-click the track to seek.
- **Transport:** Play/Pause (Space), ⏮ previous trial (←), ↺ restart trial
  (R), next trial ⏭ (→), mute. Pausing stops the experiment clock, so no cue
  fires. A replayed trial gets `attempt` 2, 3, … in the log. Skipped cues are
  never emitted.
- **Cue schedule:** built from the trial table (`build_brock_schedule`):
  - Walker order follows the protocol: rep 1 = first walker, rep 2 flipped.
  - **Timing is a first guess** (`BrockTiming`: Stop is timed as distance at
    1.0 m/s plus 3 s for the hand-off, 8 s between trials). The exact track
    is saved as `schedule.csv`; edit it and pass it back with `--schedule`.
  - Clips are rendered once by macOS `say` into `~/.cache/stride_imu_cues/`.
    Drop in a recorded human voice under the same file name to replace one.
- **Also included:**
  - A note box that logs a timestamped, trial-tagged note ("B started before
    command"), plus manual Mark / Start / Stop buttons.
  - A running event log.
  - An **Opal link-health table**: received Hz, ms since the last packet,
    missing samples. Red when stale or dropping.
  - A recording indicator.
- **Outputs** (in `--out`): `session.h5`, `events.csv`, `schedule.csv`,
  `roles.json`.

Verified offscreen on replayed s07_s08 data (screenshots) and by
`tests/test_live_app.py`. Checked: cue order, Start/Stop logging, the Opal
clock stamps, pause, previous trial with the attempt counter, notes, and
the assignment dialog. Audio itself was only exercised muted/offscreen:
**check that clips play, and their start latency, on the real machine.**

### Webcam (`--camera INDEX`, `stride_imu.live.video`)

Built for 2-hour sessions:
- **Capture thread → bounded queue → writer thread → ffmpeg.** The capture
  thread only grabs and timestamps frames. If the writer ever falls behind,
  frames are dropped and logged; capture timing is never stalled.
- **Hardware H.264** (Apple VideoToolbox; libx264 as fallback). Measured on
  720p30: the encoder used ~4% of a core, and Python ~10% including
  generating the synthetic frames. The default 3 Mbit/s is **~1.35 GB/hour**,
  so about 2.7 GB for a 2-hour session. Lower `--cam-size`, `--cam-fps` or
  `--cam-bitrate` to shrink it.
- **30-minute segments** (`video_000.mp4`, …), rotated by frame count in
  Python, so `video_frames.csv` maps every frame to its exact file and frame
  number. Verified: ffprobe frame counts equal the CSV rows per segment.
- **Crash-safe:** fragmented MP4. Hard-killing the encoder mid-segment left
  164 of ~195 frames readable in that file, and recording continued in a new
  segment.
- **Timing:** `video_frames.csv` has `host_s` (the same clock as
  `events.csv`) and `sensor_us` (the Opal clock via `SensorClock`; blank
  until the IMU stream is up). Each frame also carries a burned-in frame
  number and time. The camera's own delay from exposure to delivery (often
  30–100 ms) is not included: **measure it once with a stomp test** (a foot
  impact on camera vs. the accelerometer spike).
- **In the console:** a live preview and a status line (segment, fps, drops,
  queue depth, file size) that turns red on drops, errors, or a low frame rate.
- **Camera indices:** `--list-cameras` prints them. Continuity-camera
  iPhones show up too, so check the index. macOS asks once for camera
  permission for the app running Python (Terminal / VS Code).
- **Not yet tested with a real webcam** (only the synthetic test pattern).
  `--synthetic-camera` rehearses the whole path without one.

## 8. Day-1 hardware test plan (~1 hour)

1. Close Motion Studio, plug in the AP, and run `src/stride_imu/live/scripts/build_apdm_bridge.sh`.
   Expect `access points configured: …`.
2. Dock the four foot Opals plus the `event` Opal, then run
   `live_feet.py --apdm --configure`. Undock and wait for synchronized green.
3. Run `live_feet.py --apdm --record t1.h5`. Tap each foot and confirm the
   right trace moves (the label mapping comes from the Opals' own labels).
4. **Timestamps:** press the `event` Opal button while tapping a foot. Compare
   the `B` time with the tap in the `S` stream to settle the button time units.
5. **Latency:** try `--latency-ms` 30, 100 and 250. Watch the dropped/late
   behaviour and the lag between a tap and the plot.
6. **Range:** walk the full Brock path, with the AP where it will live, and
   look for gaps in `Time`.
7. **Device logging:** dock afterwards, import in Motion Studio, and diff the
   Opal-logged data against the host `t1.h5` (the sample times should
   match, and any gaps in the host file should be filled on the device).
8. **Sync box** (if you have one): feed a TTL edge and look for an `X` line.
   Press `o` and scope the OUT BNC.
9. Re-configure the Opals for synchronized logging in Motion Studio before
   the next normal session.

## Sources

- [APDM: Does APDM provide an SDK?](https://support.apdm.com/hc/en-us/articles/360000817606-Does-APDM-provide-an-SDK)
- [APDM SDK Developer Guide (mirror)](https://dokumen.tips/documents/apdm-sdk-developer-guide.html): autoconfigure / record-list / extract-by-device workflow
- [Opal System Technical Guide (PDF)](https://share.apdm.com/documentation/TechnicalGuide.pdf): HDF5 v5 layout, `Wireless Protocol` codes, Sync Box I/O
- [Motion Studio User Guide (PDF)](https://share.apdm.com/documentation/MotionStudioUserGuide.pdf): 6 Opals @128 Hz per AP, latency 30/300 ms, range, robust streaming, synchronized logging, remote
- [APDM Release Notes and Errata](https://support.apdm.com/hc/en-us/articles/214504606-Release-Notes-and-Errata): button events captured while streaming
- [Annotating logged recordings with button events](https://support.apdm.com/hc/en-us/articles/115000260663-Annotating-your-logged-recordings-with-button-events)
- [aero-man/apdm-opal-simple-stream](https://github.com/aero-man/apdm-opal-simple-stream) and [apdm-opal-database](https://github.com/aero-man/apdm-opal-database): working Python 2.7 streaming code, record field names
- [Clario Opal V2R tech specs (PDF)](https://clario.com/wp-content/uploads/2024/05/24_01_PxM_OpalTechSpecsV2R_v2.pdf)
- Local: `apdm.jar` / `libapdm.dylib` inside Motion Studio (APIs read with `javap` / `nm`); session .h5 `Configuration` attributes.
