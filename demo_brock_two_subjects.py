"""Two-subject treadmill/box-handoff session (Brock 2025) — loading and trial skimming.

Single APDM .h5 file containing the entire ~128-minute session for both
subjects (96 trials: 48 sequences x 2, see ryan-data notes). Six IMUs,
associated via each sensor's 'Label 0' configuration:

    XI-021156  left_foot_a   -> S1 left foot
    XI-021165  right_foot_a  -> S1 right foot
    XI-021213  left_foot_b   -> S2 left foot
    XI-021208  right_foot_b  -> S2 right foot
    XI-021789  event         -> experimenter's hand-held IMU (shaken and/or
                                button-pressed to mark trials; the file's
                                Annotations table holds its Start/Stop presses)
    XI-021852  box           -> the box handed between subjects

This script loads all six recordings, prints the association, then finds
candidate trial markers on the experimenter's IMU by thresholding the
accelerometer norm, and plots them against the button-press annotations
for skimming.
"""
import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu

H5_FILE = os.environ.get(
    'STRIDE_DATA_FILE',
    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s01_s02_20260507.h5')

# Role -> 'Label 0' string in the file
ROLES = {
    's1_left_foot':  'left_foot_a',
    's1_right_foot': 'right_foot_a',
    's2_left_foot':  'left_foot_b',
    's2_right_foot': 'right_foot_b',
    'experimenter':  'event',
    'box':           'box',
}

# Shake detection on the experimenter's IMU
SHAKE_THRESHOLD = 20.0   # m/s^2 accel norm (gravity-quiet baseline is ~10)
SHAKE_MIN_GAP = 5.0      # seconds; threshold crossings closer than this are one event


def load_recordings(file_path):
    """Load all six IMUs from the session file, keyed by role."""
    label_to_id = {label: sid for sid, label in imu.list_sensors(file_path).items()}
    recordings = {}
    for role, label in ROLES.items():
        recordings[role] = imu.load_imu_recording(file_path, sensor_id=label_to_id[label])
    return recordings


def load_annotations(file_path, reference_recording):
    """Read the button-press Annotations table.

    Returns (elapsed_seconds, labels) with times on the same clock as
    reference_recording's samples.
    """
    with h5py.File(file_path, 'r') as f:
        ann = f['Annotations'][:]
    elapsed_s = (ann['Time'] - reference_recording.raw_time[0]) / 1e6
    labels = np.array([a.decode() for a in ann['Annotation']])
    return elapsed_s, labels


def detect_shake_events(recording, threshold=SHAKE_THRESHOLD, min_gap=SHAKE_MIN_GAP):
    """Find shake events as bursts where the accel norm exceeds threshold.

    Crossings separated by less than min_gap seconds are grouped into one
    event. Returns the start time (elapsed seconds) of each event.
    """
    norm = np.linalg.norm(recording.Ab, axis=1)
    t = recording.time_elapsed_samples * recording.period
    over = np.where(norm > threshold)[0]
    if len(over) == 0:
        return np.array([]), norm, t
    new_event = np.where(np.diff(t[over]) > min_gap)[0]
    starts = np.r_[over[0], over[new_event + 1]]
    return t[starts], norm, t


if __name__ == '__main__':
    print(f"Loading 6 IMUs from {H5_FILE}")
    recordings = load_recordings(H5_FILE)

    ref = recordings['experimenter']
    print(f"\nSession: {ref.time_datetime[0]} -> {ref.time_datetime[-1]}"
          f"  ({len(ref.time_datetime) * ref.period / 60:.1f} min at {1/ref.period:.0f} Hz)")
    print("\nSensor association:")
    for role, rec in recordings.items():
        sid = os.path.basename(str(rec.file_path))
        print(f"  {role:14s} label={ROLES[role]:13s} samples={len(rec.Wb)}")

    # All sensors are logged through one access point and share sample counts;
    # verify they are actually time-aligned before treating them as synced
    for role, rec in recordings.items():
        offset_ms = abs(float(rec.raw_time[0]) - float(ref.raw_time[0])) / 1e3
        assert offset_ms < 1000, f"{role} starts {offset_ms:.0f} ms away from reference"

    # Candidate trial markers: shakes of the experimenter's IMU
    shake_times, norm, t = detect_shake_events(recordings['experimenter'])
    print(f"\nDetected {len(shake_times)} shake events "
          f"(norm > {SHAKE_THRESHOLD} m/s^2, {SHAKE_MIN_GAP}s grouping)")

    # Button presses recorded by the same IMU, for cross-checking
    ann_t, ann_labels = load_annotations(H5_FILE, ref)
    print(f"Annotations: {np.sum(ann_labels == 'Start')} Start, "
          f"{np.sum(ann_labels == 'Stop')} Stop button presses")

    # Skim plot: experimenter accel norm with shakes and button presses marked
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax = axes[0]
    ax.plot(t / 60, norm, lw=0.3, color='gray')
    ax.axhline(SHAKE_THRESHOLD, color='orange', ls='--', lw=0.8, label='shake threshold')
    ax.plot(shake_times / 60, np.full_like(shake_times, SHAKE_THRESHOLD + 5),
            'v', color='red', ms=5, label=f'shake events (n={len(shake_times)})')
    for lbl, color in [('Start', 'green'), ('Stop', 'purple')]:
        sel = ann_t[ann_labels == lbl]
        ax.plot(sel / 60, np.full_like(sel, 2), '|', color=color, ms=12,
                label=f'{lbl} button (n={len(sel)})')
    ax.set_ylabel('experimenter accel norm [m/s$^2$]')
    ax.legend(loc='upper right')
    ax.set_title('Experimenter (event) IMU — trial markers')

    ax = axes[1]
    box_norm = np.linalg.norm(recordings['box'].Ab, axis=1)
    ax.plot(t / 60, box_norm, lw=0.3, color='steelblue')
    ax.set_ylabel('box accel norm [m/s$^2$]')
    ax.set_xlabel('elapsed time [min]')
    ax.set_title('Box IMU')
    plt.tight_layout()
    plt.show()
