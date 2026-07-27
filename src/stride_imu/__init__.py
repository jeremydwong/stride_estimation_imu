# __init__.py

from .apdm import (
    ImuRecording, load_imu_recording, find_overlapping_recordings, load_overlapping_recordings,
    sync_apdm, getdata, list_sensors, detect_quiet_time
)

from .inertial import (
    WalkingBout, FootTrajectory, Strides, detect_quiet_periods, detect_walking_section, compute_position, stride_segmentation,
    steps_from_strides, touchdown_map, snug_start,
    compute_position_two_imus, detect_walking_bouts, find_bouts_near_time
)

from .plotting import (
    plt_frwd_elev_strides, plt_ltrl_frwd_strides, plt_stride_var, plt_walk_info_position,
    side_color, ACCEL_YMAX, FOOTSPEED_YMAX, STEPSPEED_YMAX,
    draw_accel, draw_velocity, draw_step_speed,
    draw_overhead, make_bout_axes, draw_bout_block,
    draw_event_timeline, assign_bout_times, pair_status,
    EVENT_COLORS, TIMELINE_COLORS
)