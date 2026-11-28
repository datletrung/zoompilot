import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

MODEL_T_IDXS = np.asarray(ModelConstants.T_IDXS, dtype=float)

LAUNCH_SPEED_MAX = 4.0
LAUNCH_FADE_SPEED = 1.0
LATCH_SPEED_MAX = 0.5
LATCH_MAX_TIME = 4.0

SLOWDOWN_START = 1.0
SLOWDOWN_END = 5.0
SLOWDOWN_TOLERANCE = 0.5
LAT_ACCEL_LIMIT = 1.5
LAT_HORIZON = 4.0

PROFILE_GENTLE = 0
PROFILE_NORMAL = 1
PROFILE_SPORTY = 2

# profile: (maximum launch accel m/s^2, onset jerk limit m/s^3, standstill kick m/s^2)
LAUNCH_PROFILES = {
  PROFILE_GENTLE: (1.2, 2.0, 0.3),
  PROFILE_NORMAL: (1.6, 3.0, 0.5),
  PROFILE_SPORTY: (2.0, 5.0, 0.8),
}


class PlannedLaunchBoost:
  def __init__(self):
    self._was_stopped = False
    self._latched = False
    self._latch_time = 0.0
    self._prev_accel = 0.0

  def _unlatch(self) -> None:
    self._latched = False
    self._latch_time = 0.0
    self._prev_accel = 0.0

  def update(self, model, v_ego: float, planner_accel: float,
             enabled: bool, longitudinal_enabled: bool, has_lead: bool, should_stop: bool,
             profile: int = PROFILE_NORMAL) -> tuple[float, bool]:
    released = self._was_stopped and not should_stop
    self._was_stopped = bool(should_stop)

    finite = all(math.isfinite(value) for value in (v_ego, planner_accel))
    if (not enabled or not longitudinal_enabled or has_lead or should_stop or not finite
            or v_ego < 0.0 or v_ego >= LAUNCH_SPEED_MAX):
      self._unlatch()
      return planner_accel, False

    if released and v_ego < LATCH_SPEED_MAX:
      self._latched = True
      self._latch_time = 0.0
      self._prev_accel = 0.0
    if not self._latched:
      return planner_accel, False

    self._latch_time += DT_MDL
    if self._latch_time > LATCH_MAX_TIME or not self._model_allows_launch(model, v_ego):
      self._unlatch()
      return planner_accel, False

    max_accel, onset_jerk, kick = LAUNCH_PROFILES.get(profile, LAUNCH_PROFILES[PROFILE_NORMAL])
    max_accel = min(max_accel, ACCEL_MAX)

    ramp_accel = min(max(self._prev_accel, kick) + onset_jerk * DT_MDL, max_accel)
    self._prev_accel = ramp_accel
    fade = float(np.clip((LAUNCH_SPEED_MAX - v_ego) / LAUNCH_FADE_SPEED, 0.0, 1.0))
    target = planner_accel + fade * max(ramp_accel - planner_accel, 0.0)

    requested_accel = max(target, planner_accel)
    boosted = requested_accel > planner_accel + 1e-3
    return float(np.clip(requested_accel, ACCEL_MIN, ACCEL_MAX)), boosted

  def _model_allows_launch(self, model, v_ego: float) -> bool:
    velocity = np.asarray(model.velocity.x, dtype=float)
    yaw_rate = np.asarray(model.orientationRate.z, dtype=float)
    n = len(MODEL_T_IDXS)
    shapes_ok = all(a.ndim == 1 and len(a) == n for a in (velocity, yaw_rate))
    if not shapes_ok or not all(np.all(np.isfinite(a)) for a in (velocity, yaw_rate)):
      return False

    slowdown_window = (MODEL_T_IDXS >= SLOWDOWN_START) & (MODEL_T_IDXS <= SLOWDOWN_END)
    planned_speed = velocity[slowdown_window]
    speed_drop = np.maximum.accumulate(planned_speed) - planned_speed
    if float(np.max(speed_drop)) > SLOWDOWN_TOLERANCE:
      return False

    lateral_window = MODEL_T_IDXS <= LAT_HORIZON
    lateral_accel = np.abs(yaw_rate[lateral_window]) * np.maximum(velocity[lateral_window], v_ego)
    return float(np.max(lateral_accel)) <= LAT_ACCEL_LIMIT
