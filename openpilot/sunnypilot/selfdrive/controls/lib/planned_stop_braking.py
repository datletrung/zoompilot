import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog

STOP_VELOCITY_THRESHOLD = 0.5
STOP_CONFIRMATION_SAMPLES = 3
# Measured accel response lags the command by ~0.4 s (first-order) on top of pure dead time.
# actuator_delay only covers the dead time, so add the response lag here.
ACTUATOR_LAG = 0.3
# Keep using the last good stop estimate (dead-reckoned) when detection briefly drops out.
STOP_HOLD_TIME = 1.0

PROFILE_GENTLE = 0
PROFILE_NORMAL = 1
PROFILE_SPORTY = 2

# profile: (max decel the patch may request in m/s^2, buffer before the stop point in m)
STOP_PROFILES = {
  PROFILE_GENTLE: (2.0, 3.0),
  PROFILE_NORMAL: (3.0, 2.0),
  PROFILE_SPORTY: (abs(ACCEL_MIN), 1.0),
}


class PlannedStopBraking:
  def __init__(self, actuator_delay: float):
    self.actuator_delay = max(0.0, actuator_delay) + ACTUATOR_LAG
    self._invalid_trajectory_logged = False
    self._held_distance: float | None = None
    self._hold_frames_left = 0

  def _reset(self) -> None:
    self._held_distance = None
    self._hold_frames_left = 0

  def update(self, model, v_ego: float, model_accel: float, planner_accel: float,
             enabled: bool, longitudinal_enabled: bool, has_lead: bool,
             profile: int = PROFILE_NORMAL) -> tuple[float, bool]:
    if not enabled or not longitudinal_enabled or has_lead:
      self._reset()
      return planner_accel, False

    inputs_finite = all(math.isfinite(value) for value in (v_ego, model_accel, planner_accel))
    if not inputs_finite or v_ego <= STOP_VELOCITY_THRESHOLD:
      self._reset()
      return planner_accel, False

    stop_distance = self._infer_stop_distance(model)
    if stop_distance is not None:
      self._held_distance = stop_distance
      self._hold_frames_left = round(STOP_HOLD_TIME / DT_MDL)
    elif self._held_distance is not None and self._hold_frames_left > 0:
      # detection dropped out: reuse the last estimate, moved closer by the distance driven this frame
      self._held_distance -= v_ego * DT_MDL
      self._hold_frames_left -= 1
      stop_distance = self._held_distance
    else:
      self._reset()
      return planner_accel, False

    max_decel, buffer = STOP_PROFILES.get(profile, STOP_PROFILES[PROFILE_NORMAL])
    max_decel = min(max_decel, abs(ACCEL_MIN))

    available_distance = stop_distance - buffer - v_ego * self.actuator_delay
    if available_distance <= 0.0:
      required_accel = -max_decel
    else:
      required_accel = max(-(v_ego ** 2) / (2.0 * available_distance), -max_decel)

    requested_accel = min(planner_accel, model_accel, required_accel)
    return float(np.clip(requested_accel, ACCEL_MIN, ACCEL_MAX)), True

  def _infer_stop_distance(self, model) -> float | None:
    position = np.asarray(model.position.x, dtype=float)
    velocity = np.asarray(model.velocity.x, dtype=float)

    bad_shape = (
      position.ndim != 1
      or velocity.ndim != 1
      or len(position) != len(velocity)
      or len(position) < STOP_CONFIRMATION_SAMPLES + 1
    )
    if bad_shape:
      self._log_invalid_trajectory()
      return None
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
      self._log_invalid_trajectory()
      return None

    for i in range(1, len(velocity) - STOP_CONFIRMATION_SAMPLES + 1):
      sustained_stop = velocity[i:i + STOP_CONFIRMATION_SAMPLES] <= STOP_VELOCITY_THRESHOLD
      if velocity[i - 1] > STOP_VELOCITY_THRESHOLD and np.all(sustained_stop):
        velocity_delta = velocity[i - 1] - velocity[i]
        if velocity_delta <= 0.0:
          continue
        fraction = (velocity[i - 1] - STOP_VELOCITY_THRESHOLD) / velocity_delta
        distance = position[i - 1] + fraction * (position[i] - position[i - 1])
        if math.isfinite(distance) and distance > 0.0:
          return float(distance)
        break

    return None

  def _log_invalid_trajectory(self) -> None:
    if not self._invalid_trajectory_logged:
      cloudlog.warning(
        "PlannedStopBraking: invalid model position/velocity trajectory; skipping stop adjustment"
      )
      self._invalid_trajectory_logged = True
