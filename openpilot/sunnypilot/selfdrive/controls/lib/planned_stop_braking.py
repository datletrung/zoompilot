import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.swaglog import cloudlog

STOP_LINE_BUFFER = 1.0
STOP_VELOCITY_THRESHOLD = 0.5
STOP_CONFIRMATION_SAMPLES = 3


class PlannedStopBraking:
  def __init__(self, actuator_delay: float):
    self.actuator_delay = max(0.0, actuator_delay)
    self._invalid_trajectory_logged = False

  def update(self, model, v_ego: float, model_accel: float, planner_accel: float,
             enabled: bool, longitudinal_enabled: bool, has_lead: bool) -> tuple[float, bool]:
    if not enabled or not longitudinal_enabled or has_lead:
      return planner_accel, False
    if not all(math.isfinite(value) for value in (v_ego, model_accel, planner_accel)) or v_ego <= STOP_VELOCITY_THRESHOLD:
      return planner_accel, False

    stop_distance = self._infer_stop_distance(model)
    if stop_distance is None:
      return planner_accel, False

    available_distance = stop_distance - STOP_LINE_BUFFER - v_ego * self.actuator_delay
    if available_distance <= 0.0:
      required_accel = ACCEL_MIN
    else:
      required_accel = -(v_ego ** 2) / (2.0 * available_distance)

    requested_accel = min(planner_accel, model_accel, required_accel)
    return float(np.clip(requested_accel, ACCEL_MIN, ACCEL_MAX)), True

  def _infer_stop_distance(self, model) -> float | None:
    position = np.asarray(model.position.x, dtype=float)
    velocity = np.asarray(model.velocity.x, dtype=float)
    if position.ndim != 1 or velocity.ndim != 1 or len(position) != len(velocity) or len(position) < STOP_CONFIRMATION_SAMPLES + 1:
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
      cloudlog.warning("PlannedStopBraking: invalid model position/velocity trajectory; skipping stop adjustment")
      self._invalid_trajectory_logged = True
