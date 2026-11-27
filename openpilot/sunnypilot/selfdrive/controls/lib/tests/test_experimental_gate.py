"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The planner's e2e candidate runs whenever openpilot drives the gas and brakes (chill mode went
2026-11). Also the fixture the planner hook tests build on.
"""
import pytest

from openpilot.cereal import custom, messaging
from opendbc.car import structs
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, LongitudinalPlanSource
from openpilot.sunnypilot.selfdrive.controls.lib.planned_stop_braking import PlannedStopBraking

V_EGO = 20.0
E2E_ACCEL = -3.0  # low enough that e2e wins the min() whenever it is a candidate


class MockSubMaster(dict):
  def __init__(self, services: dict):
    super().__init__(services)
    self.valid = dict.fromkeys(services, True)
    self.logMonoTime = dict.fromkeys(services, 0)
    self.updated = dict.fromkeys(services, True)
    self.recv_frame = dict.fromkeys(services, 1)

  def all_checks(self, service_list=None) -> bool:
    return True


def build_sm() -> MockSubMaster:
  services = {}
  for service in ("radarState", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation", "vehicleParameters"):
    services[service] = getattr(messaging.new_message(service), service)

  # engaged on openpilot long: longcontrol running, so the planner does not reset
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.longControlState = LongCtrlState.pid
  services['controlsState'] = controls_state.controlsState.as_reader()

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = V_EGO
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
  services['carState'] = car_state.carState.as_reader()

  selfdrive_state = messaging.new_message('selfdriveState')
  selfdrive_state.selfdriveState.enabled = True
  services['selfdriveState'] = selfdrive_state.selfdriveState.as_reader()

  car_control = messaging.new_message('carControl')
  car_control.carControl.enabled = True
  services['carControl'] = car_control.carControl.as_reader()

  model = messaging.new_message('modelV2')
  model.modelV2.orientationRate.z = [0.01] * 33  # nonzero: a straight path divides by zero in SCC vision
  model.modelV2.velocity.x = [V_EGO] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  model.modelV2.action.desiredAcceleration = E2E_ACCEL
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def build_planner(op_long: bool = True) -> LongitudinalPlanner:
  CP = structs.CarParams()
  CP.longitudinalActuatorDelay = 0.2
  CP.openpilotLongitudinalControl = op_long
  CP_SP = custom.CarParamsSP.new_message().as_reader()
  return LongitudinalPlanner(CP, CP_SP, init_v=V_EGO)


class TestExperimentalGate(OpenpilotTestCase):
  def _source(self, op_long: bool = True) -> LongitudinalPlanSource:
    planner = build_planner(op_long)
    planner.update(build_sm())
    return planner.mpc.source

  def test_e2e_on_openpilot_long(self):
    assert self._source() == LongitudinalPlanSource.e2e

  def test_no_e2e_without_openpilot_long(self):
    assert self._source(op_long=False) != LongitudinalPlanSource.e2e

  def test_without_openpilot_long_the_candidate_never_wins_or_stops(self):
    planner = build_planner(op_long=False)
    sm = build_sm()
    model = sm['modelV2'].as_builder()
    model.action.shouldStop = True
    sm['modelV2'] = model.as_reader()
    assert planner.update_e2e_target(sm, E2E_ACCEL, True, False) == (ACCEL_MAX, False)
    planner.update(sm)
    assert not planner.output_should_stop

  def test_cornering_estimate_is_controlsds_curvature(self):
    # plannerd no longer subscribes to vehicleParameters: the lifts read controlsState's curvature
    planner = build_planner()
    sm = build_sm()
    controls_state = sm['controlsState'].as_builder()
    controls_state.curvature = 0.002
    sm['controlsState'] = controls_state.as_reader()
    seen = []
    update = planner.e2e_set_speed.update
    planner.e2e_set_speed.update = lambda *args: seen.append(args[-1]) or update(*args)
    planner.update(sm)
    assert seen == [pytest.approx(0.002 * V_EGO ** 2)]

  def test_planned_stop_braking_targets_one_meter_before_inferred_stop(self, monkeypatch):
    v_ego = 85.0 / 3.6
    planner = build_planner()
    monkeypatch.setattr(planner.params, "get_bool", lambda key: key == "PlannedStopBraking")
    sm = build_sm()
    actuator_delay = planner.CP.longitudinalActuatorDelay
    stop_line_distance = v_ego * actuator_delay + v_ego ** 2 / (2.0 * abs(ACCEL_MIN)) + 1.0
    distance_travelled = 0.0
    speed = v_ego
    measured_accel = 0.0
    commands = []
    delay_frames = round(actuator_delay / DT_MDL)

    for frame in range(500):
      car_state = sm["carState"].as_builder()
      car_state.vEgo = speed
      car_state.aEgo = measured_accel
      car_state.vCruise = 110.0
      car_state.vCruiseCluster = 110.0
      sm["carState"] = car_state.as_reader()

      distance_to_line = stop_line_distance - distance_travelled
      predicted_decel_distance = max(0.01, distance_to_line - speed * actuator_delay)
      predicted_decel = -(speed ** 2) / (2.0 * predicted_decel_distance)
      stop_time = speed / abs(predicted_decel)
      model = sm["modelV2"].as_builder()
      predicted_x = []
      predicted_v = []
      for t in ModelConstants.T_IDXS:
        if t <= actuator_delay:
          predicted_x.append(speed * t)
          predicted_v.append(speed)
        else:
          brake_time = min(t - actuator_delay, stop_time)
          predicted_x.append(min(speed * actuator_delay + speed * brake_time +
                                 0.5 * predicted_decel * brake_time ** 2, distance_to_line))
          predicted_v.append(max(0.0, speed + predicted_decel * brake_time))
      model.position.x = predicted_x
      model.velocity.x = predicted_v
      model.action.desiredAcceleration = -0.5
      model.action.shouldStop = False
      sm["modelV2"] = model.as_reader()

      planner.update(sm)
      command = float(planner.output_a_target)
      commands.append(command)
      if frame == 0:
        assert command == pytest.approx(ACCEL_MIN)

      applied_accel = commands[frame - delay_frames] if frame >= delay_frames else 0.0
      if speed > 0.0 and applied_accel < 0.0 and speed + applied_accel * DT_MDL < 0.0:
        stop_time = speed / -applied_accel
        distance_travelled += speed * stop_time + 0.5 * applied_accel * stop_time ** 2
        speed = 0.0
        break

      distance_travelled += speed * DT_MDL + 0.5 * applied_accel * DT_MDL ** 2
      speed = max(0.0, speed + applied_accel * DT_MDL)
      measured_accel = applied_accel

    assert speed == 0.0
    assert stop_line_distance - distance_travelled == pytest.approx(1.0, abs=0.1)
    assert not planner.output_should_stop

  def test_late_stop_request_is_capped_and_existing_braking_is_preserved(self):
    v_ego = 85.0 / 3.6
    model = messaging.new_message("modelV2").modelV2
    model.position.x = [min(5.0, 5.0 * i / 24) for i in range(33)]
    model.velocity.x = [max(0.0, v_ego * (1.0 - i / 24)) for i in range(33)]
    model.action.desiredAcceleration = -0.5
    planned_stop = PlannedStopBraking(actuator_delay=0.2)

    adjusted, inferred_stop = planned_stop.update(
      model, v_ego, model.action.desiredAcceleration, -1.0, True, True, False)

    assert inferred_stop
    assert adjusted == pytest.approx(ACCEL_MIN)

  def test_planned_stop_braking_is_inactive_when_disabled_or_a_lead_is_present(self):
    v_ego = 20.0
    model = messaging.new_message("modelV2").modelV2
    model.position.x = [float(i) for i in range(33)]
    model.velocity.x = [max(0.0, v_ego - i) for i in range(33)]
    planned_stop = PlannedStopBraking(actuator_delay=0.2)

    for enabled, has_lead in ((False, False), (True, True)):
      adjusted, inferred_stop = planned_stop.update(model, v_ego, -0.5, -0.8, enabled, True, has_lead)
      assert adjusted == pytest.approx(-0.8)
      assert not inferred_stop
