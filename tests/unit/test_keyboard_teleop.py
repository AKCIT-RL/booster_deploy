"""KeyboardTeleop: keys, safe envelope, ramp, phases, normalization.

    python3 tests/unit/test_keyboard_teleop.py      # or: pytest tests/unit
"""

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from booster_deploy.utils.keyboard_teleop import (  # noqa: E402
    EVENT_CUSTOM, EVENT_DAMPING, EVENT_RL_GAIT, KeyboardTeleop,
    PHASE_DAMPING, PHASE_RUNNING, PHASE_WAIT_CUSTOM, PHASE_WAIT_GAIT,
    TeleopLimits, limits_from_args)

DT = 1.0 / 30.0


def settle(t, seconds=10.0):
    for _ in range(int(seconds / DT)):
        out = t.step(DT, now=0.0)
    return out


def test_forward_reaches_speed_and_holds():
    t = KeyboardTeleop(TeleopLimits(speed=0.3))
    t.handle_key("i", now=0.0)
    vx, vy, vyaw = settle(t)
    assert abs(vx - 0.3) < 1e-9 and vy == 0.0 and vyaw == 0.0


def test_teleop_twist_bindings():
    t = KeyboardTeleop(TeleopLimits(speed=0.2, turn=0.4, safe_vy=0.3))
    cases = {"j": (0, 0, 0.4), "l": (0, 0, -0.4), "u": (0.2, 0, 0.4),
             ",": (-0.2, 0, 0), "J": (0, 0.2, 0), "L": (0, -0.2, 0),
             "O": (0.2, -0.2, 0), "m": (-0.2, 0, -0.4)}
    for key, expect in cases.items():
        t.handle_key(key, now=0.0)
        got = t.target(now=0.0)
        assert all(abs(a - b) < 1e-9 for a, b in zip(got, expect)), (key, got)


def test_stop_keys_and_unknown_keys_stop():
    for key in ("k", " ", "p", "K"):
        t = KeyboardTeleop()
        t.handle_key("i", now=0.0)
        t.handle_key(key, now=0.0)
        assert t.target(now=0.0) == (0.0, 0.0, 0.0), key


def test_ramp_bounded_per_step():
    lim = TeleopLimits(speed=0.8, accel=0.5, yaw_accel=1.0, turn=0.6)
    t = KeyboardTeleop(lim)
    t.handle_key("u", now=0.0)
    prev = (0.0, 0.0, 0.0)
    for _ in range(100):
        cur = t.step(DT, now=0.0)
        assert cur[0] - prev[0] <= lim.accel * DT + 1e-12
        assert cur[2] - prev[2] <= lim.yaw_accel * DT + 1e-12
        prev = cur
    # stopping is ramped too
    t.handle_key("k", now=0.0)
    cur = t.step(DT, now=0.0)
    assert prev[0] - cur[0] <= lim.accel * DT + 1e-12 and cur[0] > 0.0


def test_speed_scale_clamped_to_envelope():
    t = KeyboardTeleop(TeleopLimits(safe_vx=0.5, safe_vyaw=0.4))
    for _ in range(50):
        t.handle_key("q", now=0.0)
    assert t.speed <= 0.5 and t.turn <= 0.4
    t.handle_key("u", now=0.0)
    vx, _, vyaw = t.target(now=0.0)
    assert vx <= 0.5 and vyaw <= 0.4
    t.handle_key("x", now=0.0)      # linear only
    assert t.speed < 0.5 and abs(t.turn - 0.4) < 1e-9


def test_holonomic_vy_clamped_separately():
    t = KeyboardTeleop(TeleopLimits(speed=0.8, safe_vy=0.3))
    t.handle_key("U", now=0.0)
    vx, vy, _ = t.target(now=0.0)
    assert abs(vx - 0.8) < 1e-9 and abs(vy - 0.3) < 1e-9


def test_enter_is_damping_and_immediate():
    t = KeyboardTeleop(TeleopLimits(speed=0.8))
    t.handle_key("i", now=0.0)
    settle(t)
    assert t.handle_key("\n", now=0.0) == EVENT_DAMPING
    assert t.phase == PHASE_DAMPING
    assert t.step(DT, now=0.0) == (0.0, 0.0, 0.0)
    t.handle_key("i", now=0.0)      # ignored while damped
    assert t.step(DT, now=0.0) == (0.0, 0.0, 0.0)
    assert t.handle_key("\r", now=0.0) == EVENT_DAMPING
    t.resume()
    assert t.phase == PHASE_RUNNING and t.target(now=0.0) == (0.0, 0.0, 0.0)


def test_phase_keys_only_before_policy():
    t = KeyboardTeleop(phase=PHASE_WAIT_CUSTOM)
    assert t.handle_key("r", now=0.0) is None and t.phase == PHASE_WAIT_CUSTOM
    t.handle_key("i", now=0.0)      # not latched while waiting
    assert t.handle_key("x", now=0.0) == EVENT_CUSTOM and t.phase == PHASE_WAIT_GAIT
    assert t.handle_key("r", now=0.0) == EVENT_RL_GAIT and t.phase == PHASE_RUNNING
    assert t.target(now=0.0) == (0.0, 0.0, 0.0)
    speed = t.speed
    assert t.handle_key("x", now=0.0) is None and t.speed < speed


def test_enter_damps_in_any_phase():
    for phase in (PHASE_WAIT_CUSTOM, PHASE_WAIT_GAIT, PHASE_RUNNING):
        t = KeyboardTeleop(phase=phase)
        assert t.handle_key("\n", now=0.0) == EVENT_DAMPING


def test_timeout_zeroes_target():
    t = KeyboardTeleop(TeleopLimits(timeout=0.5))
    t.handle_key("i", now=10.0)
    assert t.target(now=10.4)[0] > 0.0
    assert t.target(now=10.6) == (0.0, 0.0, 0.0)


def test_normalization_round_trips_through_update_vel_command():
    # KeyboardTeleopService returns phys/vx_max; BoosterRobotController
    # .update_vel_command does cmd*vx_max + init, clipped to vx_max
    vx_max, init = 2.5, 0.0
    for phys in (0.0, 0.3, -0.8, 0.8):
        norm = phys / vx_max
        back = min(max(norm * vx_max + init, -vx_max), vx_max)
        assert abs(back - phys) < 1e-12


def test_limits_refuse_envelope_wider_than_task():
    vel = types.SimpleNamespace(vx_max=1.0, vy_max=1.0, vyaw_max=1.0)
    args = types.SimpleNamespace(safe_vx=1.5, safe_vy=0.3, safe_vyaw=0.6,
                                 teleop_speed=0.3, teleop_turn=0.3,
                                 teleop_accel=0.5, teleop_yaw_accel=1.0,
                                 teleop_timeout=None)
    try:
        limits_from_args(args, vel)
    except ValueError:
        pass
    else:
        raise AssertionError("safe_vx above vx_max must be refused")
    args.safe_vx = 0.8
    assert limits_from_args(args, vel).safe_vx == 0.8


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print("{} passed".format(len(tests)))
