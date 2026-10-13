"""Drive the MimicKit steering policy from the keyboard, in the MuJoCo viewer.

    cd booster_deploy
    PYTHONPATH=../booster_assets/src DISPLAY=:1 \\
        python3 tasks/mimickit_steering/teleop.py --pd explicit

    W / X    forward speed  +/- 0.1 m/s
    A / D    turn left / right   +/- 0.1 rad/s
    Q / E    strafe left / right +/- 0.1 m/s
    Z        zero the command
    P        print command and robot state

Click the MuJoCo window once before typing. The viewer's own keys all still
work, because this chains into MujocoController's key_callback rather than
replacing it: Space pauses, Left/Right step a frame while paused, Backspace
resets, 0-9 pick a state slot with S to save and L to load, R toggles recording,
G toggles the ghost.

Speed is W/X rather than the obvious W/S because S is the viewer's save-state
key, and reset is left to the viewer's Backspace. Every binding here was checked
against that list.

The terminal also still accepts `x y yaw` + Enter at any time; MujocoController
polls stdin every step. Note that when stdin is not a live terminal (a pipe, or
`< /dev/null`) select() reports EOF as readable and that path prints "Invalid
input" every step - harmless, and absent in interactive use.

--pd selects the damping scheme; see controllers.py for what the three mean and
why the choice is not cosmetic. Default is `explicit`, the scheme the policy was
trained and validated against.

--degrade zero_root replaces root_pos_w and root_lin_vel_b with zeros, which is
exactly what booster_robot_controller reports on the real T1. Watch it fall
forward in about a second - see controllers.make_degraded.

--effort selects the torque ceiling: `urdf` (default) is what the policy was
trained against, `derated` is what the firmware actually enforces. `derated`
isolates that one hardware constraint - see controllers.apply_effort_source.

--teleop keyboard drives the policy from the TERMINAL with the same
teleop_twist_keyboard-style layout `deploy.py --teleop keyboard` uses on the
robot (u i o / j k l / m , . ; Enter = simulated damping), with its safe
envelope and ramp (--safe-vx, --teleop-accel, ...). The default, `viewer`, is
the W/X A/D Q/E keys in the MuJoCo window described above. See
docs/17-teleop-teclado.md.

--checkpoint overrides the policy checkpoint (default: mimickit_steering.py's
MimicKitSteeringPolicyCfg.checkpoint_path). Relative paths resolve against the
task dir, same as the config default.

HOW THE VIEWER KEYS ARE PRESERVED

MujocoController.run() builds its key_callback locally and passes it straight to
mujoco.viewer.launch_passive, so there is no hook to extend. Rather than copy
run() - which silently freezes a copy of upstream's viewer features at whatever
revision it was copied from, and did exactly that during development - this
wraps launch_passive for the duration of the call and chains the two callbacks.
booster_deploy is left untouched and its viewer features keep working as they
evolve.
"""

from __future__ import annotations

import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from tasks.mimickit_steering.controllers import (  # noqa: E402
    CONTROLLERS, DEGRADE_MODES, DEGRADE_NONE, EFFORT_SOURCES, EFFORT_URDF,
    IMU_NOISE_NONE, IMU_NOISE_PRESETS,
    apply_effort_source, apply_tn_curve, make_degraded, make_noisy_imu,
    apply_heading_mode, parse_hip_yaw_limit, parse_imu_noise)

VX_STEP = 0.1
VY_STEP = 0.1
VYAW_STEP = 0.1

# chosen to avoid MujocoController.run's own bindings:
# Space pause, Backspace, Left/Right step, R record, G ghost
# MujocoController.run binds Space(32) Backspace(259) Left(263) Right(262)
# 0-9(48-57) S(83) L(76) R(82) G(71); none of the below overlaps that set
_KEY_W, _KEY_X = 87, 88          # speed up / down (S is the viewer's save)
_KEY_A, _KEY_D = 65, 68          # turn
_KEY_Q, _KEY_E = 81, 69          # strafe
_KEY_Z, _KEY_P = 90, 80          # zero command, print

HELP = ("\n[teleop]  W/X speed   A/D turn   Q/E strafe   Z zero   P print"
        "\n[viewer]  Space pause   Left/Right step   Backspace reset"
        "   0-9 slot   S save   L load   R record   G ghost"
        "\n[stdin]   or type: x y yaw <Enter>\n")


def make_keyboard_controller(base_cls):
    """Add velocity keys to any MujocoController subclass.

    A factory rather than a fixed class so the damping scheme stays the only
    axis of variation: every controller in controllers.py can be driven.
    """

    class KeyboardController(base_cls):

        def _clamp_command(self):
            cmd = self.vel_command
            cmd.lin_vel_x = float(np.clip(cmd.lin_vel_x, -cmd.vx_max, cmd.vx_max))
            cmd.lin_vel_y = float(np.clip(cmd.lin_vel_y, -cmd.vy_max, cmd.vy_max))
            cmd.ang_vel_yaw = float(np.clip(cmd.ang_vel_yaw, -cmd.vyaw_max, cmd.vyaw_max))

        def _print_command(self):
            cmd = self.vel_command
            speed = math.hypot(cmd.lin_vel_x, cmd.lin_vel_y)
            lo = getattr(self.cfg.policy, "tar_speed_min", 0.0)
            hi = getattr(self.cfg.policy, "tar_speed_max", float("inf"))
            clamped = min(max(speed, lo), hi)
            heading = getattr(self.policy, "_face_heading", 0.0)
            note = "" if abs(clamped - speed) < 1e-6 else \
                "  <- clamped to the trained range [{:.1f}, {:.1f}]".format(lo, hi)
            print("\rcmd vx={:+.2f} vy={:+.2f} yaw={:+.2f} | tar_speed={:.2f} "
                  "facing={:+.0f} deg{}\nSet command (x, y, yaw): ".format(
                      cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw,
                      clamped, math.degrees(heading), note), end="", flush=True)

        def _velocity_key(self, keycode):
            """True when the key was ours, so the viewer's callback is skipped."""
            cmd = self.vel_command
            if (keycode == _KEY_W):
                cmd.lin_vel_x += VX_STEP
            elif (keycode == _KEY_X):
                cmd.lin_vel_x -= VX_STEP
            elif (keycode == _KEY_A):
                cmd.ang_vel_yaw += VYAW_STEP
            elif (keycode == _KEY_D):
                cmd.ang_vel_yaw -= VYAW_STEP
            elif (keycode == _KEY_Q):
                cmd.lin_vel_y += VY_STEP
            elif (keycode == _KEY_E):
                cmd.lin_vel_y -= VY_STEP
            elif (keycode == _KEY_Z):
                cmd.lin_vel_x = cmd.lin_vel_y = cmd.ang_vel_yaw = 0.0
            elif (keycode == _KEY_P):
                pos, vel = self.mj_data.qpos[0:3], self.mj_data.qvel[0:3]
                print("\n[state] pos=({:+.2f}, {:+.2f}, {:.3f})  |v|={:.2f} m/s".format(
                    pos[0], pos[1], pos[2], math.hypot(vel[0], vel[1])), flush=True)
                return True
            else:
                return False
            self._clamp_command()
            self._print_command()
            return True

        def run(self):
            real_launch = mujoco.viewer.launch_passive

            def launch_with_velocity_keys(model, data, *args, **kwargs):
                viewer_callback = kwargs.pop("key_callback", None)

                def chained(keycode):
                    if (not self._velocity_key(keycode) and viewer_callback is not None):
                        viewer_callback(keycode)

                return real_launch(model, data, *args, key_callback=chained, **kwargs)

            print(HELP, flush=True)
            # the upstream MujocoController (deploy.py --mujoco without --pd)
            # has no describe_pd
            if hasattr(self, "describe_pd"):
                print(self.describe_pd(), flush=True)
            mujoco.viewer.launch_passive = launch_with_velocity_keys
            try:
                return super().run()
            finally:
                mujoco.viewer.launch_passive = real_launch

    KeyboardController.__name__ = "Keyboard" + base_cls.__name__
    return KeyboardController


def make_terminal_teleop_controller(base_cls, limits=None):
    """Drive any MujocoController subclass with the robot's TERMINAL keys.

    The same KeyboardTeleop (booster_deploy/utils/keyboard_teleop.py) that
    `deploy.py --teleop keyboard` uses on the T1, so the layout, the safe
    envelope and the ramp rehearsed here are the ones used on the robot. Type
    in the TERMINAL, not in the viewer window; the viewer's own keys (Space,
    Backspace, S/L, R, G) keep working in the window.

    Enter, and the fall guard, put the sim in SIMULATED DAMPING: the policy is
    no longer called and every joint gets tau = -kd*qd with kd from
    prepare_state.damping and kp = 0 - an approximation of the firmware's
    damping mode, there so the operator sees what Enter does before pressing
    it on the robot. Backspace (the viewer's reset) leaves damping and restarts
    the policy, which the robot cannot do.

    There are no phases here: the policy runs from the first step, so 'x' and
    'r' are plain teleop keys ('x' = linear -10%).
    """
    from booster_deploy.utils.keyboard_teleop import (
        HELP as TELEOP_HELP, KeyboardTeleop, TerminalKeyReader,
        EVENT_DAMPING, PHASE_DAMPING)

    class TerminalTeleopController(base_cls):

        def __init__(self, cfg):
            super().__init__(cfg)
            if (self.vel_command is None):
                raise ValueError("--teleop keyboard: a task nao tem vel_command")
            self.teleop = KeyboardTeleop(limits)
            self._damping_kd = np.asarray(
                self.cfg.robot.prepare_state.damping, dtype=np.float64)
            self._status_t = 0.0
            # Leave damping on every policy reset: start() and the viewer's
            # Backspace both go through policy.reset(), and wrapping it here
            # (rather than the key) means the flag can never be cleared one
            # step before the physics is actually reset.
            policy_reset = self.policy.reset

            def reset_and_resume():
                if (self.teleop.phase == PHASE_DAMPING):
                    print("\n[teleop] reset: saindo do damping, politica reiniciada",
                          flush=True)
                    self.log_event("reset")
                self.teleop.resume()
                policy_reset()

            self.policy.reset = reset_and_resume

        @property
        def damping(self):
            return self.teleop.phase == PHASE_DAMPING

        def _on_key(self, ch):
            if (self.teleop.handle_key(ch) == EVENT_DAMPING):
                self.log_event("damping")
                print("\n[teleop] ENTER -> DAMPING simulado (Backspace na janela "
                      "do viewer reinicia)", flush=True)
            else:
                print("\r\033[K" + self.teleop.status_line(), end="", flush=True)

        def update_vel_command(self):
            # replaces MujocoController's stdin line reader: both would fight
            # over the same stdin, and this one owns it in cbreak mode
            cmd = self.vel_command
            vx, vy, vyaw = self.teleop.step(self.cfg.policy_dt)
            # the --vx/--vy/--vyaw baseline, as on the robot
            cmd.lin_vel_x = float(np.clip(vx + cmd.lin_vel_x_init, -cmd.vx_max, cmd.vx_max))
            cmd.lin_vel_y = float(np.clip(vy + cmd.lin_vel_y_init, -cmd.vy_max, cmd.vy_max))
            cmd.ang_vel_yaw = float(np.clip(vyaw + cmd.ang_vel_yaw_init,
                                            -cmd.vyaw_max, cmd.vyaw_max))
            # repaint at ~5 Hz so the ramp is visible without flooding
            now = time.monotonic()
            if (not self.damping and now - self._status_t > 0.2):
                self._status_t = now
                print("\r\033[K" + self.teleop.status_line(), end="", flush=True)

        def stop(self, reason=None):
            if (reason == "fall"):
                self.log_event("fall")
                if (not self.damping):
                    print("\n[teleop] queda detectada -> DAMPING simulado "
                          "(Backspace reinicia)", flush=True)
                self.teleop.enter_damping()
                return
            super().stop(reason)

        def policy_step(self):
            if (self.damping):
                # the policy is not consulted; the target is unused by
                # ctrl_step below, returned only to keep the loop's contract
                self._step_count += 1
                self._elapsed_s = self._step_count * self.cfg.policy_dt
                if (self.cmd_log is not None):
                    # keep the timeline continuous: the take shows the damped
                    # stretch as rows at zero command, not as a gap
                    self.cmd_log.record(self._step_count, self.vel_command,
                                        getattr(self.policy, "_face_heading", None))
                return self.robot.data.joint_pos.clone()
            return super().policy_step()

        def ctrl_step(self, dof_targets):
            if (not self.damping):
                return super().ctrl_step(dof_targets)
            self.update_vel_command()
            kd = self._damping_kd
            effort = self.robot.effort_limit.numpy()
            for _ in range(self.decimation):
                dof_vel = self.mj_data.qvel[6:]
                self.mj_data.ctrl = np.clip(-kd * dof_vel, -effort, effort)
                mujoco.mj_step(self.mj_model, self.mj_data)

        def run(self):
            print(TELEOP_HELP, flush=True)
            print(self.teleop.describe(), flush=True)
            print("[teleop] digite no TERMINAL (nao na janela do viewer)", flush=True)
            if hasattr(self, "describe_pd"):
                print(self.describe_pd(), flush=True)
            reader = TerminalKeyReader(self._on_key)
            reader.start()
            try:
                return super().run()
            finally:
                reader.close()
                print()

    TerminalTeleopController.__name__ = "TerminalTeleop" + base_cls.__name__
    return TerminalTeleopController


def main():
    import argparse
    import pkgutil

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="t1_mimickit_steering")
    parser.add_argument("--pd", default="explicit", choices=sorted(CONTROLLERS),
                        help="damping scheme (see controllers.py)")
    parser.add_argument("--checkpoint", default=None,
                        help="policy checkpoint path (default: cfg.policy.checkpoint_path, "
                             "relative to the task dir unless absolute)")
    parser.add_argument("--effort", default=EFFORT_URDF, choices=sorted(EFFORT_SOURCES),
                        help="torque ceiling the PD loop clips at: 'urdf' is what the "
                             "policy trained against, 'derated' is what the firmware "
                             "enforces (see controllers.apply_effort_source)")
    parser.add_argument("--tn", action="store_true",
                        help="enable the catalogue torque-speed falloff "
                             "(see controllers.apply_tn_curve)")
    parser.add_argument("--degrade", default=DEGRADE_NONE, choices=sorted(DEGRADE_MODES),
                        help="'zero_root' zeroes root_pos_w and root_lin_vel_b, exactly "
                             "what the hardware reports (see controllers.make_degraded)")
    parser.add_argument("--imu-noise", dest="imu_noise", default=IMU_NOISE_NONE,
                        metavar="SPEC",
                        help="corrupt the simulated IMU: '{}' (default), a "
                             "preset ({}), or KEY=VALUE pairs - e.g. "
                             "'typical,pitch_bias_deg=3' or "
                             "'pitch_bias_deg=3'. Only root_quat_w and "
                             "root_ang_vel_b are touched; the PD loop keeps "
                             "reading true encoders, as the firmware does."
                             .format(IMU_NOISE_NONE,
                                     "/".join(sorted(IMU_NOISE_PRESETS))))
    parser.add_argument("--imu-seed", dest="imu_seed", type=int, default=0,
                        help="seed for the per-run IMU constants (bias, drift). "
                             "A preset is a RANGE: two seeds lean the robot "
                             "different ways, so the seed is part of the "
                             "result and is printed with it.")
    parser.add_argument("--hip-yaw-limit", dest="hip_yaw_limit", default=None,
                        metavar="BASE[,GAIN]",
                        help="cap the hip-yaw targets to default +/- (BASE + "
                             "GAIN*|yaw_cmd|) rad, e.g. '0.05,0.3'. Off by "
                             "default (see MimicKitSteeringPolicy._hip_yaw_clamp)")
    parser.add_argument("--heading-mode", dest="heading_mode", default=None,
                        metavar="world|start|body[,K]",
                        help="'world' integrates the yaw command from the IMU's "
                             "yaw zero (task default); 'start' from the heading "
                             "measured at start; 'body' re-anchors on the "
                             "measured heading every step, leading it by K*vyaw "
                             "(see docs/16). Pair with --imu-noise yaw_bias_deg=60 "
                             "to compare")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--log_states", default=None,
                        help="write <path>.npz of the rollout, for analysis")
    parser.add_argument("--teleop", default="viewer", choices=("viewer", "keyboard"),
                        help="'viewer': W/X A/D Q/E in the MuJoCo window (default); "
                             "'keyboard': the robot's terminal layout, see docs/17")
    from booster_deploy.utils.keyboard_teleop import add_teleop_args, limits_from_args
    add_teleop_args(parser)
    args = parser.parse_args()

    import tasks as tasks_pkg
    for mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod.name)
    from booster_deploy.utils.registry import get_task

    cfg = get_task(args.task)
    if (args.log_states is not None):
        cfg.mujoco.log_states = args.log_states
    if (args.checkpoint is not None):
        cfg.policy.checkpoint_path = args.checkpoint
    apply_effort_source(cfg, args.effort)
    apply_tn_curve(cfg, args.tn)
    cfg.policy.hip_yaw_limit = parse_hip_yaw_limit(args.hip_yaw_limit)
    heading = apply_heading_mode(cfg, args.heading_mode)

    base = make_degraded(CONTROLLERS[args.pd], args.degrade)
    base = make_noisy_imu(base, parse_imu_noise(args.imu_noise), args.imu_seed)
    print(f"[exp] pd={args.pd}  effort={args.effort}  tn={args.tn}  "
          f"degrade={args.degrade}  imu_noise={args.imu_noise}  "
          f"hip_yaw_limit={args.hip_yaw_limit}  heading={heading}  "
          f"teleop={args.teleop}")
    if (args.teleop == "keyboard"):
        # --vx/--vy/--yaw become the standing baseline the keys add to, which
        # is what deploy.py --vx means on the robot
        cfg.vel_command.lin_vel_x_init = args.vx
        cfg.vel_command.lin_vel_y_init = args.vy
        cfg.vel_command.ang_vel_yaw_init = args.yaw
        try:
            limits = limits_from_args(args, cfg.vel_command)
        except ValueError as e:
            parser.error(str(e))
        make_terminal_teleop_controller(base, limits)(cfg).run()
        return
    controller = make_keyboard_controller(base)(cfg)
    controller.vel_command.lin_vel_x = args.vx
    controller.vel_command.lin_vel_y = args.vy
    controller.vel_command.ang_vel_yaw = args.yaw
    controller._clamp_command()
    controller.run()


if __name__ == "__main__":
    main()
