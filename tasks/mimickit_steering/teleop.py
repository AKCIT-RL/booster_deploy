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

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from tasks.mimickit_steering.controllers import CONTROLLERS  # noqa: E402

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
            print(self.describe_pd(), flush=True)
            mujoco.viewer.launch_passive = launch_with_velocity_keys
            try:
                return super().run()
            finally:
                mujoco.viewer.launch_passive = real_launch

    KeyboardController.__name__ = "Keyboard" + base_cls.__name__
    return KeyboardController


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
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--log_states", default=None,
                        help="write <path>.npz of the rollout, for analysis")
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

    controller = make_keyboard_controller(CONTROLLERS[args.pd])(cfg)
    controller.vel_command.lin_vel_x = args.vx
    controller.vel_command.lin_vel_y = args.vy
    controller.vel_command.ang_vel_yaw = args.yaw
    controller._clamp_command()
    controller.run()


if __name__ == "__main__":
    main()
