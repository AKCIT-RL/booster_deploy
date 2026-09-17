import argparse
import sys

sys.path.append(".")

parser = argparse.ArgumentParser()
# require either --task or --list (mutually exclusive)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Name of the configuration file.")
group.add_argument("-l", "--list", action="store_true", dest="list_tasks",
                   default=False, help="list available tasks")

parser.add_argument("--net", type=str, default="127.0.0.1",
                    help="Network interface for SDK communication.")
parser.add_argument("--mujoco", action="store_true", default=False,
                    help="deploy in mujoco simulation")
parser.add_argument("--webots", action="store_true", default=False,
                    help="deploy in webots simulation")
parser.add_argument(
    "--device", type=str, default="cpu",
    help="Device to run the evaluation on (e.g., 'cpu', 'cuda')")

# --- Actuator / command tuning, for debugging ON THE ROBOT -------------------
# These exist because teleop.py's equivalents are MuJoCo-only, and some failure
# modes do not reproduce in MuJoCo at all: T1_23dof.xml declares zero
# frictionloss and zero joint damping (docs/08 section 4d), so dry friction,
# gear backlash and the ankle's parallel linkage — everything that turns a
# chattering position target into an audible buzz — exist only on hardware.
# Tuning there therefore needs knobs there.
#
# All of them scale or override what the task config already declares; none
# invents a value. Every one is echoed at startup, because a run tuned by flag
# and not recorded is a measurement you cannot repeat.
tune = parser.add_argument_group("actuator tuning (debug)")
tune.add_argument("--kp-scale", type=float, default=None, metavar="X",
                  help="multiply every joint_stiffness by X")
tune.add_argument("--kd-scale", type=float, default=None, metavar="X",
                  help="multiply every joint_damping by X")
tune.add_argument("--ankle-kd", type=float, default=None, metavar="KD",
                  help="absolute joint_damping for the 4 ankle joints")
tune.add_argument("--ankle-kp", type=float, default=None, metavar="KP",
                  help="absolute joint_stiffness for the 4 ankle joints")
tune.add_argument("--target-lowpass", type=str, default=None, metavar="HZ",
                  help="first-order low-pass on the position target, in Hz "
                       "('none' disables). Use against target chatter: it "
                       "attacks the frequency content, which a rate limit "
                       "does not.")
tune.add_argument("--max-target-rate", type=str, default=None, metavar="RAD_S",
                  help="slew ceiling for the position target ('none' disables)")

# --- Standing velocity command -----------------------------------------------
# The command the policy receives with the operator's input at neutral, in
# physical units. 0.0 (the default) leaves every path bit-identical to not
# having these flags: on hardware the baseline is added after the remote's
# normalized axes are scaled, so adding zero reproduces the old expression
# exactly, and in MuJoCo the seed is the 0.0 it already was.
#
# These exist because the command is not state this process owns on hardware:
# _low_state_handler rewrites it from the remote at ~500 Hz, so there is no
# "set it once at startup" to be had. As a baseline it survives, and the remote
# keeps authority to add to it or cancel it.
#
# SAFETY: a non-zero --vx means the robot walks as soon as the policy takes
# over, with nobody touching the remote. Intended for a gantry run where the
# operator is watching the robot rather than a keyboard.
cmdv = parser.add_argument_group("standing velocity command")
cmdv.add_argument("--vx", type=float, default=0.0, metavar="M_S",
                  help="forward velocity with the remote at neutral [m/s]")
cmdv.add_argument("--vy", type=float, default=0.0, metavar="M_S",
                  help="lateral velocity with the remote at neutral [m/s]")
cmdv.add_argument("--vyaw", type=float, default=0.0, metavar="RAD_S",
                  help="yaw rate with the remote at neutral [rad/s]")
args = parser.parse_args()


def _apply_command(cfg):
    """Write the standing velocity command onto the task cfg, and say so.

    Echoed unconditionally when non-zero, because this is the one debug flag
    that makes the robot move on its own: a run where nobody remembers a --vx
    was passed is a run whose first second is unexplainable.
    """
    axes = (("vx", "lin_vel_x_init", "vx_max", "m/s"),
            ("vy", "lin_vel_y_init", "vy_max", "m/s"),
            ("vyaw", "ang_vel_yaw_init", "vyaw_max", "rad/s"))
    requested = {flag: getattr(args, flag) for flag, _, _, _ in axes}
    if not any(requested.values()):
        return

    if cfg.vel_command is None:
        print("[cmd] --vx/--vy/--vyaw ignorados: a task '{}' nao tem "
              "vel_command".format(args.task))
        return

    parts = []
    for flag, attr, max_attr, unit in axes:
        value = requested[flag]
        if value == 0.0:
            continue
        limit = getattr(cfg.vel_command, max_attr)
        if abs(value) > limit:
            # Refuse rather than clamp: asking for more than the envelope is an
            # operator error, and silently getting a different speed than the
            # one typed is exactly how a run stops being reproducible.
            print("[cmd] ERRO: --{} {} excede {}={} da task"
                  .format(flag, value, max_attr, limit))
            sys.exit(1)
        setattr(cfg.vel_command, attr, value)
        parts.append("{}={} {}".format(flag, value, unit))

    print("[cmd] comando permanente com o controle neutro: " + " | ".join(parts))
    print("[cmd] o robo comeca a se mover quando a politica assumir, sem "
          "acao do operador")


def _apply_tuning(cfg):
    """Apply the debug flags to a task cfg, and say what was applied."""
    robot = cfg.robot
    ankles = [i for i, n in enumerate(robot.joint_names) if "Ankle" in n]
    changed = []

    if args.kp_scale is not None:
        robot.joint_stiffness = [k * args.kp_scale for k in robot.joint_stiffness]
        changed.append("kp x{}".format(args.kp_scale))
    if args.kd_scale is not None:
        robot.joint_damping = [k * args.kd_scale for k in robot.joint_damping]
        changed.append("kd x{}".format(args.kd_scale))
    if args.ankle_kp is not None:
        for i in ankles:
            robot.joint_stiffness[i] = args.ankle_kp
        changed.append("ankle kp={}".format(args.ankle_kp))
    if args.ankle_kd is not None:
        for i in ankles:
            robot.joint_damping[i] = args.ankle_kd
        changed.append("ankle kd={}".format(args.ankle_kd))

    for flag, attr in (("target_lowpass", "target_lowpass_hz"),
                       ("max_target_rate", "max_target_rate")):
        raw = getattr(args, flag)
        if raw is None:
            continue
        if not hasattr(cfg.policy, attr):
            print("[tune] {} ignored: {} has no '{}'".format(
                flag, type(cfg.policy).__name__, attr))
            continue
        value = None if raw.lower() == "none" else float(raw)
        setattr(cfg.policy, attr, value)
        changed.append("{}={}".format(attr, value))

    _apply_command(cfg)

    if changed:
        print("[tune] " + " | ".join(changed))
        # The gains are what the operator is most likely to get wrong, so print
        # the ones that matter rather than trusting the multiplier was intended.
        knee = robot.joint_names.index("Left_Knee_Pitch")
        print("[tune] joelho kp={:.1f} kd={:.3f} | tornozelo kp={:.1f} kd={:.3f}"
              .format(robot.joint_stiffness[knee], robot.joint_damping[knee],
                      robot.joint_stiffness[ankles[0]],
                      robot.joint_damping[ankles[0]]))
    return cfg


def main():
    # load task registry and dispatch
    import pkgutil
    import tasks as tasks_pkg

    # auto-import all submodules under tasks (recursive) so they can register themselves
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        full_name = mod_info.name
        try:
            __import__(full_name)
        except Exception as e:
            raise e
    from booster_deploy.utils.registry import get_task, list_tasks

    if args.list_tasks:
        print("Available tasks:")
        for task_name, cfg in list_tasks().items():
            cls = type(cfg)
            full_cls = f"{cls.__module__}.{cls.__qualname__}"
            print(f"  {task_name}\t:\t{full_cls}")
        sys.exit(0)

    try:
        task_cfg = get_task(args.task)
    except KeyError:
        print(f"Unknown task '{args.task}'. Available tasks: {list(list_tasks().keys())}")
        sys.exit(1)

    # Set device for policy
    task_cfg.policy.device = args.device

    _apply_tuning(task_cfg)

    # decide how to run based on flags
    if args.mujoco:
        # run mujoco controller
        from booster_deploy.controllers.mujoco_controller import MujocoController

        MujocoController(task_cfg).run()
    else:
        # initialize network and run robot portal
        try:
            from booster_robotics_sdk_python import ChannelFactory  # type: ignore
            ChannelFactory.Instance().Init(0, args.net)
        except ImportError as e:
            print(
                "Error: booster_robotics_sdk_python is not installed.\n"
                "Please install it to use real robot deployment.\n"
                "For MuJoCo simulation, use --mujoco flag instead."
            )
            sys.exit(1)

        # adjust ankle dampings for webots
        if args.webots:
            ankles = [-8, -7, -2, -1]  # indices of ankle joints
            for i in ankles:
                task_cfg.robot.joint_damping[i] = 0.5

        from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal
        with BoosterRobotPortal(task_cfg, use_sim_time=args.webots) as portal:
            portal.run()


if __name__ == "__main__":
    main()
