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
args = parser.parse_args()


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
