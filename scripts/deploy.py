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
tune.add_argument("--kp", action="append", default=[], metavar="GRUPO=V",
                  help="kp por grupo de atuador: 'arm=12', 'knee=200', "
                       "'arm=x3' para multiplicar. Repetivel, e aceita varios "
                       "separados por virgula. --list-groups mostra os nomes.")
tune.add_argument("--kd", action="append", default=[], metavar="GRUPO=V",
                  help="kd por grupo, mesma sintaxe de --kp")
tune.add_argument("--list-groups", action="store_true", default=False,
                  help="imprime os grupos de atuador da task e sai (use com "
                       "--task; nada e enviado ao robo)")
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


# Grupos de atuador, pelos nomes de junta. Os sete primeiros são os grupos
# FÍSICOS de docs/09 ("Os tres tetos, em numeros") - juntas que compartilham
# atuador, catálogo e teto de torque. Os agregados abaixo existem porque é
# assim que se fala do robô na prática ("os bracos", "as pernas"), e resolver
# um agregado é mais seguro que digitar seis flags e esquecer uma.
#
# A ordem importa na impressão, não na resolução: um mesmo índice pode ser
# alcançado por vários grupos, e a última flag na linha de comando é a que
# vale, como em qualquer override.
_GAIN_GROUPS = {
    # grupos físicos
    "head":        ("AAHead_yaw", "Head_pitch"),
    "shoulder":    ("Shoulder",),
    "elbow":       ("Elbow",),
    "waist":       ("Waist",),
    "hip_pitch":   ("Hip_Pitch",),
    "hip_roll":    ("Hip_Roll",),
    "hip_yaw":     ("Hip_Yaw",),
    "knee":        ("Knee",),
    "ankle_pitch": ("Ankle_Pitch",),
    "ankle_roll":  ("Ankle_Roll",),
    # agregados
    "arm":         ("Shoulder", "Elbow"),
    "hip":         ("Hip_",),
    "ankle":       ("Ankle_",),
    "leg":         ("Hip_", "Knee", "Ankle_"),
    "all":         ("",),
}


def _group_indices(joint_names, group):
    pats = _GAIN_GROUPS[group]
    return [i for i, n in enumerate(joint_names)
            if any(pt in n for pt in pats)]


def _print_groups(joint_names):
    print("grupos de atuador validos para --kp / --kd:")
    for g in _GAIN_GROUPS:
        idx = _group_indices(joint_names, g)
        print("  {:12s} {:2d} juntas   {}".format(
            g, len(idx), ", ".join(joint_names[i] for i in idx[:4])
            + (" ..." if len(idx) > 4 else "")))


def _parse_gain_specs(specs, joint_names, flag):
    """['arm=12', 'knee=x2.5'] -> [(grupo, 'abs'|'mul', valor), ...].

    Recusa em vez de ignorar: um grupo mal digitado que passasse batido
    deixaria o operador convicto de ter mudado um ganho que segue no valor
    antigo, e o run inteiro mediria a coisa errada.
    """
    out = []
    for spec in specs:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                print("[tune] ERRO: --{} '{}' nao tem a forma GRUPO=VALOR"
                      .format(flag, item))
                sys.exit(1)
            group, raw = item.split("=", 1)
            group, raw = group.strip().lower(), raw.strip()
            if group not in _GAIN_GROUPS:
                print("[tune] ERRO: grupo '{}' desconhecido em --{}".format(
                    group, flag))
                _print_groups(joint_names)
                sys.exit(1)
            mode = "abs"
            if raw[:1].lower() == "x":
                mode, raw = "mul", raw[1:]
            try:
                value = float(raw)
            except ValueError:
                print("[tune] ERRO: valor '{}' em --{} {} nao e numerico"
                      .format(raw, flag, item))
                sys.exit(1)
            if value < 0.0:
                print("[tune] ERRO: --{} {} negativo".format(flag, item))
                sys.exit(1)
            out.append((group, mode, value))
    return out


def _apply_group_gains(cfg):
    """kp/kd por grupo. Roda DEPOIS das flags globais e das de tornozelo, para
    que a flag mais especifica vencha quando as duas forem passadas."""
    robot = cfg.robot
    names = list(robot.joint_names)
    kp_specs = _parse_gain_specs(args.kp, names, "kp")
    kd_specs = _parse_gain_specs(args.kd, names, "kd")
    if not kp_specs and not kd_specs:
        return

    before_kp = list(robot.joint_stiffness)
    before_kd = list(robot.joint_damping)

    for specs, attr in ((kp_specs, "joint_stiffness"),
                        (kd_specs, "joint_damping")):
        vals = list(getattr(robot, attr))
        for group, mode, value in specs:
            for i in _group_indices(names, group):
                vals[i] = vals[i] * value if mode == "mul" else value
        setattr(robot, attr, vals)

    touched_kp = {g for g, _, _ in kp_specs}
    touched_kd = {g for g, _, _ in kd_specs}
    print("[tune] ganhos por grupo:")
    print("[tune]   {:12s} {:>5s} {:>9s} {:>9s} {:>9s} {:>9s} {:>8s}".format(
        "grupo", "n", "kp_antes", "kp_depois", "kd_antes", "kd_depois", "kd/kp"))
    for group in _GAIN_GROUPS:
        if group not in touched_kp and group not in touched_kd:
            continue
        idx = _group_indices(names, group)
        if not idx:
            continue
        i = idx[0]
        kp_new = robot.joint_stiffness[i]
        kd_new = robot.joint_damping[i]
        print("[tune]   {:12s} {:5d} {:9.3f} {:9.3f} {:9.4f} {:9.4f} {:8.4f}"
              .format(group, len(idx), before_kp[i], kp_new,
                      before_kd[i], kd_new,
                      (kd_new / kp_new) if kp_new else float("nan")))

    # Mexer em kp sem mexer em kd muda o amortecimento do laço, e a task inteira
    # é construída sobre uma razão kd/kp fixa (0.0637, de scripts/prepare_t1_asset
    # no repo de treino). Subir kp sozinho reduz zeta na raiz de kp e é o caminho
    # curto para um laço que oscila no robô e não oscilava em MuJoCo, onde esta
    # fork zera kd. Avisar é barato; descobrir no tornozelo, não.
    only_kp = touched_kp - touched_kd
    if only_kp:
        print("[tune]   AVISO: kp mudou sem kd em {} -> a razao kd/kp mudou e "
              "zeta caiu".format(", ".join(sorted(only_kp))))
        print("[tune]   para preservar a razao da task, passe tambem "
              "--kd <grupo>=x<mesmo fator>")


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

    _apply_group_gains(cfg)

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

    if args.list_groups:
        # antes do device, do tuning e de qualquer coisa de rede: e uma
        # consulta, nao um deploy
        _print_groups(list(task_cfg.robot.joint_names))
        sys.exit(0)

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
