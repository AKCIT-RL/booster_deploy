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


def _span(values, fmt):
    """Um numero se o grupo e uniforme, 'a-b' se nao.

    Existe porque a versao anterior reportava `valores[idx[0]]` como se fosse o
    grupo inteiro. Com `--kd ankle=5,ankle_pitch=2` isso imprimia "ankle -> 2"
    enquanto o roll estava em 5: a tabela afirmava algo falso sobre o estado que
    seria enviado ao robo, que e a unica coisa que ela existe para fazer.
    """
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        return fmt.format(lo)
    return (fmt + "-" + fmt.strip()).format(lo, hi)


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

    # Aplicar do grupo mais ABRANGENTE para o mais especifico, independente da
    # ordem em que o operador digitou. Sem isto, `--kd ankle_pitch=2,ankle=5`
    # e `--kd ankle=5,ankle_pitch=2` dao resultados diferentes, e so a segunda
    # forma faz o que as duas parecem dizer. Um agregado que apaga o ajuste fino
    # digitado antes dele e silencioso: os dois comandos rodam, o robo anda, e a
    # medicao sai do tornozelo errado.
    #
    # Especificidade = numero de juntas atingidas. Empates (hip_pitch e
    # ankle_pitch, ambos 2) nunca se sobrepoem, e o indice de digitacao desempata
    # para a ordem ser deterministica.
    def by_breadth(indexed_spec):
        order, (group, _, _) = indexed_spec
        return (-len(_group_indices(names, group)), order)

    for specs, attr in ((kp_specs, "joint_stiffness"),
                        (kd_specs, "joint_damping")):
        vals = list(getattr(robot, attr))
        for _, (group, mode, value) in sorted(enumerate(specs), key=by_breadth):
            for i in _group_indices(names, group):
                vals[i] = vals[i] * value if mode == "mul" else value
        setattr(robot, attr, vals)

    touched_kp = {g for g, _, _ in kp_specs}
    touched_kd = {g for g, _, _ in kd_specs}
    print("[tune] ganhos por grupo:")
    print("[tune]   {:12s} {:>5s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "grupo", "n", "kp_antes", "kp_depois", "kd_antes", "kd_depois", "kd/kp"))
    for group in _GAIN_GROUPS:
        if group not in touched_kp and group not in touched_kd:
            continue
        idx = _group_indices(names, group)
        if not idx:
            continue
        kp_new = [robot.joint_stiffness[i] for i in idx]
        kd_new = [robot.joint_damping[i] for i in idx]
        ratio = [d / p if p else float("nan") for p, d in zip(kp_new, kd_new)]
        print("[tune]   {:12s} {:5d} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
            group, len(idx),
            _span([before_kp[i] for i in idx], "{:.3f}"),
            _span(kp_new, "{:.3f}"),
            _span([before_kd[i] for i in idx], "{:.4f}"),
            _span(kd_new, "{:.4f}"),
            _span(ratio, "{:.4f}")))
        # Um grupo agregado que sai heterogeneo quase sempre e ajuste fino
        # sobrevivendo por baixo, que e o comportamento desejado - mas dizer isso
        # e mais barato que deixar o operador decidir se e bug.
        if len(set(kd_new)) > 1 or len(set(kp_new)) > 1:
            print("[tune]   {:12s} {:>5s} (heterogeneo: um grupo mais especifico "
                  "venceu em parte dele)".format("", ""))

    # Mexer em kp sem mexer em kd muda o amortecimento do laço, e a task inteira
    # é construída sobre uma razão kd/kp fixa (0.0637, de scripts/prepare_t1_asset
    # no repo de treino). Subir kp sozinho reduz zeta na raiz de kp e é o caminho
    # curto para um laço que oscila no robô e não oscilava em MuJoCo, onde esta
    # fork zera kd. Avisar é barato; descobrir no tornozelo, não.
    # Comparado por JUNTA, nao por nome de grupo: `--kp ankle=50 --kd
    # ankle_pitch=...,ankle_roll=...` cobre as quatro juntas, e a versao por nome
    # avisava assim mesmo porque "ankle" nao esta literalmente em touched_kd. Um
    # aviso que dispara quando nao devia treina o operador a ignora-lo, e este
    # existe para pegar uma queda real de zeta.
    kp_joints = {i for g in touched_kp for i in _group_indices(names, g)}
    kd_joints = {i for g in touched_kd for i in _group_indices(names, g)}
    only_kp = sorted(kp_joints - kd_joints)
    if only_kp:
        print("[tune]   AVISO: kp mudou sem kd em {} -> a razao kd/kp mudou e "
              "zeta caiu por sqrt(kp)".format(
                  ", ".join(names[i] for i in only_kp[:4])
                  + (" ..." if len(only_kp) > 4 else "")))
        print("[tune]   para preservar a razao da task, passe tambem "
              "--kd <grupo>=x<mesmo fator>; para seguir a formula da Booster, "
              "kd = 2*zeta*sqrt(kp*J_eq) (docs/14)")


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
    # Sempre, nao so quando `changed`: as flags de grupo nao alimentam `changed`,
    # e esta e a unica linha que mostra o que de fato vai para o MotorCmd.
    #
    # Pitch e roll aparecem separados porque sao slots independentes no LowCmd
    # (15/16 e 21/22 sob CMD_TYPE_SERIAL) e porque suas inercias de junta diferem
    # por 3,33x, entao o mesmo kd produz zeta bem diferente nos dois - ver
    # docs/14. Imprimir so um deles, como esta linha fazia, esconde exatamente a
    # assimetria que esta task precisa ajustar.
    def _one(joint):
        i = robot.joint_names.index(joint)
        return robot.joint_stiffness[i], robot.joint_damping[i]

    print("[tune] joelho kp={:.1f} kd={:.3f} | ankle_pitch kp={:.1f} kd={:.3f} "
          "| ankle_roll kp={:.1f} kd={:.3f}".format(
              *_one("Left_Knee_Pitch"), *_one("Left_Ankle_Pitch"),
              *_one("Left_Ankle_Roll")))
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
