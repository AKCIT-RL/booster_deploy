"""Valida o ambiente do robô ANTES de energizar.

Motivo de existir: a forma como uma dependência ausente se manifesta no deploy é
cara. `scripts/deploy.py` constrói a policy dentro do processo de inferência, que
é forkado DEPOIS de o robô já ter entrado em modo CUSTOM e já ter rampado até a
pose de preparo. Um `ModuleNotFoundError` trivial aparece, então, com o robô de
pé sob os ganhos de preparo, e o que o operador vê é:

    [INFO] Custom mode started, initialized with prepare pose
    [INFO] Inference process started
    ModuleNotFoundError: No module named 'gymnasium'
    [ERROR] Inference process died unexpectedly

Isto é, o robô se move, e só então o processo morre. Esta é a primeira falha real
que tivemos no T1, e ela é 100% detectável em terra, sem energizar nada.

Este script importa exatamente o que o deploy importa, na ordem em que o deploy
importa, e não move o robô.

Uso:

    python3 scripts/preflight.py --task t1_mimickit_steering
    python3 scripts/preflight.py --task t1_mimickit_steering --mujoco

Código de saída 0 se tudo passou, 1 se há qualquer FAIL. WARN não reprova.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"

_results: list[tuple[str, str, str]] = []


def record(status: str, check: str, detail: str = "") -> str:
    _results.append((status, check, detail))
    line = "[{}] {}".format(status, check)
    if detail:
        line += "\n         " + detail.replace("\n", "\n         ")
    print(line)
    return status


def try_import(mod: str):
    try:
        return importlib.import_module(mod), None
    except BaseException as e:  # SystemExit/ImportError de bindings nativos
        return None, e


def check_python() -> None:
    v = sys.version_info
    ver = "{}.{}.{}".format(v.major, v.minor, v.micro)
    if v[:2] == (3, 10):
        record(OK, "Python {} (versão do robô)".format(ver))
    elif v[:2] > (3, 10):
        record(WARN, "Python {} — o robô roda 3.10".format(ver),
               "Ok para sim2sim nesta máquina. No robô, NÃO crie venv com outro\n"
               "interpretador: rclpy e booster_robotics_sdk_python são extensões C\n"
               "compiladas contra o CPython 3.10 do sistema e não carregam em 3.11+.")
    else:
        record(FAIL, "Python {} é antigo demais (mínimo 3.10)".format(ver))


def check_venv_isolation() -> None:
    """Um venv sem --system-site-packages esconde rclpy e o SDK."""
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        record(OK, "Interpretador do sistema (vê rclpy e o SDK)")
        return
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    inherits = False
    if os.path.exists(cfg):
        with open(cfg) as f:
            inherits = "include-system-site-packages = true" in f.read()
    if inherits:
        record(OK, "venv com --system-site-packages (herda rclpy e o SDK)")
    else:
        record(WARN, "venv ISOLADO em {}".format(sys.prefix),
               "Sem --system-site-packages este venv não enxerga rclpy,\n"
               "booster_interface nem booster_robotics_sdk_python, e eles não podem\n"
               "ser reinstalados via pip. Para deploy real, recrie com:\n"
               "    python3 -m venv --system-site-packages env_deploy")


def check_core_deps() -> None:
    for mod, label in (("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy")):
        m, err = try_import(mod)
        if m is None:
            record(FAIL, "{} ausente".format(label), "{}".format(err))
        else:
            record(OK, "{} {}".format(label, getattr(m, "__version__", "?")))


def check_numpy_abi() -> None:
    """numpy 2.x quebra o ABI do rclpy do Humble, que é compilado contra 1.x."""
    np, err = try_import("numpy")
    if np is None:
        return
    major = int(np.__version__.split(".")[0])
    rclpy, _ = try_import("rclpy")
    if major >= 2 and rclpy is not None:
        record(WARN, "numpy {} (2.x) junto com rclpy".format(np.__version__),
               "O rclpy do ROS 2 Humble é compilado contra numpy 1.x. Se /low_state\n"
               "vier truncado ou com dtype errado, esta é a primeira suspeita.")


def check_hardware_stack(mujoco: bool) -> None:
    if mujoco:
        m, err = try_import("mujoco")
        if m is None:
            record(FAIL, "mujoco ausente (necessário para --mujoco)", str(err))
        else:
            record(OK, "mujoco {}".format(getattr(m, "__version__", "?")))
        return

    for mod, hint in (
        ("rclpy", "source /opt/booster/BoosterRos2Interface/install/setup.bash"),
        ("booster_interface.msg", "idem — vem do BoosterRos2Interface"),
        ("booster_robotics_sdk_python",
         "recompile o SDK com -DBUILD_PYTHON_BINDING=ON"),
    ):
        m, err = try_import(mod)
        if m is None:
            record(FAIL, "{} não importa".format(mod), "{}\n{}".format(err, hint))
        else:
            record(OK, "{} disponível".format(mod))

    m, _ = try_import("evdev")
    if m is None:
        record(WARN, "evdev ausente",
               "O controle por joystick cai para teclado — o deploy funciona.")


def check_task(task_name: str) -> object | None:
    """Registra as tasks do mesmo jeito que o deploy.py, e relata o que falhou.

    tasks/__init__ engole exceções de import por design (uma task quebrada não
    deve derrubar o registro das outras), e é por isso que `--list` nunca avisou
    que faltava gymnasium. Aqui a exceção é o produto, não o ruído.
    """
    import pkgutil
    import tasks

    failures = []
    for mod in pkgutil.walk_packages(tasks.__path__, prefix="tasks."):
        try:
            importlib.import_module(mod.name)
        except BaseException as e:
            failures.append((mod.name, e))

    from booster_deploy.utils.registry import get_task
    try:
        cfg = get_task(task_name)
    except BaseException as e:
        detail = str(e)
        for name, err in failures:
            if task_name.split("_", 1)[-1] in name or name.endswith(task_name):
                detail += "\nimport de {} falhou: {}: {}".format(
                    name, type(err).__name__, err)
        record(FAIL, "task '{}' não registrada".format(task_name), detail)
        return None

    record(OK, "task '{}' registrada".format(task_name))
    for name, err in failures:
        record(WARN, "outra task não registrou: {}".format(name),
               "{}: {}".format(type(err).__name__, err))
    return cfg


def check_mimickit(cfg) -> None:
    """A cadeia que observation.py carrega no PRIMEIRO passo de inferência."""
    from tasks.mimickit_steering import MIMICKIT_PATH

    if not os.path.isdir(MIMICKIT_PATH):
        record(FAIL, "checkout do MimicKit não encontrado",
               "procurado em {}\ndefina MIMICKIT_PATH apontando para "
               "<checkout>/mimickit".format(MIMICKIT_PATH))
        return
    record(OK, "MimicKit em {}".format(MIMICKIT_PATH))

    for mod in ("util.torch_util", "anim.mjcf_char_model",
                "envs.char_env", "envs.task_steering_env"):
        m, err = try_import(mod)
        if m is None:
            hint = ""
            if isinstance(err, ModuleNotFoundError) and err.name in (
                    "gymnasium", "yaml"):
                hint = ("\npip install {} — e NÃO rode o requirements.txt do repo "
                        "de treino:\nele pina numpy==2.5.0 (exige Python 3.12) e "
                        "quebra a resolução inteira.".format(
                            "PyYAML" if err.name == "yaml" else err.name))
            record(FAIL, "MimicKit: {} não importa".format(mod),
                   "{}: {}{}".format(type(err).__name__, err, hint))
            return
    record(OK, "cadeia de import do MimicKit completa")

    asset = cfg.policy.asset_path
    if not os.path.isabs(asset):
        asset = os.path.join(os.path.dirname(MIMICKIT_PATH), asset)
    if os.path.exists(asset):
        record(OK, "asset do personagem presente")
    else:
        record(FAIL, "asset não encontrado", asset)


def check_checkpoint(cfg) -> None:
    """Identifica o checkpoint pelo md5, não pelo nome.

    `deploy.py` não tem `--checkpoint`: no robô, `cfg.policy.checkpoint_path` é o
    único seletor da política. Os nomes de arquivo não bastam para conferir que a
    validação em simulação e o run no robô falam do mesmo modelo — já houve dois
    arquivos distintos com nomes parecidos. O md5 aqui é para comparar contra o
    md5 do run de sim, e é o único jeito barato de fechar essa dúvida.
    """
    path = cfg.policy.checkpoint_path
    if not os.path.isabs(path):
        import inspect
        mod = inspect.getmodule(cfg.policy.constructor)
        path = os.path.normpath(os.path.join(os.path.dirname(mod.__file__), path))
    if not os.path.exists(path):
        record(FAIL, "checkpoint não encontrado", path)
        return
    import hashlib
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    m, _ = try_import("torch")
    if m is None:
        return
    try:
        m.jit.load(path, map_location="cpu").eval()
    except BaseException as e:
        record(FAIL, "checkpoint não carrega", "{}: {}".format(type(e).__name__, e))
        return
    record(OK, "checkpoint carrega",
           "{}\nmd5 {}  — confira contra o run de sim".format(path, digest))


def check_handover(cfg) -> None:
    """A POSE de preparo deve bater com a da policy; os GANHOS não devem.

    Uma política de alvo absoluto de posição não corrige a partir da pose em que
    recebe o robô: ela comanda a sua. Se a rampa termina em outra pose, o
    primeiro passo é um degrau, com o robô já de pé.

    Os ganhos são o caso oposto, e essa checagem já esteve errada na direção
    contrária: exigia que os ganhos de preparo fossem os da policy, que é o que
    derrubou o T1 em 2026-09-15. Segurar uma pose estaticamente carrega o
    pêndulo invertido inteiro na rigidez das juntas; a política re-planeja a
    30 Hz e equilibra ativamente. Ganhos de preparo mais moles que ~200 no
    quadril/joelho põem o robô no chão durante a rampa. Ver a nota PREPARE STATE
    em mimickit_steering.py.
    """
    prep = getattr(cfg.robot, "prepare_state", None)
    if prep is None:
        return

    if list(prep.joint_pos) == list(cfg.robot.default_joint_pos):
        record(OK, "pose de preparo == a pose que a policy assume")
    else:
        record(WARN, "pose de preparo != default_joint_pos da task",
               "a policy comanda alvo ABSOLUTO: o primeiro passo seria um degrau")

    # 11 = Waist, 14 = Left_Knee_Pitch: as juntas que sustentam o robô de pé.
    weakest = min(prep.stiffness[11], prep.stiffness[14])
    if weakest >= 200:
        record(OK, "ganhos de preparo sustentam a pose (kp joelho {:.0f})".format(
            prep.stiffness[14]))
    else:
        record(FAIL, "ganhos de preparo moles demais para segurar o robô",
               "kp joelho {:.0f}, cintura {:.0f} — abaixo de ~200 o robô AFUNDA\n"
               "durante a rampa, antes mesmo da política assumir. Não use os\n"
               "ganhos da policy aqui; herde os de T1_23DOF_CFG.prepare_state."
               .format(prep.stiffness[14], prep.stiffness[11]))


def check_safety(cfg) -> None:
    if getattr(cfg.policy, "enable_safety_fallback", False):
        record(OK, "guard de queda armado (fall_gravity_z={})".format(
            getattr(cfg.policy, "fall_gravity_z", "?")))
    else:
        record(WARN, "guard de queda DESARMADO",
               "enable_safety_fallback=False nesta task")

    rate = getattr(cfg.policy, "max_target_rate", None)
    if rate is None:
        record(WARN, "limitador de taxa desativado",
               "max_target_rate=None — o alvo de posição pode dar um degrau de "
               "curso inteiro em um passo")
    else:
        record(OK, "limitador de taxa {} rad/s ({:.3f} rad/passo)".format(
            rate, rate * cfg.policy_dt))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="t1_mimickit_steering")
    p.add_argument("--mujoco", action="store_true",
                   help="valida o alvo sim2sim em vez do robô real")
    args = p.parse_args()

    print("=== preflight: {} ({}) ===\n".format(
        args.task, "mujoco" if args.mujoco else "robô real"))

    check_python()
    check_venv_isolation()
    check_core_deps()
    check_numpy_abi()
    check_hardware_stack(args.mujoco)

    cfg = check_task(args.task)
    if cfg is not None:
        if "mimickit" in args.task:
            check_mimickit(cfg)
        check_checkpoint(cfg)
        check_handover(cfg)
        check_safety(cfg)

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("\n=== {} OK, {} WARN, {} FAIL ===".format(
        len(_results) - len(fails) - len(warns), len(warns), len(fails)))

    if fails:
        print("\nNÃO energize o robô: o deploy falharia DEPOIS de entrar em modo\n"
              "CUSTOM e rampar até a pose de preparo, com o robô já de pé.")
        return 1

    print("\nAmbiente ok. Isto valida DEPENDÊNCIAS E CONFIGURAÇÃO, não a\n"
          "viabilidade da política: para t1_mimickit_steering, root_h e root_vel\n"
          "continuam indisponíveis no hardware (4 das 200 dimensões). Ver\n"
          "docs/08-caminho-para-hardware-mimickit.md antes de um teste no chão.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
