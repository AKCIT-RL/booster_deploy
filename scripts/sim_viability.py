"""Escada de viabilidade em simulação: degraus 0 (orçamento de torque) e 1 (pior caso).

Ver docs/08-caminho-para-hardware-mimickit.md.

Roda um rollout headless e mede duas coisas que não exigem o robô:

  degrau 0  orçamento de torque — o torque aplicado excede os limites derated do
            firmware (T1_EFFORT_FIRMWARE) em algum momento? A task sobrescreve
            effort_limit para os limites do URDF (joelho 130.5 Nm) e o MuJoCo clipa
            nesses; o firmware clipa nos derated (joelho 60 Nm). Se o rollout nunca
            passa do derated, o descasamento é irrelevante para esse envelope.

  degrau 1  pior caso — com --degrade zero_root, root_pos_w e root_lin_vel_b são
            zerados no update_state, reproduzindo exatamente a mentira do hardware
            (booster_robot_controller.py:244-249) dentro da simulação. Mede tempo até
            a queda e em que direção.

Uso:

    MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit .venv/bin/python \
        scripts/sim_viability.py --vx 1.0 --degrade none --duration 60

    MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit .venv/bin/python \
        scripts/sim_viability.py --vx 1.0 --degrade zero_root --duration 60

Nota sobre amostragem de torque: qfrc_actuator é lido uma vez por passo de política
(30 Hz na nossa task, 50 Hz na do fabricante), não por substep de física, então o pico
medido é um LIMITE INFERIOR
do pico real. Se ele já excede o derated, a questão está resolvida; se ficar marginal,
refinar a amostragem antes de concluir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEGRADE_NONE = "none"
DEGRADE_ZERO_ROOT = "zero_root"

# docs/info_sources/boosteractuatordata.pdf, tabela "Booster T1".
# (rated_tau_Nm, peak_tau_Nm, rated_rpm, peak_rpm)
# Rated/peak speed sao do EIXO DE SAIDA: com gear 18 e 48 V, 140 rpm de saida da
# ~2520 rpm de motor, que e a faixa de um BLDC desse porte. Se fossem velocidades
# de motor, o joelho andaria a 140/18 = 7.8 rpm, o que e absurdo.
T1_ACTUATORS = {
    "Neck":       (3, 7, 120, 400),
    "Arm":        (10, 36, 75, 89),
    "Waist":      (12, 40, 55, 70),
    "HipRollYaw": (12, 40, 55, 70),
    "HipPitch":   (20, 55, 155, 157),
    "Knee":       (25, 65, 132, 140),
    "Ankle":      (15, 50, 109, 117),
}
T1_JOINT_GROUPS = (["Neck"] * 2 + ["Arm"] * 8 + ["Waist"] +
                   ["HipPitch", "HipRollYaw", "HipRollYaw", "Knee", "Ankle", "Ankle"] * 2)
RPM = np.pi / 30.0  # rpm -> rad/s


def make_harness(base_cls):
    """Subclasse headless: comando fixo, sem polling de stdin.

    A degradação de estado vive em controllers.make_degraded, compartilhada com
    o teleop, para que a rodada batch e a rodada com GUI meçam a mesma coisa.
    """

    class Harness(base_cls):
        def update_vel_command(self):
            # O update_vel_command do MujocoController faz select() em stdin
            # (mujoco_controller.py:125) e imprime "Invalid input" a cada passo
            # quando stdin não é um terminal. Numa rodada batch o comando é fixo.
            return

    Harness.__name__ = f"{base_cls.__name__}Harness"
    return Harness


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """R^T . [0,0,-1] a partir de um quaternion wxyz (convenção MuJoCo)."""
    w, x, y, z = quat_wxyz
    # terceira coluna de R, negada = R^T aplicado a [0,0,-1]
    return -np.array([2 * (x * z + w * y),
                      2 * (y * z - w * x),
                      1 - 2 * (x * x + y * y)], dtype=np.float64)



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="t1_mimickit_steering")
    p.add_argument("--checkpoint", default="models/t1_wrturn_s1/policy.pt",
                   help="caminho do checkpoint (relativo ao cwd ou absoluto)")
    p.add_argument("--pd", default="explicit", choices=("explicit", "implicit", "none"),
                   help="esquema de damping; 'explicit' é o análogo do hardware")
    p.add_argument("--degrade", default=DEGRADE_NONE,
                   choices=(DEGRADE_NONE, DEGRADE_ZERO_ROOT))
    p.add_argument("--tn", action="store_true",
                   help="liga a queda de torque com a velocidade do catalogo do fabricante")
    p.add_argument("--json", default=None, help="grava o resumo como JSON (para varreduras)")
    p.add_argument("--effort", default="urdf", choices=("urdf", "derated", "catalog"),
                   help="teto de torque do laco PD: 'urdf' e o que a politica treinou, "
                        "'derated' e o que o firmware impoe (degrau 0b)")
    p.add_argument("--vx", type=float, default=1.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=60.0, help="segundos de tempo simulado")
    p.add_argument("--fall-gravity-z", type=float, default=-0.5,
                   help="limiar de queda em projected_gravity[2] (igual ao guard da locomotion)")
    p.add_argument("--out", default=None, help="grava .npz com as séries temporais")
    p.add_argument("--policy-dt", type=float, default=None,
                   help="sobrescreve a taxa de controle [s]; None mantem a da task")
    p.add_argument("--decimation", type=int, default=None,
                   help="sobrescreve substeps de fisica por passo de politica")
    p.add_argument("--gains", default="task", choices=("task", "default"),
                   help="'task' usa os ganhos da task; 'default' usa os de "
                        "robots/t1.py (o conjunto f_n=4 Hz, zeta=1.5)")
    args = p.parse_args()

    import tasks  # noqa: F401  (registra as tasks)
    import pkgutil
    for mod in pkgutil.walk_packages(tasks.__path__, prefix="tasks."):
        try:
            __import__(mod.name)
        except Exception:
            pass

    from booster_deploy.utils.registry import get_task
    from booster_deploy.robots.t1 import T1_23DOF_CFG
    from booster_deploy.robots.t1_actuators import T1_EFFORT_FIRMWARE
    from scripts.actuator_models import (
        CONTROLLERS, apply_effort_source, apply_gains, apply_rate,
        apply_tn_curve, make_degraded)

    cfg = get_task(args.task)
    cfg.policy.checkpoint_path = os.path.abspath(args.checkpoint)
    apply_effort_source(cfg, args.effort)
    apply_tn_curve(cfg, args.tn)
    apply_rate(cfg, args.policy_dt, args.decimation)
    apply_gains(cfg, args.gains)

    controller = make_harness(
        make_degraded(CONTROLLERS[args.pd], args.degrade))(cfg)
    controller.vel_command.lin_vel_x = args.vx
    controller.vel_command.lin_vel_y = args.vy
    controller.vel_command.ang_vel_yaw = args.yaw
    controller.start()

    # NOT T1_23DOF_CFG.effort_limit: upstream 7bb1462e repointed that at the URDF
    # values, which would silently make "derated" and "urdf" the same experiment.
    derated = np.asarray(T1_EFFORT_FIRMWARE, dtype=np.float64)
    used = np.asarray(cfg.robot.effort_limit, dtype=np.float64)
    dt = cfg.policy_dt
    n_steps = int(round(args.duration / dt))

    rec = {k: [] for k in ("t", "root_h", "speed", "grav_x", "grav_z", "pos_xy",
                           "tau", "dof_vel")}
    fall_step = None

    for i in range(n_steps):
        controller.update_state()
        targets = controller.policy_step()
        controller.ctrl_step(targets)

        qpos = controller.mj_data.qpos
        qvel = controller.mj_data.qvel
        grav = projected_gravity(np.asarray(qpos[3:7], dtype=np.float64))
        tau = np.asarray(controller.mj_data.qfrc_actuator[6:], dtype=np.float64)

        rec["t"].append((i + 1) * dt)
        rec["root_h"].append(float(qpos[2]))
        rec["speed"].append(float(np.linalg.norm(qvel[:2])))
        rec["grav_x"].append(float(grav[0]))
        rec["grav_z"].append(float(grav[2]))
        rec["pos_xy"].append([float(qpos[0]), float(qpos[1])])
        rec["tau"].append(tau.copy())
        rec["dof_vel"].append(np.asarray(qvel[6:], dtype=np.float64).copy())

        if fall_step is None and grav[2] > args.fall_gravity_z:
            fall_step = i + 1
            break

    data = {k: np.asarray(v) for k, v in rec.items()}
    names = T1_23DOF_CFG.joint_names

    # ---------------- relatório ----------------
    print()
    print("=" * 74)
    print(f"task={args.task}  pd={args.pd}  degrade={args.degrade}  "
          f"effort={args.effort}  tn={args.tn}")
    print(f"checkpoint={cfg.policy.checkpoint_path}")
    print(f"comando: vx={args.vx} vy={args.vy} yaw={args.yaw}   policy_dt={dt:.4f}s")
    print("=" * 74)

    cmd_speed = float(np.hypot(args.vx, args.vy))
    # t1_mimickit_steering clamps the command to the speed band it was trained on
    # (its steering_command does this internally, so the tracking denominator has
    # to match). Booster's t1_walk has no such band -- it takes the command as
    # given -- so the clamp only applies where the fields exist.
    tar_min = getattr(cfg.policy, "tar_speed_min", None)
    tar_max = getattr(cfg.policy, "tar_speed_max", None)
    if (tar_min is not None and tar_max is not None):
        cmd_speed = min(max(cmd_speed, tar_min), tar_max)
    elapsed = float(data["t"][-1])

    print("\n-- DEGRAU 1: sobrevivência --")
    if fall_step is None:
        print(f"   NAO CAIU em {elapsed:.2f}s ({len(data['t'])} passos)")
    else:
        dx, dy = data["pos_xy"][-1] - data["pos_xy"][0]
        along = dx * np.cos(0.0) + dy * np.sin(0.0) if cmd_speed == 0 else (
            (dx * args.vx + dy * args.vy) / max(cmd_speed, 1e-9))
        print(f"   CAIU em t={elapsed:.2f}s  (limiar projected_gravity[2] > {args.fall_gravity_z})")
        print(f"   deslocamento: dx={dx:+.3f} dy={dy:+.3f} m")
        print(f"   projecao do deslocamento na direcao comandada: {along:+.3f} m")
        print(f"   grav_x no instante da queda: {data['grav_x'][-1]:+.3f}  "
              f"(inclinacao no eixo de avanco)")
    print(f"   altura do tronco: inicial={data['root_h'][0]:.4f}  "
          f"min={data['root_h'].min():.4f}  final={data['root_h'][-1]:.4f} m")

    print("\n-- rastreamento de velocidade --")
    half = len(data["speed"]) // 2
    print(f"   comandado (apos clamp): {cmd_speed:.3f} m/s")
    print(f"   medido: media={data['speed'].mean():.3f}  "
          f"2a metade={data['speed'][half:].mean():.3f} m/s")
    if cmd_speed > 1e-9:
        print(f"   razao (2a metade / comando): "
              f"{data['speed'][half:].mean() / cmd_speed * 100:.1f}%")

    print("\n-- DEGRAU 0: orcamento de torque --")
    if args.effort == "derated":
        print("   (--effort derated: o teto JA e o do firmware, logo nada pode excede-lo;")
        print("    o que importa aqui e quanto tempo cada junta passa SATURADA)")
    print(f"   (qfrc_actuator amostrado a {1.0 / dt:.0f} Hz, fisica a "
          f"{1.0 / cfg.mujoco.physics_dt:.0f} Hz => pico medido é limite INFERIOR)")
    peak = np.abs(data["tau"]).max(axis=0)
    ceiling = np.asarray(cfg.robot.effort_limit, dtype=np.float64)
    sat = (np.abs(data["tau"]) >= ceiling * 0.999).mean(axis=0) * 100.0
    if sat.max() > 0.0:
        print(f"   juntas saturando no teto ativo ({args.effort}):")
        for j in np.argsort(-sat)[:6]:
            if sat[j] > 0.0:
                print(f"     {names[j]:<22}{sat[j]:>6.1f}% do tempo no teto de "
                      f"{ceiling[j]:.1f} Nm")
    over = peak > derated
    print(f"   juntas que excedem o limite derated do firmware: {int(over.sum())}/{len(names)}")
    if over.any():
        print(f"   {'junta':<22}{'pico':>9}{'derated':>10}{'urdf':>9}{'razao':>8}{'tempo>':>9}")
        order = np.argsort(-(peak / np.maximum(derated, 1e-9)))
        for j in order[:int(over.sum())]:
            print(f"   {names[j]:<22}{peak[j]:>9.2f}{derated[j]:>10.2f}"
                  f"{used[j]:>9.2f}{peak[j] / derated[j]:>7.2f}x"
                  f"{(np.abs(data['tau'][:, j]) > derated[j]).mean() * 100:>8.1f}%")
    else:
        duty = (np.abs(data["tau"]) > derated).mean(axis=0) * 100.0
        if duty.max() > 0.0:
            print(f"   acima do derated em algum instante: "
                  f"{int((duty > 0).sum())} juntas")
        worst = int(np.argmax(peak / np.maximum(derated, 1e-9)))
        print(f"   maior uso relativo: {names[worst]} "
              f"{peak[worst]:.2f} / {derated[worst]:.2f} Nm "
              f"({peak[worst] / derated[worst] * 100:.0f}% do derated)")

    print("\n-- VELOCIDADE DE JUNTA vs catalogo do fabricante --")
    print("   (docs/info_sources/boosteractuatordata.pdf)")
    if args.tn:
        print("   curva T-N LIGADA: acima do rated o torque disponivel decai e no peak e zero,")
        print("   entao estes numeros ja sao o resultado do envelope sendo imposto")
    else:
        print("   curva T-N DESLIGADA (velocity_limit=None): nada disto e imposto na")
        print("   simulacao; e so a comparacao entre o que a marcha pede e o catalogo")
    vpk = np.abs(data["dof_vel"]).max(axis=0)
    rows = []
    for j, nm in enumerate(names):
        g = T1_JOINT_GROUPS[j]
        rated, peak = T1_ACTUATORS[g][2] * RPM, T1_ACTUATORS[g][3] * RPM
        frac_rated = float((np.abs(data["dof_vel"][:, j]) > rated).mean() * 100)
        rows.append((vpk[j] / peak, nm, g, vpk[j], rated, peak, frac_rated))
    rows.sort(reverse=True)
    print(f"   {'junta':<22}{'pico':>8}{'rated':>8}{'peak':>8}{'pico/peak':>11}{'t>rated':>9}")
    for ratio, nm, g, v, rated, peak, fr in rows[:8]:
        flag = "  <<<" if ratio > 1.0 else ("  <" if fr > 0.0 else "")
        print(f"   {nm:<22}{v:>8.2f}{rated:>8.2f}{peak:>8.2f}{ratio:>10.2f}x"
              f"{fr:>8.1f}%{flag}")
    n_over_peak = sum(1 for r in rows if r[0] > 1.0)
    n_over_rated = sum(1 for r in rows if r[6] > 0.0)
    print(f"   juntas que passam do PEAK speed: {n_over_peak}/{len(names)}"
          f"   |   que passam do RATED speed: {n_over_rated}/{len(names)}")

    summary = {
        "task": args.task, "pd": args.pd, "degrade": args.degrade,
        "effort": args.effort, "tn": bool(args.tn),
        "vx": args.vx, "vy": args.vy, "yaw": args.yaw,
        "cmd_speed": cmd_speed, "duration_budget_s": args.duration,
        "fell": fall_step is not None, "elapsed_s": elapsed,
        "speed_mean": float(data["speed"].mean()),
        "speed_2nd_half": float(data["speed"][half:].mean()),
        "tracking_pct": (float(data["speed"][half:].mean() / cmd_speed * 100)
                         if cmd_speed > 1e-9 else None),
        "root_h_min": float(data["root_h"].min()),
        "n_joints_over_derated": int(over.sum()),
        "n_joints_over_catalog_peak": int((peak > np.array(
            [T1_ACTUATORS[g][1] for g in T1_JOINT_GROUPS])).sum()),
        "n_joints_over_rated_speed": int(n_over_rated),
        "n_joints_over_peak_speed": int(n_over_peak),
        "vel_peak_max_ratio": float(max(r[0] for r in rows)),
    }
    if fall_step is not None:
        dxy = data["pos_xy"][-1] - data["pos_xy"][0]
        summary["fall_dx"] = float(dxy[0])
        summary["fall_dy"] = float(dxy[1])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"resumo gravado em {args.json}")

    if args.out:
        peak_tau = np.array([T1_ACTUATORS[g][1] for g in T1_JOINT_GROUPS], dtype=np.float64)
        peak_vel = np.array([T1_ACTUATORS[g][3] * RPM for g in T1_JOINT_GROUPS])
        rated_vel = np.array([T1_ACTUATORS[g][2] * RPM for g in T1_JOINT_GROUPS])
        np.savez(args.out, joint_names=np.array(names), derated=derated,
                 effort_used=used, catalog_peak_tau=peak_tau,
                 catalog_peak_vel=peak_vel, catalog_rated_vel=rated_vel, **data)
        print(f"\nseries gravadas em {args.out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
