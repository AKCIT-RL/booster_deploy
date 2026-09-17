"""Registrador e analisador de um deploy no robô real.

POR QUE ESTE ARQUIVO EXISTE

`log_states` só existe no `MujocoController` (`mujoco_controller.py`), então no
caminho real uma queda é inatribuível: não se distingue "a política pediu
besteira" de "o hardware não entregou". É o item 6 das correções independentes
de `docs/08` §7, e o que `docs/12` hoje resolve pedindo que alguém filme o run.

Este script NÃO toca o `deploy.py`. É um observador separado: sobe um nó ROS 2
próprio, assina `/low_state` e `/joint_ctrl`, e grava. Rode em outro terminal,
antes do `deploy.py`, e deixe correndo. Não há risco de interferir no laço de
controle porque ele não publica nada.

O QUE ELE MEDE, E POR QUE CADA COISA

  taxa            `/low_state` ~500 Hz e `/joint_ctrl` na taxa da política
                  (30 Hz nas tasks de steering, 50 Hz na `t1_walk`). É a
                  checagem que `docs/12` manda ler no fim do run, aqui com a
                  série inteira em vez de só a média.

  erro de         |q_comandado - q_medido| por junta. É a métrica de "afiado":
  rastreamento    um alvo de posição só vira movimento através do PD, e o erro
                  em regime é a assinatura direta de kp baixo demais para a
                  carga. Comparável entre juntas porque está em graus.

  cedência        o mesmo erro, com SINAL e contra a gravidade, nas juntas de
  de braço        braço. kp=4 nas tasks de steering contra 50 na `t1_walk`
                  prevê ~9,4° de cedência no ombro em roll na pose da task; se
                  o número medido bater, "braços soltos" está explicado e
                  quantificado.

  tau_est         torque estimado por junta contra os três tetos
                  (`t1_actuators`: firmware, URDF, pico de catálogo). É o que
                  resolve a Q1 de `docs/10` empiricamente, e o que diz se o
                  firmware está clipando de fato.

  chatter         reversões de direção por segundo no alvo comandado. A
                  hipótese de `docs/11` §5a é que o alvo desta política vibra a
                  ~12 Hz no tornozelo; medido no robô, confirma ou refuta.

  yaw             yaw inicial, deriva e variação total. É a rotação inicial
                  indesejada: as 5 dims de steering são direções de MUNDO
                  projetadas pelo yaw da IMU, que não tem referência absoluta.

  kp/kd na rede   o que os drivers de motor receberam de fato. Pega uma flag de
                  tuning que não foi aplicada, e distingue o regime de rampa
                  (kp 350) do regime da política (kp 80) no próprio log.

USO

    # no robô, em outro terminal, ANTES do deploy.py
    source /opt/booster/BoosterRos2Interface/install/setup.bash
    python3 scripts/log_deploy.py --task t1_mimickit_steering_meas --out run01

    # Ctrl-C encerra, grava run01.npz e imprime o relatorio

    # reanalisar depois, sem robô e sem ROS
    python3 scripts/log_deploy.py --replay run01.npz

    # validar a analise sem robô nenhum
    python3 scripts/log_deploy.py --selftest

A gravação e a análise são deliberadamente separadas: `analyze()` é função pura
sobre arrays, então `--replay` e `--selftest` exercitam exatamente o código que
roda no robô. Sem isso, a única forma de testar este arquivo seria energizando.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.append(".")


# --------------------------------------------------------------------------- #
# tetos de atuador, para a seção de torque
# --------------------------------------------------------------------------- #

# Tetos por grupo de atuador, de docs/09 secao "Os tres tetos, em numeros"
# (firmware = effort_limit derated da Booster; pico = peak torque do catalogo
# em docs/info_sources/boosteractuatordata.pdf). Tabelados aqui porque
# booster_deploy/robots/t1_actuators.py nao existe nesta branch - quando ele
# existir, _ceilings() o prefere e esta tabela deixa de ser usada.
_GROUP_CEILINGS = {           # grupo: (firmware, urdf, pico_catalogo)
    "Head":       (7.0,   7.0,  7.0),
    "Shoulder":   (18.0, 38.3, 36.0),
    "Elbow":      (18.0, 38.3, 36.0),
    "Waist":      (25.0, 68.0, 40.0),
    "Hip_Pitch":  (45.0, 98.8, 55.0),
    "Hip_Roll":   (25.0, 68.0, 40.0),
    "Hip_Yaw":    (25.0, 68.0, 40.0),
    "Knee":       (60.0, 130.5, 65.0),
    "Ankle_Pitch": (24.0, 73.1, 50.0),
    "Ankle_Roll": (15.0, 17.2, 50.0),
}


def _ceilings(joint_names):
    """Os tres tetos por junta: de t1_actuators se existir, senao tabelados.

    O modulo e preferido porque tem a assercao que falha se firmware e URDF
    coincidirem (docs/11 2a); a tabela e o fallback para uma branch que nao o
    tem. Retorna None se nem um nem outro cobrir todas as juntas, e a secao de
    torque entao cai para "so o pico medido" em vez de inventar um teto.
    """
    nj = len(joint_names)
    try:
        from booster_deploy.robots.t1_actuators import (
            T1_CATALOG_PEAK_TORQUE, T1_EFFORT_FIRMWARE, T1_EFFORT_URDF)
        out = {"firmware": np.asarray(T1_EFFORT_FIRMWARE, dtype=float),
               "urdf": np.asarray(T1_EFFORT_URDF, dtype=float),
               "catalogo": np.asarray(T1_CATALOG_PEAK_TORQUE, dtype=float),
               "fonte": "t1_actuators"}
        if all(len(v) == nj for k, v in out.items() if k != "fonte"):
            return out
    except Exception:
        pass

    firm = np.zeros(nj); urdf = np.zeros(nj); cat = np.zeros(nj)
    for i, nm in enumerate(joint_names):
        key = next((k for k in _GROUP_CEILINGS if k in nm), None)
        if key is None:
            return None
        firm[i], urdf[i], cat[i] = _GROUP_CEILINGS[key]
    return {"firmware": firm, "urdf": urdf, "catalogo": cat,
            "fonte": "docs/09 (tabelado)"}


# --------------------------------------------------------------------------- #
# análise — função pura, testável sem ROS
# --------------------------------------------------------------------------- #

def analyze(data, joint_names, policy_hz=None):
    """Monta o relatório a partir dos arrays gravados.

    `data` é o dict que vai para o .npz. Nenhuma dependência de ROS aqui: é o
    que permite --replay e --selftest exercitarem este mesmo caminho.
    """
    lines = []
    def w(s=""):
        lines.append(s)

    t = np.asarray(data["t"], dtype=float)            # [N] monotonic, /low_state
    q = np.asarray(data["q"], dtype=float)            # [N, J] medido
    dq = np.asarray(data["dq"], dtype=float)
    tau = np.asarray(data["tau_est"], dtype=float)
    rpy = np.asarray(data["rpy"], dtype=float)        # [N, 3]
    q_cmd = np.asarray(data["q_cmd"], dtype=float)    # [N, J] ZOH do ultimo cmd
    kp = np.asarray(data["kp"], dtype=float)          # [N, J] o que foi para a rede
    kd = np.asarray(data["kd"], dtype=float)
    cmd_t = np.asarray(data["cmd_t"], dtype=float)    # [M] eventos /joint_ctrl
    cmd_q = np.asarray(data["cmd_q"], dtype=float)    # [M, J]
    has_cmd = np.asarray(data["has_cmd"], dtype=bool) # [N] ja houve /joint_ctrl?

    n, nj = q.shape
    names = list(joint_names) if joint_names is not None else \
        [f"j{i:02d}" for i in range(nj)]
    dur = (t[-1] - t[0]) if n > 1 else 0.0

    w("=" * 78)
    w(f"amostras={n}  duracao={dur:.2f} s  juntas={nj}")
    w("=" * 78)

    # ---- taxas ----------------------------------------------------------- #
    w("")
    w("-- TAXAS --")
    if n > 2:
        dt = np.diff(t)
        w(f"   /low_state  media={1.0/dt.mean():7.1f} Hz   "
          f"p95_gap={np.percentile(dt, 95)*1e3:6.2f} ms  "
          f"max_gap={dt.max()*1e3:7.2f} ms")
        # um gap grande é perda de amostra ou thread travada, e o estado
        # congela silenciosamente enquanto isso (docs/10, Parte IV)
        stall = int((dt > 5e-3).sum())
        w(f"   gaps > 5 ms: {stall} ({100.0*stall/len(dt):.2f}% das amostras)")
    if len(cmd_t) > 2:
        cdt = np.diff(cmd_t)
        hz = 1.0 / cdt.mean()
        w(f"   /joint_ctrl media={hz:7.1f} Hz   "
          f"p95_gap={np.percentile(cdt, 95)*1e3:6.2f} ms  "
          f"max_gap={cdt.max()*1e3:7.2f} ms")
        if policy_hz:
            off = 100.0 * (hz - policy_hz) / policy_hz
            flag = "" if abs(off) < 5 else "   <-- FORA DO ESPERADO"
            w(f"   esperado {policy_hz:.0f} Hz pela task -> desvio {off:+.1f}%{flag}")
    else:
        w("   /joint_ctrl: NENHUMA mensagem - a politica nunca comandou")

    # A partir daqui só interessa a janela em que a política estava comandando.
    # Antes do primeiro /joint_ctrl o robô está em rampa ou parado, e misturar
    # as duas janelas contamina toda média.
    act = has_cmd
    if act.sum() < 10:
        w("")
        w("!! menos de 10 amostras com comando ativo: relatorio encerrado aqui")
        return "\n".join(lines), {}

    w("")
    w(f"-- JANELA COM POLITICA ATIVA: {act.sum()} amostras, "
      f"{t[act][-1] - t[act][0]:.2f} s --")

    qa, qca, taua, dqa = q[act], q_cmd[act], tau[act], dq[act]
    err = np.degrees(qca - qa)      # graus, com sinal: cmd - medido

    # ---- ganhos na rede -------------------------------------------------- #
    w("")
    w("-- kp/kd QUE CHEGARAM AOS DRIVERS (regime da politica) --")
    kpa, kda = kp[act], kd[act]
    uniq = np.unique(np.round(kpa, 3), axis=0)
    w(f"   conjuntos distintos de kp na janela: {len(uniq)}"
      + ("" if len(uniq) == 1 else "   <-- mudou durante o run"))
    w(f"   {'junta':22s} {'kp':>8s} {'kd':>8s}  kd/kp")
    for i, nm in enumerate(names):
        w(f"   {nm:22s} {kpa[-1, i]:8.2f} {kda[-1, i]:8.4f}  {kda[-1, i]/kpa[-1, i] if kpa[-1, i] else float('nan'):6.4f}")

    # ---- erro de rastreamento -------------------------------------------- #
    w("")
    w("-- ERRO DE RASTREAMENTO |cmd - medido| (graus) --")
    w("   a assinatura de 'afiado': erro em regime alto = kp baixo para a carga")
    w(f"   {'junta':22s} {'p50':>7s} {'p95':>7s} {'max':>7s} {'medio_com_sinal':>16s}")
    order = np.argsort(-np.abs(err).mean(axis=0))
    summary = {}
    for i in order:
        a = np.abs(err[:, i])
        w(f"   {names[i]:22s} {np.percentile(a,50):7.2f} {np.percentile(a,95):7.2f} "
          f"{a.max():7.2f} {err[:, i].mean():+16.2f}")
        summary[names[i]] = dict(p50=float(np.percentile(a, 50)),
                                 p95=float(np.percentile(a, 95)),
                                 mean_signed=float(err[:, i].mean()))

    # ---- cedência de braço ----------------------------------------------- #
    arm = [i for i, nm in enumerate(names)
           if any(k in nm for k in ("Shoulder", "Elbow"))]
    if arm:
        w("")
        w("-- CEDENCIA DE BRACO (erro medio COM SINAL, graus) --")
        w("   previsao analitica com kp=4 na pose da task: ombro_roll ~9.4,")
        w("   ombro_pitch ~4.4, elbow_yaw ~1.4. Com kp=50 seria 12,5x menor.")
        for i in arm:
            w(f"   {names[i]:22s} {err[:, i].mean():+7.2f}  "
              f"(|.| p95 {np.percentile(np.abs(err[:, i]), 95):6.2f})")
        w(f"   |erro| medio em todo o braco: "
          f"{np.abs(err[:, arm]).mean():.2f} graus")

    # ---- torque ---------------------------------------------------------- #
    w("")
    w("-- TORQUE tau_est vs TETOS --")
    ceil = _ceilings(names)
    peak = np.abs(taua).max(axis=0)
    if ceil is None:
        w("   (sem tabela de tetos para estas juntas: so o pico medido)")
        for i in order:
            w(f"   {names[i]:22s} pico {peak[i]:8.2f} Nm")
    else:
        w(f"   fonte dos tetos: {ceil['fonte']}")
        w(f"   {'junta':22s} {'pico':>8s} {'firmware':>9s} {'catalogo':>9s} "
          f"{'p/firm':>7s} {'t>firm':>7s}")
        for i in np.argsort(-peak / np.maximum(ceil['firmware'], 1e-9)):
            frac = 100.0 * (np.abs(taua[:, i]) > ceil['firmware'][i] * 0.99).mean()
            w(f"   {names[i]:22s} {peak[i]:8.2f} {ceil['firmware'][i]:9.2f} "
              f"{ceil['catalogo'][i]:9.2f} "
              f"{peak[i]/ceil['firmware'][i]:6.2f}x {frac:6.1f}%")
        over = int((peak > ceil['firmware'] * 0.99).sum())
        w(f"   juntas no teto de firmware: {over}/{nj}   "
          f"acima do pico de catalogo: {int((peak > ceil['catalogo']).sum())}/{nj}")

    # ---- chatter --------------------------------------------------------- #
    if len(cmd_q) > 4:
        w("")
        w("-- CHATTER DO ALVO (reversoes de direcao por segundo) --")
        nyq = (policy_hz / 2.0) if policy_hz else None
        w("   acima do teto de Nyquist o alvo esta oscilando na propria taxa")
        w("   de controle, e o PD so pode brigar com ele (docs/11 5a).")
        if nyq:
            w(f"   teto de Nyquist a {policy_hz:.0f} Hz: {nyq:.1f} rev/s")
        d = np.diff(cmd_q, axis=0)
        sgn = np.sign(d)
        rev = (np.diff(sgn, axis=0) != 0) & (sgn[1:] != 0)
        span = cmd_t[-1] - cmd_t[0]
        rps = rev.sum(axis=0) / max(span, 1e-9)
        step = np.degrees(np.abs(d))
        for i in np.argsort(-rps):
            w(f"   {names[i]:22s} {rps[i]:6.1f} rev/s   "
              f"passo mediano {np.percentile(step[:, i], 50):6.2f} graus   "
              f"p95 {np.percentile(step[:, i], 95):6.2f}")

    # ---- atitude e yaw --------------------------------------------------- #
    w("")
    w("-- ATITUDE E YAW --")
    ra = rpy[act]
    # projected_gravity[2] = -cos(roll)cos(pitch): a grandeza do guard de queda
    pg_z = -np.cos(ra[:, 0]) * np.cos(ra[:, 1])
    w(f"   projected_gravity[2]: pior={pg_z.max():+.4f}  inicial={pg_z[0]:+.4f}"
      "   (-1 = ereto, 0 = de lado)")
    w(f"     guard de queda dispara em > -0.5  ->  "
      f"{'DISPAROU' if pg_z.max() > -0.5 else 'nao chegou perto'}")
    w(f"   roll  [graus] inicial={math.degrees(ra[0,0]):+7.2f} "
      f"min={math.degrees(ra[:,0].min()):+7.2f} max={math.degrees(ra[:,0].max()):+7.2f}")
    w(f"   pitch [graus] inicial={math.degrees(ra[0,1]):+7.2f} "
      f"min={math.degrees(ra[:,1].min()):+7.2f} max={math.degrees(ra[:,1].max()):+7.2f}")
    yaw = np.unwrap(ra[:, 2])
    dyaw = math.degrees(yaw[-1] - yaw[0])
    span_a = max(t[act][-1] - t[act][0], 1e-9)
    w(f"   yaw   [graus] inicial={math.degrees(yaw[0]):+7.2f}  "
      f"final={math.degrees(yaw[-1]):+7.2f}  variacao={dyaw:+7.2f}")
    w(f"     taxa media de giro: {dyaw/span_a:+.2f} graus/s")
    w("     ATENCAO: o yaw da IMU nao tem referencia absoluta, e as 5 dims de")
    w("     steering sao direcoes de MUNDO projetadas por ele. Um yaw inicial")
    w("     nao-nulo apresenta a politica um erro de rumo igual a ele, e ela")
    w("     gira para corrigir. Compare 'inicial' com a rotacao observada.")
    w(f"     yaw inicial = {math.degrees(yaw[0]):+.2f} graus  ->  giro previsto "
      f"por esse mecanismo: {-math.degrees(yaw[0]):+.2f} graus")

    # ---- velocidade de junta --------------------------------------------- #
    w("")
    w("-- VELOCIDADE DE JUNTA (rad/s) --")
    vpk = np.abs(dqa).max(axis=0)
    for i in np.argsort(-vpk)[:8]:
        w(f"   {names[i]:22s} pico {vpk[i]:7.2f}")

    return "\n".join(lines), summary


# --------------------------------------------------------------------------- #
# gravação ao vivo (precisa de ROS 2)
# --------------------------------------------------------------------------- #

def record(args, joint_names):
    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from booster_interface.msg import LowCmd, LowState

    nj = len(joint_names)
    buf = {k: [] for k in ("t", "q", "dq", "tau_est", "rpy", "gyro",
                           "q_cmd", "kp", "kd", "has_cmd")}
    cmd_buf = {"cmd_t": [], "cmd_q": []}
    # último /joint_ctrl visto, mantido para amostrar junto com cada
    # /low_state: é zero-order hold, que é exatamente o que o driver de motor
    # faz entre dois comandos. Alinhar assim evita interpolar nada.
    latest = {"q": np.zeros(nj), "kp": np.zeros(nj), "kd": np.zeros(nj),
              "seen": False}

    rclpy.init()
    node = rclpy.create_node("booster_deploy_logger")
    t0 = time.monotonic()

    def on_state(msg):
        buf["t"].append(time.monotonic() - t0)
        q = np.zeros(nj); dq = np.zeros(nj); tau = np.zeros(nj)
        for i, mt in enumerate(msg.motor_state_serial):
            if i >= nj:
                break
            q[i], dq[i], tau[i] = mt.q, mt.dq, mt.tau_est
        buf["q"].append(q); buf["dq"].append(dq); buf["tau_est"].append(tau)
        buf["rpy"].append(np.asarray(msg.imu_state.rpy, dtype=float))
        buf["gyro"].append(np.asarray(msg.imu_state.gyro, dtype=float))
        buf["q_cmd"].append(latest["q"].copy())
        buf["kp"].append(latest["kp"].copy())
        buf["kd"].append(latest["kd"].copy())
        buf["has_cmd"].append(latest["seen"])

    def on_cmd(msg):
        q = np.zeros(nj); kp = np.zeros(nj); kd = np.zeros(nj)
        for i, mc in enumerate(msg.motor_cmd):
            if i >= nj:
                break
            q[i], kp[i], kd[i] = mc.q, mc.kp, mc.kd
        latest["q"], latest["kp"], latest["kd"] = q, kp, kd
        latest["seen"] = True
        cmd_buf["cmd_t"].append(time.monotonic() - t0)
        cmd_buf["cmd_q"].append(q.copy())

    # /low_state é BEST_EFFORT no publisher do firmware; pedir RELIABLE aqui
    # não casaria o QoS e a subscrição nunca receberia nada.
    node.create_subscription(
        LowState, "/low_state", on_state,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                   history=HistoryPolicy.KEEP_LAST))
    node.create_subscription(
        LowCmd, "/joint_ctrl", on_cmd,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                   history=HistoryPolicy.KEEP_LAST))

    print(f"[log] gravando /low_state e /joint_ctrl para '{args.out}.npz'")
    print("[log] Ctrl-C encerra e imprime o relatorio")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.duration and (time.monotonic() - t0) >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n[log] interrompido")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    data = {k: np.asarray(v) for k, v in buf.items()}
    data["cmd_t"] = np.asarray(cmd_buf["cmd_t"])
    data["cmd_q"] = (np.asarray(cmd_buf["cmd_q"]) if cmd_buf["cmd_q"]
                     else np.zeros((0, nj)))
    data["joint_names"] = np.asarray(joint_names)
    return data


# --------------------------------------------------------------------------- #
# autoteste: dados sinteticos, para exercitar analyze() sem robô
# --------------------------------------------------------------------------- #

def synth(joint_names, hz_state=500.0, hz_cmd=30.0, dur=6.0,
          yaw0_deg=35.0, arm_droop_deg=9.4):
    """Série sintética com os defeitos que o script deve encontrar.

    Injeta de propósito: yaw inicial de 35 graus, cedência de 9,4 graus no
    ombro em roll, e um alvo que reverte a cada passo no tornozelo. Se o
    relatorio não acusar os três, a analise está errada.
    """
    nj = len(joint_names)
    n = int(dur * hz_state)
    t = np.arange(n) / hz_state
    idx = {nm: i for i, nm in enumerate(joint_names)}

    q_cmd = np.zeros((n, nj)); q = np.zeros((n, nj))
    # marcha nominal a 1,5 Hz no quadril e no joelho
    gait = 0.35 * np.sin(2 * np.pi * 1.5 * t)
    for nm in ("Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Knee_Pitch",
               "Right_Knee_Pitch"):
        if nm in idx:
            q_cmd[:, idx[nm]] = gait
            q[:, idx[nm]] = gait - np.radians(1.2) * np.sign(gait)

    # cedência de braço: alvo fixo, medido abaixo dele
    for nm, droop in (("Left_Shoulder_Roll", arm_droop_deg),
                      ("Right_Shoulder_Roll", arm_droop_deg),
                      ("Left_Shoulder_Pitch", 4.4),
                      ("Right_Shoulder_Pitch", 4.4)):
        if nm in idx:
            q_cmd[:, idx[nm]] = -1.3 if "Roll" in nm else 0.2
            q[:, idx[nm]] = q_cmd[:, idx[nm]] - np.radians(droop)

    # chatter no tornozelo: reverte a cada passo de politica
    for nm in ("Left_Ankle_Pitch", "Right_Ankle_Pitch"):
        if nm in idx:
            per = max(int(hz_state / hz_cmd), 1)
            q_cmd[:, idx[nm]] = 0.10 * (-1.0) ** (np.arange(n) // per)
            q[:, idx[nm]] = q_cmd[:, idx[nm]] * 0.2

    rpy = np.zeros((n, 3))
    rpy[:, 0] = np.radians(1.5) * np.sin(2 * np.pi * 1.5 * t)
    rpy[:, 1] = np.radians(-2.0) + np.radians(1.0) * np.sin(2 * np.pi * 1.5 * t)
    # yaw parte do offset da IMU e o robô gira para "corrigir" o rumo
    rpy[:, 2] = np.radians(yaw0_deg) * (1.0 - 0.8 * t / dur)

    kp = np.tile(np.where([any(k in nm for k in ("Shoulder", "Elbow", "Head"))
                           for nm in joint_names], 4.0, 80.0), (n, 1))
    kd = kp * 0.0637
    tau = np.zeros((n, nj))
    for nm, amp in (("Left_Knee_Pitch", 62.0), ("Right_Knee_Pitch", 58.0),
                    ("Left_Hip_Pitch", 47.0), ("Right_Hip_Pitch", 44.0)):
        if nm in idx:
            tau[:, idx[nm]] = amp * np.sin(2 * np.pi * 1.5 * t)

    ci = np.arange(0, n, max(int(hz_state / hz_cmd), 1))
    return {
        "t": t, "q": q, "dq": np.gradient(q, axis=0) * hz_state,
        "tau_est": tau, "rpy": rpy, "gyro": np.zeros((n, 3)),
        "q_cmd": q_cmd, "kp": kp, "kd": kd,
        "has_cmd": np.ones(n, dtype=bool),
        "cmd_t": t[ci], "cmd_q": q_cmd[ci],
        "joint_names": np.asarray(joint_names),
    }


# --------------------------------------------------------------------------- #

def _joint_names_from_task(task):
    """Nomes de junta e taxa da política, lidos da task registrada.

    Feito via registro em vez de hardcode para que o log saia com a mesma
    ordenação que o deploy usa - é a única forma de o relatorio nomear a junta
    certa quando a ordem mudar.
    """
    import pkgutil
    import tasks as tasks_pkg
    for mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        try:
            __import__(mod.name)
        except Exception:
            pass
    from booster_deploy.utils.registry import get_task, list_tasks
    try:
        cfg = get_task(task)
    except KeyError:
        print(f"task '{task}' desconhecida. Disponiveis: "
              f"{list(list_tasks().keys())}")
        sys.exit(1)
    return list(cfg.robot.joint_names), 1.0 / cfg.policy_dt


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="t1_mimickit_steering_meas",
                   help="task registrada, so para ler ordem de juntas e taxa")
    p.add_argument("--out", default="deploy_log",
                   help="prefixo do .npz e do .txt de relatorio")
    p.add_argument("--duration", type=float, default=0.0,
                   help="encerra sozinho apos N s (0 = ate Ctrl-C)")
    p.add_argument("--replay", default=None,
                   help="reanalisa um .npz ja gravado, sem ROS")
    p.add_argument("--selftest", action="store_true",
                   help="roda a analise sobre dados sinteticos com defeitos")
    args = p.parse_args()

    if args.selftest:
        names, hz = _joint_names_from_task(args.task)
        data = synth(names)
        report, _ = analyze(data, names, policy_hz=hz)
        print(report)
        return

    if args.replay:
        z = np.load(args.replay, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        names = [str(x) for x in data.get("joint_names", [])] or None
        report, _ = analyze(data, names)
        print(report)
        return

    names, hz = _joint_names_from_task(args.task)
    data = record(args, names)
    if len(data["t"]) == 0:
        print("[log] NENHUMA amostra de /low_state. O topico esta publicando? "
              "Confira o `source` do setup.bash e se o robo esta ligado.")
        sys.exit(1)
    np.savez_compressed(args.out + ".npz", **data)
    report, _ = analyze(data, names, policy_hz=hz)
    print(report)
    with open(args.out + ".txt", "w") as f:
        f.write(report + "\n")
    print(f"\n[log] bruto: {args.out}.npz   relatorio: {args.out}.txt")
    print(f"[log] tamanho: {os.path.getsize(args.out + '.npz')/1e6:.1f} MB")


if __name__ == "__main__":
    main()
