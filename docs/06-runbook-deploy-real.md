# 6. Runbook: deploy no robô real

> Referência: commit `0dba32b` (branch `tsinghua`), revisto contra o release upstream
> `7bb1462e` (2026-09-13) na branch `tsinghua-v2`. Este runbook complementa
> [02 — Modos e ciclo de vida](02-modos-e-ciclo-de-vida.md); leia aquele documento para
> entender *por que* cada passo existe.

## Pré-requisitos (`README.md:6-13`)

| Item | Requisito |
|---|---|
| Firmware Booster | `>= v1.4` |
| Python | 3.10+ (já instalado no robô) |
| ROS 2 | Humble, com `/low_state` + `/joint_ctrl` disponíveis (já instalado no robô) |
| SDK | **Não é mais necessário compilar o binding Python.** O release `7bb1462e` eliminou o `booster_robotics_sdk_python`: toda a comunicação passou a ser ROS 2 (`booster_interface.msg` + o serviço `booster_rpc_service`). O que precisa existir é a interface ROS 2 da Booster |
| Placa | No T1 Standard Edition, o deploy deve rodar na **placa de movimento**, não na de percepção (`README.md:48`) |

## Checklist antes de energizar

- [ ] Tarefa já validada em `--mujoco` (sim2sim) com o mesmo checkpoint que será usado no
      robô.
- [ ] Robô fisicamente suspenso ou apoiado com segurança para o primeiro teste de uma tarefa
      nova (a rampa de preparo e a política podem produzir movimento inesperado).
- [ ] Área ao redor do robô livre de obstáculos e pessoas na trajetória de movimento previsto.
- [ ] Acesso rápido a um kill switch físico ou ao controle remoto que dispara `kDamping`,
      independente do processo de deploy.
- [ ] Robô já está em modo `Prepare` (postura bipedal de apoio) antes de iniciar o script —
      não é o script que coloca o robô em `Prepare`.
- [ ] Amortecimento de tornozelo da tarefa é de **lado motor**, não de junta — ver
      [Kd de junta paralela vs. Kd de motor](#kd-de-junta-paralela-vs-kd-de-motor). Um erro de
      2× ou 1,67× aqui **não é observável em sim2sim**; só aparece no hardware.

> **Nota de versão (release `7bb1462e`):** o `--net` que este checklist pedia não existe mais —
> `scripts/deploy.py` perdeu `--net` e `--webots`, e o `ChannelFactory.Init` passou para dentro
> do `BoosterRobotPortal`. Não há mais interface de rede a informar na linha de comando.

## Kd de junta paralela vs. Kd de motor

> Acrescentado na reavaliação do release `7bb1462e` (2026-09-13). Ver
> [11 — Reavaliação](11-reavaliacao-release-2026-09.md).

O README do upstream passou a documentar, para juntas atuadas em paralelo:

```text
Kd = 2 * zeta * J_eq * (2 * pi * f_n)
```

com `J_eq` = `armature` da junta paralela, `f_n` a frequência natural e `zeta` o fator de
amortecimento. O texto vem com um aviso: `robot.joint_damping` **é enviado direto aos motores**,
então *não reutilize o Kd do simulador de treino*.

Vale entender por quê, porque o erro que isso previne não aparece em sim2sim.

### A fórmula não é ilustrativa — é o gerador dos ganhos publicados

Resolvendo a fórmula de trás para frente contra os 23 ganhos de `booster_deploy/robots/t1.py`,
com `J_eq` lido do `armature` do `T1_23dof.xml`, todo o `joint_stiffness`/`joint_damping` do T1
sai de **dois números globais** — via `Kp = J_eq (2 pi f_n)^2` e a fórmula acima:

| Grupo | `J_eq` (armature) | `f_n` resolvido | `zeta` resolvido |
|---|---:|---:|---:|
| Neck | 0,0018000 | **4,0000 Hz** | **1,500** |
| Arm | 0,0282528 | 4,0000 Hz | 1,500 |
| Waist / Hip-Roll / Hip-Yaw | 0,0478125 | 4,0000 Hz | 1,500 |
| Hip-Pitch | 0,0523908 | 4,0000 Hz | 1,500 |
| Knee | 0,0636012 | 4,0000 Hz | 1,500 |
| Ankle-Pitch | 0,0679104 | 4,0000 Hz | 0,750 ← |
| Ankle-Roll | 0,0203731 | 4,0000 Hz | 2,500 ← |

Reprodução exata até o último dígito. Os ganhos do T1 não são empíricos: são um **projeto de
segunda ordem a f_n = 4 Hz e zeta = 1,5**, uniforme no robô inteiro.

Uniforme exceto nos dois tornozelos, que é exatamente onde o mecanismo é paralelo.

### Os divisores `/2` e `/0.6`

O `robots/t1.py` escreve o amortecimento de tornozelo literalmente como
`5.120323529816263/2` e `1.5360970589448788/0.6`. Não é cosmético — é a correção paralela, e é o
que produz os `zeta` aparentes de 0,75 e 2,5 na tabela acima, se você ler o Kd como se fosse de
junta.

A armadura de **um** motor de tornozelo é `26,2e-6 × 36² = 0,0339552`. As duas armaduras do MJCF
são múltiplos exatos dela:

| Junta | `armature` do MJCF | razão para o motor único | leitura física |
|---|---:|---:|---|
| Ankle-Pitch | 0,0679104 | **2,0×** | os dois motores somam no pitch |
| Ankle-Roll | 0,0203731 | **0,6×** | razão da alavanca diferencial |

Dividir por 2 e por 0,6 desfaz exatamente esses fatores, e os dois tornozelos caem no **mesmo**
Kd de lado motor:

```
2 × 1,5 × 0,0339552 × 2π × 4      = 2,5601617649
5,120323529816263 / 2             = 2,5601617649
1,5360970589448788 / 0,6          = 2,5601617649
```

> Isto resolve a pendência de [08 §4d](08-caminho-para-hardware-mimickit.md): os dois `armature`
> que não fechavam em `inércia × gear²` não eram anomalia nem erro de asset — são `2,0×` e `0,6×`
> do motor único, e o próprio arquivo de config da Booster confirma os dois fatores.

### Onde o URDF entra (e onde não entra)

Os dois arquivos de asset são complementares, e nenhum sozinho fecha a conta:

| Arquivo | Fornece | **Não** fornece |
|---|---|---|
| `T1_23dof.urdf` | `effort=` — `7.0 / 38.3 / 68.0 / 98.8 / 130.5 / 73.1 / 17.2`, que é o `effort_limit` do release | **zero** elementos `<dynamics>` — nenhum amortecimento |
| `T1_23dof.xml` | `armature` por junta (o `J_eq` da fórmula) | nenhum `dof_damping` |

Ou seja: **o URDF fixa o teto de torque no lado da junta; a fórmula fixa o Kd no lado do motor.**
O URDF não tem opinião nenhuma sobre amortecimento. É por isso que o Kd do simulador de treino
não serve: ele é de junta, e no tornozelo o fator entre junta e motor é 2 ou 0,6 — errar isso
significa entregar ao motor o dobro, ou 1,67× menos, do amortecimento pretendido.

**Uma ressalva a manter aberta:** o teto de torque *não* escala como a armadura.
`73,1 / 17,2 = 4,25`, enquanto a razão das armaduras é `2 / 0,6 = 3,33`. O mecanismo que explica o
Kd, portanto, **não** explica sozinho o `17,2` do Ankle-Roll que [09](09-analise-graficos.md)
registra como `0,34×` do pico de catálogo. Medir a razão real com o `T1_23dof_parallel.xml`
(6 restrições de igualdade) é o próximo passo.

### Por que isto é uma armadilha de deploy

No caminho real, `RobotCfg.joint_damping` vai **direto para os motores**:
`booster_robot_controller.ctrl_step` publica apenas `q`, `kp` e `kd`, e quem fecha a malha é a
placa do motor. No `--mujoco`, o `MujocoController.ctrl_step` deste fork usa `kd = 0`
(ver [05](05-armadilhas-conhecidas.md), armadilha 1) e o `T1_23dof.xml` não declara
`dof_damping`.

O resultado é que **`joint_damping` não é exercitado em lugar nenhum da simulação**. Um Kd de
tornozelo errado por 2× ou por 1,67× passa incólume por todo o sim2sim — a validação não tem como
vê-lo — e só se manifesta no hardware, como oscilação de tornozelo (Kd baixo demais) ou tornozelo
mole e lento (Kd alto demais).

### Como conferir antes de energizar

```sh
.venv/bin/python - <<'EOF'
import math
from booster_deploy.robots.t1 import T1_23DOF_CFG
# armature do T1_23dof.xml, na ordem de joint_names
J = ([0.0018]*2 + [0.0282528]*8 + [0.0478125]
     + [0.0523908, 0.0478125, 0.0478125, 0.0636012, 0.0679104, 0.02037312]*2)
for n, kp, kd, j in zip(T1_23DOF_CFG.joint_names, T1_23DOF_CFG.joint_stiffness,
                        T1_23DOF_CFG.joint_damping, J):
    fn = math.sqrt(kp / j) / (2 * math.pi)
    print(f"{n:28} f_n={fn:6.3f} Hz   zeta={kd / (2 * j * 2 * math.pi * fn):6.3f}")
EOF
```

Espera-se `f_n = 4,000 Hz` em todas as 23 juntas. `zeta` deve dar `1,500` em todas, **exceto**
`ankle_pitch` (0,750) e `ankle_roll` (2,500) — esses dois desvios são a correção paralela e são
o sinal de que o valor está correto. Um `zeta` de 1,5 nos tornozelos significaria que alguém
"consertou" a inconsistência aparente e reintroduziu o bug.

## Procedimento

1. SSH na placa do robô; sourcing do ambiente ROS 2:
   ```bash
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   ```
2. Confirmar que a tarefa está registrada:
   ```bash
   python3 scripts/deploy.py --list
   ```
3. Iniciar o deploy:
   ```bash
   python3 scripts/deploy.py --task <TASK_NAME>
   ```
4. **Primeira confirmação** (prompt do `RemoteControlService`): dispara a entrada em modo
   CUSTOM e a rampa de 1 segundo até a pose de preparo (`prepare_state.joint_pos`). Observar o
   robô durante essa rampa — é o primeiro movimento comandado pelo script.
5. **Segunda confirmação**: inicia de fato o processo de inferência da política. A partir
   daqui, o robô está sob controle da política a 50 Hz.
6. Operar via joystick (`vx`, `vy`, `vyaw` conforme a tarefa expuser).

## Como abortar

- **`Ctrl-C`** no processo do `deploy.py`: dispara `exit_event`, o processo filho de
  inferência é finalizado (`join` com timeout de 2 s, `terminate()` se não responder), e o
  robô é colocado no **modo de saída configurado** — não mais fixo em `kWalking`
  (`booster_robot_controller.py:667-677`):

  ```python
  exit_mode = "damping" if self._safety_abort else self.cfg.booster.exit_mode
  ```

  O padrão é `"walking"`; `BoosterRobotControllerCfg(exit_mode="damping")` na config da tarefa,
  ou `--exit-mode damping` na linha de comando, muda isso. Um **abort por segurança** força
  `damping` independentemente da configuração.

  Escolher entre os dois é uma decisão de risco, não de conveniência: `walking` devolve o robô
  a um controlador que o mantém de pé, mas assume que ele *está* em condições de andar;
  `damping` solta os motores e é o destino certo se o robô já estiver em postura ruim.
- **Atenção:** não há rampa de torque na saída — o último `LowCmd` publicado continua valendo
  até a mudança de modo ser processada pelo firmware. Isso pode significar um curto intervalo
  em que o robô ainda está sob os ganhos da política, não zero.
- **Abort mais direto (emergência):** trocar manualmente para `RobotMode.kDamping` via
  controle remoto/API, **ciente de que isso solta todos os motores e o robô pode cair** — é
  a semântica documentada do próprio enum (`robot_shared.hpp:9`).

## Ao final da execução: métricas

`BoosterRobotPortal.cleanup()` imprime um resumo de frequência para dois eventos monitorados:

```python
# booster_deploy/controllers/booster_robot_controller.py:425-433
for name, metric in self.metrics.items():
    stats = metric.compute()
    print(
        f"METRICS {name}: count={stats['count']}, "
        f"freq={stats['freq_hz']:.3f}Hz, "
        f"mean_period={stats['mean_period_s']}, "
        f"min={stats['min_period_s']}, max={stats['max_period_s']}"
    )
```

| Métrica | Frequência esperada | O que indica se divergir |
|---|---|---|
| `low_state_handler` | ~500 Hz | Perda de amostras de `/low_state` ou atraso na thread dedicada — investigar carga de CPU/rede |
| `policy_step` | ~50 Hz | Política rodando mais devagar que `policy_dt` — inferência lenta demais para o hardware, ou contenção entre os dois processos |

Uma frequência de `policy_step` consistentemente abaixo do esperado é o sinal mais direto de
que o controle está degradando (comandos mais espaçados que o planejado), mesmo que nenhum
erro apareça no log.

## Ver também

- [02 — Modos e ciclo de vida](02-modos-e-ciclo-de-vida.md): detalhe de cada passo da
  sequência de troca de modo.
- [05 — Armadilhas conhecidas](05-armadilhas-conhecidas.md): comportamentos que podem
  surpreender durante um deploy real, especialmente a ausência de odometria e a falta de
  rampa de torque na saída.
- [11 — Reavaliação do release 2026-09](11-reavaliacao-release-2026-09.md): o que mudou neste
  runbook por causa do release, e o veredito de cada armadilha contra o código novo.
