# 12. Guia de teste no robô real — `t1_mimickit_steering`

> Guia operacional para o primeiro teste físico desta task, incorporando o que a tentativa de
> 2026-09-15 expôs. Pré-requisitos e procedimento genérico de deploy estão em
> [06 — Runbook](06-runbook-deploy-real.md); as correções e a justificativa de cada uma em
> [11 — Dependências e preflight](11-dependencias-e-preflight.md). Este documento é a
> sequência concreta, com o que observar em cada ponto e o que fazer quando der errado.

**Política sob teste:** `models/t1_wrturn_s1/policy.pt`, md5 `51d4ba3d275cff9ba8d385dcbc0bca51`.
É o default de `MimicKitSteeringPolicyCfg.checkpoint_path`, então `deploy.py`, `teleop.py` e
`sim_viability.py` rodam o mesmo arquivo sem flag.

**Aviso que não mudou:** esta task é **sim-only por construção**. `root_h` e `root_vel` — 4 das
200 dimensões — não têm sensor no T1 e chegam zeradas. As correções da [11](11-dependencias-e-preflight.md)
tornam um teste sobrevivível e atribuível; não tornam a task viável no chão. Ver
[08 §8](08-caminho-para-hardware-mimickit.md).

---

## Fase 0 — Na máquina de desenvolvimento (antes de ir ao robô)

```bash
cd ~/Projects/Humanoids/booster_deploy
export MIMICKIT_PATH=~/Projects/Humanoids/dev_tsinghua/MimicKit/mimickit
```

**0.1 Preflight local.** Deve sair `0 FAIL`. Anote o md5 do checkpoint — ele será comparado
com o do robô.
```bash
.venv/bin/python scripts/preflight.py --task t1_mimickit_steering --mujoco
```

**0.2 Linha de base.** Como a política anda com observação completa:
```bash
.venv/bin/python scripts/sim_viability.py --vx 1.0 --degrade none \
    --duration 60 --out baseline.npz
```

**0.3 O teste decisivo — `zero_root`.** Zera `root_pos_w` e `root_lin_vel_b`, exatamente o que
`booster_robot_controller.update_state` reporta no hardware. **Mesmo código, mesma condição,
risco zero.**
```bash
.venv/bin/python scripts/sim_viability.py --vx 1.0 --degrade zero_root \
    --duration 60 --out degraded.npz
```

**Critério de decisão:**

| resultado do 0.3 | o que fazer |
|---|---|
| cai em poucos segundos | **não vá ao robô.** O hardware faz a mesma coisa, com custo real. Prioridade passa a ser a odometria de pé de apoio ([08 §6](08-caminho-para-hardware-mimickit.md)). |
| sobrevive 60 s, rastreamento degradado | teste suspenso no pórtico é justificável, para medir a divergência sim↔real com o robô seguro. |
| sobrevive com rastreamento próximo da base | resultado inesperado — confira se o `--degrade` pegou (o nome da classe no log deve ser `ZeroRoot…`) antes de acreditar. |

**0.4 Com viewer**, se quiser ver a marcha antes:
```bash
.venv/bin/python tasks/mimickit_steering/teleop.py --pd explicit --vx 1.0
# W/X velocidade · A/D giro · Q/E strafe · Z zera · P imprime
```

---

## Fase 1 — Preparar o robô

**1.1 Código:**
```bash
cd ~/booster_deploy && git pull
```
Traz as correções, `requirements-robot.txt`, `preflight.py` e o checkpoint.

**1.2 Dependências** — arquivo diferente do da máquina de desenvolvimento:
```bash
pip install -r requirements-robot.txt
```
Sem `sudo`. Sem `--upgrade`. Sem venv de outro Python. Leia o cabeçalho do arquivo.

**1.3 Ambiente.** Vale pôr as duas linhas no `~/.bashrc` — `MIMICKIT_PATH` esquecido é a
próxima falha tardia esperando acontecer:
```bash
source /opt/booster/BoosterRos2Interface/install/setup.bash
export MIMICKIT_PATH=~/tsinghua/MimicKit/mimickit
```

---

## Fase 2 — Preflight no robô (obrigatório, não move nada)

```bash
cd ~/booster_deploy
python3 scripts/preflight.py --task t1_mimickit_steering
```

Exigir **`0 FAIL`**, e conferir duas linhas:

- `md5 51d4ba3d…` — **igual** ao da Fase 0. Se diferir, sim e robô estão rodando políticas
  diferentes e a validação não vale.
- `pose de preparo == a pose que a policy assume`
- `ganhos de preparo sustentam a pose (kp joelho 350)` — os ganhos de preparo **não** são os
  da policy, e não devem ser. Ver [11 §4a](11-dependencias-e-preflight.md).

Com FAIL, **não energize**. O `deploy.py` constrói a policy dentro do processo de inferência,
forkado depois do modo CUSTOM e da rampa — a falha apareceria com o robô já de pé.

---

## Fase 3 — Preparação física

- [ ] **Robô suspenso no pórtico**, pés sem carga. Não é opcional neste teste.
- [ ] `preflight` com `ganhos de preparo sustentam a pose (kp joelho 350)` — a checagem que
      teria pego o colapso na rampa de 15/09.
- [ ] Kill switch físico ao alcance de quem **não** está no teclado.
- [ ] Segunda pessoa com acesso ao `kDamping` pelo controle remoto, ciente de que isso **solta
      todos os motores** e o robô cai no pórtico.
- [ ] Área livre; ninguém no envelope de movimento das pernas.
- [ ] Robô já em modo `Prepare` antes de iniciar o script.
- [ ] `--net` correto (default `127.0.0.1` para rodar na própria placa).
- [ ] Alguém filmando. Não há `log_states` no caminho real (só existe no `MujocoController`,
      `mujoco_controller.py:170`), então **vídeo é o único registro do que aconteceu**.

---

## Fase 4 — O run, e o que observar em cada ponto

```bash
python3 scripts/deploy.py --task t1_mimickit_steering
```

Confira a linha `[mimickit_steering] policy: …/t1_wrturn_s1/policy.pt` antes de qualquer tecla.

### Ponto 1 — tecla `x`: modo CUSTOM + rampa de 1 s

A rampa leva o robô à **pose de treino** sob os **ganhos de preparo da Booster** (`kp` joelho
= 350). Só a pose foi trocada; os ganhos não. Ver [11 §4a](11-dependencias-e-preflight.md) —
uma versão anterior trocava os dois e **derrubou o robô aqui**, porque segurar uma pose parada
exige rigidez que a política, que equilibra ativamente a 30 Hz, não precisa ter.

| o que acontece na rampa | significado |
|---|---|
| chega à pose e segura firme | normal. Siga. |
| **afunda lentamente até o chão** | ganhos de preparo moles demais. Rode o preflight: a linha `ganhos de preparo sustentam a pose` tem de estar OK, com kp joelho 350. **Não é conclusão sobre a política** — ela nunca segura pose parada. |
| oscila / vibra na pose | amortecimento de preparo baixo demais para o link. Aborte e investigue antes do `r`. |

Observe o joelho e o quadril. A rampa são 500 passos de 2 ms.

### Ponto 2 — tecla `r`: a política assume

A **pose** é contínua, então a política não comanda um salto de postura. Os **ganhos** não são:
eles caem de 350 para 80 neste instante, e isso é inerente ao projeto — a mesma troca que a
`locomotion` faz (350 → 200), um fator maior aqui.

Espere portanto **uma cedida breve** no momento do `r`, enquanto a política assume o
equilíbrio. O que NÃO é aceitável é a cedida continuar: se o robô seguir descendo em vez de se
estabilizar no primeiro segundo, aborte. Essa é a diferença entre o transiente do handover e a
política não estar equilibrando.

A partir daqui a política roda a 30 Hz com `root_h = 0` e `root_vel = 0`. O que esperar, em
ordem de probabilidade:

1. **Movimento de marcha no ar** com os pés sem carga — o pórtico remove a realimentação de
   contato, então isso não valida marcha; valida que o laço fecha e as juntas respondem.
2. **Guard de queda dispara** (`Falling detected, stopping policy for safety`) se o tronco
   passar de ~60° — parada controlada. É o comportamento correto.
3. **Divergência rápida.** Esperada, dada a lacuna de observação. Aborte.

### Ponto 3 — comandos

`w`/`s` vx · `a`/`d` vy · `q`/`e` vyaw · `Espaço` para. **Comece em `vx = 0`** e suba devagar.
Note que `tar_speed_min = 0.0` significa que parado é **extrapolação** — o treino amostrou
velocidades em `[min, max]` e nunca viu um comando de parada
([08 §7](08-caminho-para-hardware-mimickit.md), item 2).

### Ponto 4 — ao sair

`Ctrl-C` dispara `exit_event`, encerra o processo de inferência e muda para `kWalking`.
**Não há rampa de torque na saída** — o último `LowCmd` continua valendo até o firmware
processar a troca de modo.

Ao final, anote as métricas impressas:
```
METRICS low_state_handler: freq≈500Hz     <- saúde do link
METRICS policy_step:       freq≈30Hz      <- a política de fato rodou
```
`policy_step count=0` significa que a inferência nunca rodou.

---

## Troubleshooting

### Falhas de ambiente (aparecem antes de o robô se mexer, se você rodar o preflight)

| Sintoma | Causa | Correção |
|---|---|---|
| `ModuleNotFoundError: No module named 'gymnasium'` | `envs.char_env` importa `gymnasium.spaces` | `pip install gymnasium`. **Não** rode o `requirements.txt` do repo de treino. |
| `No module named 'yaml'` | idem, via `anim.motion_lib` | `pip install PyYAML` |
| `[mimickit_steering] not registered: …` no `--list` | `MIMICKIT_PATH` errado ou checkout ausente | `export MIMICKIT_PATH=~/tsinghua/MimicKit/mimickit` (aponta para o subdiretório `mimickit`, não para a raiz do repo) |
| `ERROR: Could not find a version that satisfies numpy==2.5.0` | você está no `requirements.txt` de **treino** | use `requirements-robot.txt`. Aquele exige Python 3.12. |
| `Command 'python3.11' not found` | tentativa de venv com outro Python | não faça. `rclpy` e o SDK são extensões C do CPython 3.10 do sistema. Se precisar de venv: `python3 -m venv --system-site-packages` |
| `No module named 'rclpy'` / `'booster_robotics_sdk_python'` | shell sem `source` do ROS, ou venv isolado | `source /opt/booster/BoosterRos2Interface/install/setup.bash`; se em venv, recrie com `--system-site-packages` |
| `/low_state` truncado, dtype estranho | numpy 2.x contra `rclpy` do Humble | volte o numpy do sistema para 1.x. Nunca faça `pip install -U numpy` no Python do sistema. |
| `asset não encontrado` no preflight | `data/assets/t1/t1.xml` ausente no checkout do MimicKit | confira que `MIMICKIT_PATH` aponta para o checkout completo |
| `ValueError: joint order differs between the asset and the robot config` | asset e `RobotCfg.joint_names` divergiram | **não contorne.** Com a ordem errada, cada comando vai para o atuador errado. Ver `observation.assert_joint_order_matches`. |
| md5 do checkpoint ≠ o da Fase 0 | sim e robô com políticas diferentes | `git pull`; confira `[mimickit_steering] policy:` na saída do deploy |

### Falhas durante o run

| Sintoma | Causa provável | Ação |
|---|---|---|
| `Waiting for '/joint_ctrl' subscriber, retry in 0.5s` em loop | firmware não está escutando; robô não está em `Prepare`, ou placa errada | ponha em `Prepare`; no T1 Standard Edition o deploy roda na **placa de movimento**, não na de percepção |
| `Inference process died unexpectedly` | exceção no processo filho — o traceback aparece **acima** dessa linha | leia o traceback, não a mensagem. É quase sempre ambiente; o preflight pega antes. |
| Robô **afunda até o chão durante a rampa** (antes do `r`) | ganhos de preparo moles demais — foi o que a 1ª versão da correção causou | confira `prepare_state.stiffness` (kp joelho deve ser 350, herdado de `T1_23DOF_CFG`). **Não** é veredito sobre a política: ver [11 §4a](11-dependencias-e-preflight.md) |
| Salto de **postura** no instante do `r` | pose de preparo divergiu do `default_joint_pos` da task | rode o preflight: `pose de preparo == a pose que a policy assume` |
| **Cedida** no instante do `r`, que estabiliza | queda de ganho 350→80, inerente ao projeto | esperado. Ver [11 §4a](11-dependencias-e-preflight.md) |
| Cedida no `r` que **não** estabiliza | a política não está equilibrando | aborte |
| `Falling detected, stopping policy for safety` | guard de queda funcionou (tronco >~60°) | comportamento correto. Se disparar cedo demais no pórtico por causa da suspensão, ajuste `fall_gravity_z`, **ciente** de que isso afrouxa o guard. |
| Divergência em ~1 s com o robô no chão | lacuna de observabilidade (`root_h`/`root_vel`) | esperado. Ver [08](08-caminho-para-hardware-mimickit.md). Não é ajustável por ganho. |
| `METRICS policy_step: count=0` | a inferência nunca rodou | a política morreu antes do primeiro passo; veja o traceback |
| `low_state_handler` bem abaixo de 500 Hz | link DDS saturado ou CPU competindo | feche o resto; confira `--net` |
| Juntas na ordem errada / robô se contorce | ordem de juntas ou ganho por junta errado | aborte imediatamente via `kDamping`. Não repita. |

### Abortos, em ordem de preferência

1. **`Ctrl-C`** no `deploy.py` — encerra a inferência e volta para `kWalking`. Sem rampa de
   torque; há um intervalo curto sob os ganhos da política.
2. **`kDamping`** pelo controle remoto — **solta todos os motores**. No pórtico é seguro; no
   chão o robô cai. É a semântica documentada do enum (`robot_shared.hpp:9`).
3. **Kill switch físico.**

---

## O que registrar depois, independente do resultado

Não há logging no caminho real, então isto tem de ser manual — e sem isso o teste não gera
conhecimento, só risco:

- vídeo do run inteiro, incluindo a rampa;
- as duas linhas de `METRICS`;
- comportamento na rampa (segurou? cedeu? quanto?);
- tempo entre o `r` e a primeira divergência;
- comandos dados e em que ordem;
- md5 do checkpoint e `git rev-parse HEAD`.

Portar `log_states` para o `BoosterRobotController` é o item 6 da
[08 §7](08-caminho-para-hardware-mimickit.md) e o que tornaria o próximo teste atribuível sem
depender de vídeo.
