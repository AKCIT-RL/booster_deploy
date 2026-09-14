# 8. Caminho para hardware: `t1_mimickit_steering`

> Referências: `booster_deploy` commit `0dba32b` (branch `tsinghua`); `booster_robotics_sdk` e
> `dev_tsinghua/` no checkout local; `htwk-gym` commit `cbb4f51`.
>
> Plano de execução derivado de [07 — Método comparado](07-htwk-gym-metodo-comparado.md). O
> diagnóstico está lá; aqui está o que fazer com ele, e os **resultados dos dois primeiros
> degraus**, já medidos.

## Resumo do que já está medido

Duas restrições de hardware, isoladas uma a uma em simulação, mesmo checkpoint
(`t1_wrturn_s1`) e mesmo comando. Runlist completa na §9, figuras em
[09 — Análise gráfica](09-analise-graficos.md).

| Degrau | Resultado |
|---|---|
| **0 — demanda de torque** | 11–12 das 23 juntas pedem acima do teto derated; o quadril chega a **1,58× o pico físico** do catálogo do fabricante. |
| **0b — teto do firmware** | A política **anda**, e o rastreamento melhora (99,1% / 99,8%). A 1,0 m/s cai aos 52,8 s. |
| **0c — envelope como especificado** | Efeito de **interação**: teto derated + curva T-N derruba em **9,6 s** a 1,0 m/s, onde cada fator sozinho dá 52,8 s e 60 s. A 0,5 m/s nenhum dos seis modelos derruba. |
| **1 — pior caso com zeros** | **Cai em ~1,0 s** nas duas velocidades, **para frente, na direção do movimento**, após acelerar a 235% do comando. |
| **Combinado** | ~1,0 s com ou sem o teto derated — dominado pela observabilidade. |

**A lacuna de observabilidade é a variável dominante, por ~50× no tempo de sobrevivência.** O
envelope de atuador é de segunda ordem, mas define um limite operacional: a fronteira física
fica **entre 1,3 e 1,6 m/s**, e sob o modelo de torque do treino não existe fronteira nenhuma
até 2,5 m/s.

> **Duas correções registradas.** (1) Ao ver só os picos do degrau 0, concluí que o
> descasamento de torque inviabilizava o checkpoint sozinho; o degrau 0b mostrou que não —
> excesso de demanda medido não é falha sob oferta limitada, o laço PD satura e a marcha segue.
> (2) O PDF do fabricante então mostrou que o teto do URDF contra o qual a política treinou é
> **não-físico** (joelho 130,5 Nm contra um pico real de 65), o que refaz a leitura do degrau 0
> mais uma vez. Em ambos os casos a medição direta corrigiu a inferência.

## 1. Inventário do que o T1 realmente expõe

Levantado no SDK, não suposto. Define o que qualquer estimador pode usar.

| Fonte | Campos | Ref. |
|---|---|---|
| `LowState` | `imu_state`, `motor_state_parallel`, `motor_state_serial` | `idl/b1/LowState.h:229-231` |
| `ImuState` | `rpy`, `gyro`, **`acc`** | `idl/b1/ImuState.h:207-209` |
| `MotorState` | `q`, `dq`, **`ddq`**, **`tau_est`** | `idl/b1/MotorState.h:286-289` |
| `Odometer` | `x`, `y`, `theta` — pose planar, **sem z, sem velocidade** | `idl/b1/Odometer.h:189-191` |
| `rt/odometer_state` | o `Odometer` acima, consumido pelo `MoveController` | `b1_api_const.hpp:36`, `move_controller.hpp:287,312` |
| `rt/odom` | "ROS-compatible odometry, requires the ROS bridge" — um `nav_msgs/Odometry` carrega **twist** | `b1_api_const.hpp:99` |

1. **Confirmado: não há sensor de pressão nem flag de contato de pé.** `LowState` tem
   exatamente três campos. Contato terá que ser inferido.
2. **`ImuState.acc` e `MotorState.ddq`/`tau_est` existem e não são lidos** — o handler em
   `booster_robot_controller.py:231-232` consome só `rpy` e `gyro`.
3. **Pode já existir odometria de perna no firmware.** `rt/odom`, se o bridge estiver ativo,
   é onde apareceria. Não verificável sem o robô (seção 6).

### Geometria da sola, de graça

`htwk-gym/envs/T1/Base_Walk.yaml:80-83` declara os cantos da sola relativos ao frame do pé:
sola de 0,223 × 0,100 m, a **z = −0,03 m**. Esse offset entra direto no cálculo de altura do
tronco e não precisa ser medido.

## 2. Ambiente de teste com `uv`

O workspace tem duplicatas e artefatos ignorados pelo git; os caminhos abaixo são os que
funcionam, verificados.

### Onde cada coisa realmente está

| Precisa | Caminho correto | Observação |
|---|---|---|
| `booster_assets` | `../dev_tsinghua/booster_assets` | **não** existe em `../booster_assets` |
| asset t1 do MimicKit | `../dev_tsinghua/MimicKit/mimickit` | `tsinghua/MimicKit/data/assets/.gitignore` é `*`, então `data/assets/t1/t1.xml` **não existe** naquele checkout |
| checkpoint | `models/t1_wrturn_s1/policy.pt` | 200 dims, mesma arquitetura do `t1_steering.pt` |

`BOOSTER_ASSETS_DIR` é `parents[1]` do pacote
(`booster_assets/src/booster_assets/__init__.py`), logo resolve para a raiz do repo
`booster_assets` e `{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml` existe.

### Setup

```sh
cd booster_deploy

uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    torch mujoco scipy numpy gymnasium pyyaml

# booster_assets como pacote instalado, em vez de PYTHONPATH
uv pip install --python .venv/bin/python -e ../dev_tsinghua/booster_assets

# só necessário para deploy.py no robô real (joystick); não para MuJoCo
uv pip install --python .venv/bin/python evdev
```

Resolvido: `uv 0.11.8`, CPython 3.12.3, `torch 2.14.0+cpu`, `mujoco`, `booster-assets 1.0.0`
editável. Torch CPU é suficiente — a rede é 200→1024→512→23.

`gymnasium` e `pyyaml` existem porque `envs.char_env` do MimicKit os importa; o resto do
`MimicKit/requirements.txt` (diffusers, moviepy, wandb, pyglet, tensorboardX) não é necessário
para inferência.

### Instalar o `booster_assets` em vez de usar `PYTHONPATH`

O comando em uso hoje é:

```sh
PYTHONPATH=../booster_assets/src/ MIMICKIT_PATH=../tsinghua/MimicKit/mimickit/ DISPLAY=:1 \
    python3 tasks/mimickit_steering/teleop.py --pd explicit --vx 0.0 \
    --checkpoint .../models/t1_wrturn_s1/policy.pt
```

Dois dos caminhos não resolvem neste workspace: `../booster_assets/src/` não existe, e
`../tsinghua/MimicKit` não tem o asset t1. A forma equivalente com o venv:

```sh
MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit DISPLAY=:1 \
    .venv/bin/python tasks/mimickit_steering/teleop.py --pd explicit --vx 0.5 \
    --checkpoint models/t1_wrturn_s1/policy.pt
```

Com o `booster_assets` instalado, `PYTHONPATH` deixa de ser necessário. `MIMICKIT_PATH`
continua sendo, porque o MimicKit não é um pacote instalável.

> **Sobre `--vx 0.0`:** o treino amostrou `tar_speed` em Uniforme(0,50 · 2,50) — recuperado do
> normalizador do próprio `t1_wrturn_s1` (média 1,4963, desvio 0,5779). Comandar zero entrega
> **−2,6σ**. `tar_speed_min` está em `0.0` (`mimickit_steering.py:170`), então o clamp não
> impede. Há uma tensão real aqui: com `0.5` não é possível comandar "parado", e o commit
> `0dba32b` mudou `0.5 → 0.0` com a mensagem "fix: minimum speed" — aparentemente para que
> `--vx 0.0` significasse parar. O que o dado diz é que **esta política não tem modo parado**;
> o valor `0.0` não cria um, só entrega uma observação que ela nunca viu. Use `--vx 0.5` como
> ponto de partida de qualquer validação.

## 3. Harness: `scripts/sim_viability.py`

Roda um rollout headless e mede os degraus 0 e 1 na mesma passada. Neutraliza o polling de
stdin de `MujocoController.update_vel_command` (`mujoco_controller.py:125`), que imprime
"Invalid input" a cada passo quando stdin não é terminal, e aplica a degradação opcional.

Duas flags, uma por restrição de hardware:

| flag | isola |
|---|---|
| `--degrade zero_root` | a observação que o robô não tem (`root_pos_w`, `root_lin_vel_b`) |
| `--effort derated` | o teto de torque que o firmware impõe |

```sh
# linha de base: nada degradado
MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit .venv/bin/python \
    scripts/sim_viability.py --vx 1.0 --duration 60

# degrau 0b: teto do firmware, estado verdadeiro
MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit .venv/bin/python \
    scripts/sim_viability.py --vx 1.0 --effort derated --duration 60

# degrau 1: zeros em root_pos_w e root_lin_vel_b, como no hardware
MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit .venv/bin/python \
    scripts/sim_viability.py --vx 1.0 --degrade zero_root --duration 60

# combinado: o mais parecido com o robô real
MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit .venv/bin/python \
    scripts/sim_viability.py --vx 1.0 --degrade zero_root --effort derated --duration 60
```

### Com GUI

As mesmas duas flags existem no `teleop.py`, então cada célula da matriz pode ser assistida no
viewer em vez de lida numa tabela. A de ver é a `zero_root`:

```sh
MIMICKIT_PATH=../dev_tsinghua/MimicKit/mimickit DISPLAY=:1 \
    .venv/bin/python tasks/mimickit_steering/teleop.py \
    --pd explicit --vx 0.5 --degrade zero_root \
    --checkpoint "$PWD/models/t1_wrturn_s1/policy.pt"
```

O robô acelera acima do comando e tomba para frente em cerca de um segundo. `Backspace` reseta
no viewer, então a queda pode ser repetida à vontade; `Space` pausa e as setas andam quadro a
quadro, que é como se vê o sub-passo acontecendo.

> `--checkpoint` relativo no `teleop.py` resolve contra o **diretório da task**, não contra o
> cwd — daí o `$PWD`. O `sim_viability.py` resolve contra o cwd. Inconsistência real entre as
> duas ferramentas.

A degradação é exatamente o que `booster_robot_controller.py:244-249` entrega:

```python
def update_state(self):
    super().update_state()
    if degrade == DEGRADE_ZERO_ROOT:
        self.robot.data.root_pos_w.zero_()
        self.robot.data.root_lin_vel_b.zero_()
```

Queda é detectada com o mesmo critério do guard da `locomotion`
(`locomotion.py:65`): `projected_gravity[2] > -0.5`.

## 4. Degrau 0 — demanda de torque

A task sobrescreve `effort_limit` para os limites do URDF (joelho 130,5 Nm); o firmware clipa
nos derated de `T1_23DOF_CFG.effort_limit` (joelho 60 Nm). A pergunta era se a política chega a
pedir acima do derated.

Pico de `|qfrc_actuator|` em 60 s, `--pd explicit`, checkpoint `t1_wrturn_s1`:

| Junta | pico a 0,5 m/s | pico a 1,0 m/s | derated | URDF | razão (1,0 m/s) |
|---|---|---|---|---|---|
| `Right_Hip_Roll` | 46,66 | **56,33** | 25,00 | 68,00 | **2,25×** |
| `Left_Hip_Pitch` | 83,89 | **87,14** | 45,00 | 98,80 | **1,94×** |
| `Right_Hip_Yaw` | 35,33 | 48,57 | 25,00 | 68,00 | 1,94× |
| `Right_Hip_Pitch` | 85,24 | 85,82 | 45,00 | 98,80 | 1,91× |
| `Left_Hip_Roll` | 45,26 | 44,37 | 25,00 | 68,00 | 1,77× |
| `Right_Ankle_Pitch` | 36,44 | 40,99 | 24,00 | 73,10 | 1,71× |
| `Left_Hip_Yaw` | 39,15 | 37,59 | 25,00 | 68,00 | 1,50× |
| `Right_Knee_Pitch` | 86,93 | 83,71 | 60,00 | 130,50 | 1,40× |
| `Left_Ankle_Pitch` | 32,46 | 33,26 | 24,00 | 73,10 | 1,39× |

**11 juntas a 0,5 m/s, 12 a 1,0 m/s**, de 23. E `qfrc_actuator` é amostrado a 30 Hz, não nos
120 Hz dos substeps, então **são limites inferiores** do pico real.

Duas leituras que se sustentam:

1. O excesso não está concentrado no joelho (o caso que o README da task cita) — está no
   **quadril**, em pitch, roll e yaw, que são as juntas que sustentam e direcionam o corpo.
2. Não é fenômeno de recuperação nem de velocidade alta. Acontece em **marcha normal a
   0,5 m/s**, o regime mais suave que faz sentido comandar. `Right_Hip_Roll` passa **12% do
   tempo** acima do derated a 1,0 m/s — é duty cycle, não pico isolado.

O que esses números **não** respondem é se a política *falha* quando o teto é imposto. Demanda
excedente e falha sob saturação são perguntas diferentes, e a segunda precisa do degrau 0b.

## 4b. Degrau 0b — teto de torque do firmware: a política anda

`--effort derated` troca o teto do laço PD pelos limites derated do `T1_23DOF_CFG`, mantendo
todo o resto igual. É o isolamento da variável de torque, do mesmo jeito que `--degrade
zero_root` isola a de observabilidade.

| | teto URDF (130,5 Nm no joelho) | teto derated (60 Nm) |
|---|---|---|
| **vx = 0,5** | 60 s sem cair · rastreio **96,8%** | 60 s sem cair · rastreio **99,1%** |
| **vx = 1,0** | 60 s sem cair · rastreio **88,8%** | **cai em 52,8 s** (após ~49 m) · rastreio **99,8%** |

Três coisas:

1. **A política anda com o torque que o robô tem.** Isso reverte a conclusão anterior. A
   saturação não quebra a marcha — o laço PD fica no teto mais tempo e o passo se mantém.
2. **O rastreamento melhora**, de 96,8% para 99,1% e de 88,8% para 99,8%. Hipótese, não
   afirmação: com `kp = 80` e alvos de posição absolutos, o laço estava sobre-acionando, e o
   clip no derated funciona como limitador. Vale investigar, porque se for isso, o teto mais
   baixo é *benéfico* e não só tolerável.
3. **Mas reduz robustez.** A 1,0 m/s a sobrevivência cai de 60 s+ para 52,8 s. O quadril em
   pitch passa **22–29% do tempo saturado** nessa velocidade. É um envelope mais estreito, não
   um impedimento.

Ressalva do modelo: um clip plano é a versão **otimista**. Atuadores reais perdem torque com
velocidade, que é o que `velocity_limit`/`knee_point_velocity` descrevem (ambos `None` aqui,
curva T-N desligada). Sobreviver a este teste não prova sobreviver ao envelope torque-velocidade
real.

## 4d. O catálogo do fabricante muda a leitura do degrau 0

Fonte: `docs/info_sources/boosteractuatordata.pdf`, tabela "Booster T1", dados do fabricante.
Transcrito em `booster_deploy/robots/booster.py` como `T1_ACTUATOR_CATALOG` — e os defaults de
`T1_23DOF_CFG` ficam intactos, a aplicação é explícita via flag.

### O teto contra o qual a política treinou é não-físico

| Grupo | `T1_EFFORT_URDF` (treino) | pico do catálogo | razão |
|---|---|---|---|
| Joelho | 130,5 | **65** | **2,01×** |
| Hip-Pitch | 98,8 | **55** | 1,80× |
| Waist / Hip-Roll / Hip-Yaw | 68,0 | **40** | 1,70× |
| Ankle-Pitch | 73,1 | **50** | 1,46× |
| Braço | 38,3 | 36 | 1,06× |

O atuador do joelho fisicamente não passa de 65 Nm e o treino permitia 130,5. Então a §4 estava
comparando a demanda contra um teto "generoso" que **nenhum derating alcança**: os picos de
85–87 Nm no hip pitch são 1,58× o pico *físico* de 55 Nm, não apenas acima de um limite
operacional.

Na direção oposta, `T1_23DOF_CFG.effort_limit` fica **abaixo** do pico do catálogo em todas as
juntas (0,30× a 1,00×). O `--effort derated` da §4b portanto clipou **mais apertado que o teto
físico** — e andou. A questão de magnitude está resolvida de forma mais favorável do que a §4
sugeria.

### O eixo que nenhum experimento tinha tocado

`velocity_limit` e `knee_point_velocity` são **ambos `None`** no `T1_23DOF_CFG`, então a curva
T-N do `ctrl_step` nunca executou. O PDF é a primeira fonte que permite preenchê-los: rated
speed → `knee_point_velocity`, peak speed → `velocity_limit`.

| Grupo | rated | peak |
|---|---|---|
| Hip-Pitch | 16,23 rad/s | 16,44 rad/s |
| Joelho | 13,82 | 14,66 |
| Ankle | 11,41 | 12,25 |
| **Waist / Hip-Roll / Hip-Yaw** | **5,76** | **7,33** |
| Braço | 7,85 | 9,32 |

Hip-roll, hip-yaw e waist têm o menor envelope de velocidade do robô — é onde a curva morde
primeiro.

Medindo a marcha contra isso (estado verdadeiro, sem impor nada):

| vx | passam do *rated* | passam do *peak* |
|---|---|---|
| **0,5 m/s** | 2/23, 0,1% do tempo | **0/23** |
| **1,0 m/s** | 10/23 | **7/23** — joelho a **1,35×** |

No modelo T-N o torque disponível no peak speed é **zero**. A 1,0 m/s o joelho exige
momentaneamente velocidade onde o atuador real não entrega nada. Duty <1%, mas nos instantes de
maior dinâmica.

### Leituras do catálogo que se sustentam sozinhas

**Rated/peak speed são do eixo de saída, não do motor.** Com gear 18, 140 rpm de motor daria
0,81 rad/s no joelho, e nenhuma marcha cabe nisso; a medição dá 19,8 rad/s, mesma ordem da
leitura de saída (14,66).

**O asset foi construído a partir desta tabela.** O `armature` de `T1_23dof.xml` é exatamente
`inércia_de_rotor × gear²` em cinco dos sete grupos: Neck `18.0e-6·10² = 0.0018`, Braço
`21.8e-6·36² = 0.0282528`, Waist/HipRollYaw `76.5e-6·25² = 0.0478125`, Hip-Pitch
`161.7e-6·18² = 0.0523908`, Joelho `196.3e-6·18² = 0.0636012` — no último dígito. Os dois
restantes são tornozelo, cujo mecanismo paralelo divide um par de motores entre pitch e roll.
Isso também valida por tabela a derivação analítica de ganhos do `chute_t1`
(`kp = armature·(2πf)²`).

**O MJCF não modela atrito nem damping.** `frictionloss` e `damping`: zero ocorrências nas 23
juntas. Além da armadilha já documentada (sem damping passivo), não há **atrito seco de junta** —
que o htwk-gym randomiza em 0–2 Nm e nós não modelamos de forma alguma. Atrito seco é a
explicação mais comum para "andava em sim e ficou lento no robô".

**Ressalva:** a linha do Neck é anômala — rated 120 → peak 400 rpm é salto de 3,3× onde todo
outro grupo fica em 1,05–1,27×. Pode ser especificada de outra forma ou erro de digitação. Não
afeta locomoção.

## 4e. Degrau 0c — o envelope de atuador como especificado

Primeira execução da curva T-N. `--effort` escolhe o teto (`urdf` / `derated` / `catalog`) e
`--tn` liga a queda com a velocidade; os dois eixos são ortogonais, então a matriz separa o
efeito de cada um.

| modelo de atuador | vx = 0,5 | vx = 1,0 |
|---|---|---|
| `urdf` (o do treino) | 60 s ✓ · 96,8% | 60 s ✓ · 88,8% |
| `urdf --tn` | 60 s ✓ · 102,7% | 60 s ✓ · 88,4% |
| `derated` (firmware) | 60 s ✓ · 99,1% | **52,8 s** ✕ · 99,8% |
| **`derated --tn`** | 60 s ✓ · 96,3% | **9,6 s** ✕ · 100,8% |
| `catalog` (pico físico) | 60 s ✓ · 89,9% | 60 s ✓ · 87,8% |
| `catalog --tn` | 60 s ✓ · 98,7% | 60 s ✓ · 87,8% |

**O efeito é de interação, não de cada fator.** T-N com teto do URDF: 60 s. Teto derated sem
T-N: 52,8 s. Os dois juntos: **9,6 s**. A curva só morde quando o teto já está apertado — o que
faz sentido, porque ela reduz o torque disponível a partir do rated speed e, se a base já é 45 Nm
em vez de 55, acaba a margem.

**A 0,5 m/s nenhum modelo derruba.** Seis modelos, 60 s cada. O regime suave está dentro do
envelope físico em qualquer leitura.

### Qual destes é o robô?

Não dá para resolver com o que temos, e vale explicitar em vez de escolher:

- O firmware clipa nos derated → argumenta por `derated`.
- O atuador fisicamente não passa do pico do catálogo e tem queda com velocidade → argumenta
  por `catalog --tn`.
- Se os valores "derated" do `T1_23DOF_CFG` **já forem** uma aproximação plana conservadora do
  envelope (prática comum: um limite de operação contínua que já embute derating térmico e de
  velocidade), então aplicar os dois é **contagem dupla**.

As três leituras formam um intervalo a 1,0 m/s: **9,6 s** (pessimista) · **52,8 s** ·
**60 s** (otimista). Qual ponta vale é pergunta para a Booster, e é barata de fazer.

### Varredura de velocidade: onde está a fronteira

| vx | `urdf` (treino) | `catalog --tn` (físico) |
|---|---|---|
| 0,3 * | 60 s · 92,2% | 60 s · 1,4% (não anda) |
| 0,5 | 60 s · 96,8% | 60 s · 98,7% |
| 0,7 | 60 s · 96,9% | 60 s · 99,7% |
| 1,0 | 60 s · 88,8% | 60 s · 87,8% |
| 1,3 | 60 s · 86,6% | 60 s · 94,8% |
| **1,6** | 60 s · 101,7% | **24,9 s** ✕ |
| 2,0 | 60 s · 92,1% | **18,4 s** ✕ |
| 2,5 | 60 s · 92,4% | **4,6 s** ✕ |

\* 0,3 m/s está **abaixo do mínimo de treino** (Uniforme 0,5 · 2,5), então é extrapolação de
comando e não deve ser lido como dado de envelope.

Duas coisas:

1. **A fronteira do envelope físico fica entre 1,3 e 1,6 m/s.** Acima disso a política pede
   velocidade de junta onde o atuador não entrega torque, e cai progressivamente mais rápido.
2. **Sob o modelo do treino não existe fronteira.** `urdf` sobrevive 60 s até 2,5 m/s, o topo da
   faixa de comando. Era esse o modelo contra o qual toda validação anterior rodou — ele é
   ilimitadamente permissivo, e por isso não podia ter revelado nada disto.

Isso dá um limite operacional defensável para qualquer primeira tentativa em hardware:
**ficar em 0,5–1,0 m/s**, onde todos os modelos concordam.

## 4f. A matriz de observabilidade

Duas variáveis, dois níveis cada, mesmo checkpoint e mesmo comando:

| | teto URDF | teto derated |
|---|---|---|
| **estado verdadeiro** | 0,5: 60 s ✓ · 1,0: 60 s ✓ | 0,5: 60 s ✓ · 1,0: 52,8 s |
| **estado zerado** | 0,5: **1,10 s** · 1,0: **1,00 s** | 0,5: **1,07 s** · 1,0: **0,97 s** |

A linha de baixo é indiferente à coluna. A observabilidade domina por ~50× e o teto de torque
some dentro dela. Se houvesse uma única coisa a consertar, é a observação.

## 5. Degrau 1 — pior caso com zeros: queda em ~1 s, para frente

`--degrade zero_root`, 60 s de orçamento:

| | linha de base (`none`) | com zeros (`zero_root`) |
|---|---|---|
| **vx = 0,5** | não caiu em 60 s · rastreio 96,8% | **caiu em 1,10 s** |
| **vx = 1,0** | não caiu em 60 s · rastreio 88,8% | **caiu em 1,00 s** |

Os detalhes da queda confirmam a previsão de sub-passo:

| | vx = 0,5 | vx = 1,0 |
|---|---|---|
| deslocamento na direção comandada | **+0,819 m** | **+0,771 m** |
| deslocamento lateral | +0,003 m | −0,009 m |
| `grav_x` na queda | −0,899 | −0,890 |
| velocidade na 2ª metade / comando | **234,9%** | **122,2%** |
| altura final do tronco | 0,264 m | 0,345 m |

A queda é **para frente, na direção do movimento**, praticamente sem componente lateral. E
antes de cair a política **acelera acima do comando** — 235% a 0,5 m/s. É exatamente o
comportamento previsto: acreditando em velocidade zero, ela continua empurrando e posiciona o
pé como quem vai ficar parado, sub-passa, e tomba na direção do deslocamento.

O contraste de 60 s contra 1 s também estabelece que a lacuna de observabilidade, isolada, é
suficiente para derrubar. Não é um efeito de segunda ordem.

## 6. O que falta, e em que ordem

### Degrau 2 — construir o estimador

Módulo autônomo em `booster_deploy/utils/`, **sem dependência de MuJoCo nem de MimicKit**, para
rodar idêntico nos dois backends. Duas saídas, para um pé em contato:

```
root_h   = −(R · p_pé_na_base)_z + 0.03
root_vel = −R (ω × p_pé_na_base + J_pé(q) · q̇)
```

A terceira linha de `Rz(ψ)·M` é a terceira linha de `M`, logo `root_h` depende só de **roll e
pitch** — as componentes referenciadas à gravidade, que não derivam. E `root_vel` entra na
observação no frame do rumo, onde um erro de yaw rotaciona a velocidade e o frame de destino,
cancelando. **O estimador inteiro é invariante ao yaw.**

Detecção de contato sem sensor de pressão: `tau_est` de tornozelo e joelho com histerese e
tempo mínimo de permanência, cruzado com a heurística cinemática do pé mais baixo. O `acc`,
hoje não lido, atravessa fases de voo e denuncia escorregamento.

Cadeia cinemática: 6 juntas por perna, offsets do `T1_23dof.xml`.

### Degrau 3 — medir o erro contra a verdade

Com o módulo desacoplado, ele roda dentro do `MujocoController`, onde `qpos[2]` e `qvel[:3]`
são a verdade. Critério em σ da distribuição de treino do **checkpoint em uso**
(`t1_wrturn_s1`), que é a única escala em que o erro significa algo para a rede:

| Grandeza | σ de treino | Orçamento a 0,1σ |
|---|---|---|
| `root_h` | 0,12192 m | **≈ 1,2 cm** |
| `root_vel` (avanço) | 0,5515 m/s | **≈ 5,5 cm/s** |

Medir também **latência**, que nenhuma odometria evita e que o treino nunca viu.

### Degrau 4 — o veredito

Injetar o erro medido (magnitude, viés, ruído, latência) e comparar contra a linha de base
desta seção: 60 s sem queda, rastreio 96,8% a 0,5 e 88,8% a 1,0 m/s. O harness já suporta a
linha de base; falta o modo de degradação por modelo de erro.

### Diagnóstico no robô, quando houver robô

**(a) `/odom` está vivo e carrega twist?** Um `ros2 topic echo /odom --once`. Se vier
`twist.twist.linear`, as componentes horizontais de `root_vel` saem de graça.

**(b) A odometria continua atualizando em modo CUSTOM?** Decisiva. Em CUSTOM o controlador de
marcha onboard não dirige, e o estimador dele pode congelar. Entrar em CUSTOM, segurar a pose
de preparo, observar `rt/odometer_state` e `rt/odom` movendo o robô à mão.

**(c) Integração SDK/firmware.** Existência de `GetStatus`/`GetMode` no binding Python (decide
se a verificação comentada em `booster_robot_controller.py:326-333` pode voltar),
`len(motor_state_serial) == 23`, taxa de `/low_state`, deriva de yaw em 60 s, e identidade de
junta com humano no loop.

## 7. Correções independentes

Nenhuma depende do robô nem do estimador, e as três primeiras melhoram os próprios degraus.

1. **Guard de queda.** Portar `projected_gravity[2] > -0.5` de `locomotion.py:63-70` para
   `MimicKitSteeringPolicy.compute_observation`. `enable_safety_fallback` já existe e vale
   `True`, mas esta task nunca o lê. ~6 linhas.
2. **`tar_speed_min`.** Ver a nota na seção 2: o valor `0.0` não cria um modo parado. Ou volta
   para `0.5` e não se comanda parado, ou se aceita que `--vx 0.0` é extrapolação medida.
3. **Limitador de taxa em `dof_targets`.** A saída é alvo absoluto de posição, limitada só pelo
   clamp interno do export.
4. **`prepare_state` no `.replace()`** da task — hoje o hardware rampa até a pose de fábrica da
   Booster, uma terceira pose diferente da de sim e da de treino.
5. **`yaw₀` no `reset()`** — referenciar o mundo no start, para as 5 dims de steering não
   perseguirem a deriva da IMU.
6. **Logging de estado no caminho real.** `log_states` só existe no `MujocoController`
   (`mujoco_controller.py:170`). Sem ele, uma queda no robô é inatribuível.

## 8. Onde isso deixa a viabilidade

A matriz da seção 4c reordena tudo:

**O estimador voltou ao caminho crítico.** A observabilidade é a única variável que derruba em
1 s, e é a única com conserto em software. O degrau 0b mostrou que o teto de torque, isolado,
não impede a marcha — então construir o estimador deixou de ser "para a próxima política" e
voltou a ser o que pode destravar *este* checkpoint.

**O retreino por torque deixou de ser obrigatório e passou a ser desejável.** A política anda no
teto derated, com rastreamento melhor e envelope mais estreito a 1,0 m/s. Retreinar com
`--effort_source deploy` continua sendo o certo — mas agora é otimização de robustez, não
pré-requisito.

**A estimativa de viabilidade continua abaixo de 50%, por outros motivos.** Não mais por torque.
Por: (a) o erro e a latência do estimador, que ninguém mediu ainda — é o degrau 3; (b) a
estabilidade marginal que o README da task já documenta e que o degrau 0b reproduziu, com queda
aos 52,8 s a 1,0 m/s; (c) deriva de yaw, ainda não simulada.

**O próximo experimento que move a agulha é o degrau 3**, e ele é o único que exige escrever
código novo de verdade. A ordem ficou:

1. Correções independentes da seção 7 (pequenas, e melhoram os próprios testes).
2. Degrau 2 — o estimador, como módulo autônomo.
3. Degrau 3 — erro contra a verdade do MuJoCo, orçamento de 0,1σ.
4. Degrau 4 — injetar o erro medido e comparar contra a linha de base desta seção.
5. Diagnóstico no robô (`/odom`, CUSTOM, SDK) quando houver robô — pode encurtar o degrau 2.

## Lição metodológica

Vale registrar, porque custou uma conclusão errada: **medir que a demanda excede a oferta não é
medir que o sistema falha quando a oferta é limitada.** Os picos de 1,4× a 2,25× e o duty cycle
de 12% pareciam conclusivos e não eram — o laço PD satura e a marcha segue. A inferência a
partir da métrica indireta errou; o experimento direto, que custou uma flag e dois minutos,
acertou.

O mesmo padrão vale para o resto deste documento: toda vez que houver como trocar uma inferência
por uma medição, trocar.

## 9. Runlist de validação

Cada item valida uma afirmação específica deste documento. **Tudo roda de
`/home/jgabriel/Projects/Humanoids/booster_deploy`.** Os comandos foram executados nesta ordem;
`R1`–`R5` são pré-condições e `R6`–`R16` são as medições.

Abreviação usada abaixo:

```sh
export MK=../dev_tsinghua/MimicKit/mimickit     # MIMICKIT_PATH
export CK="$PWD/models/t1_wrturn_s1/policy.pt"  # checkpoint
```

### R1 — os caminhos do workspace (valida §2)

```sh
for p in ../booster_assets/src ../dev_tsinghua/booster_assets/src \
         ../tsinghua/MimicKit/data/assets/t1/t1.xml \
         ../dev_tsinghua/MimicKit/data/assets/t1/t1.xml \
         models/t1_wrturn_s1/policy.pt; do
  printf "%-52s %s\n" "$p" "$([ -e "$p" ] && echo EXISTE || echo AUSENTE)"
done
```

Esperado: `../booster_assets/src` **ausente**, `../dev_tsinghua/booster_assets/src` presente;
`../tsinghua/.../t1.xml` **ausente**, `../dev_tsinghua/.../t1.xml` presente. É por isso que o
comando antigo não roda aqui.

### R2 — o venv (valida §2 Setup)

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    torch mujoco scipy numpy gymnasium pyyaml matplotlib
uv pip install --python .venv/bin/python -e ../dev_tsinghua/booster_assets
uv pip install --python .venv/bin/python evdev   # só para o robô real
```

### R3 — estatísticas do checkpoint (valida §6 degrau 3 e a nota de `--vx 0.0`)

Não precisa do venv: só `zipfile` e `struct`. O normalizador de observação viaja dentro do
TorchScript, então as estatísticas de treino das 200 dimensões são legíveis do arquivo.

```sh
python3 - <<'EOF'
import zipfile, struct, math
z = zipfile.ZipFile('models/t1_wrturn_s1/policy.pt')
root = z.namelist()[0].split('/')[0]
n = len(z.read(root + '/data/0')) // 4
mean = struct.unpack('<%df' % n, z.read(root + '/data/0'))
std  = struct.unpack('<%df' % n, z.read(root + '/data/1'))
print(f"dims={n}  root_h media={mean[0]:.5f} desvio={std[0]:.5f}")
print(f"  z de 0.0 = {(0-mean[0])/std[0]:+.3f} sigma ; orcamento 0.1s = {0.1*std[0]*100:.2f} cm")
print(f"root_vel avanco desvio={std[7]:.4f} ; orcamento 0.1s = {0.1*std[7]*100:.2f} cm/s")
m, sd = mean[n-3], std[n-3]
print(f"tar_speed media={m:.4f} desvio={sd:.4f} -> Uniforme("
      f"{(m*2-sd*math.sqrt(12))/2:.3f}, {(m*2+sd*math.sqrt(12))/2:.3f})")
print(f"  z de 0.0 = {(0-m)/sd:+.3f} sigma   <- o que --vx 0.0 entrega")
print(f"root_rot idx2 desvio={std[2]:.6f}  (~0 => heading removido)")
EOF
```

Esperado: `root_h` **−4,916σ**, orçamento 1,22 cm; `root_vel` 5,52 cm/s; `tar_speed`
Uniforme(0,495 · 2,497) com `--vx 0.0` em **−2,589σ**; `root_rot idx2` desvio `0.000100`.

### R4 — smoke test sem viewer (valida §2)

```sh
MIMICKIT_PATH=$MK .venv/bin/python - <<'EOF'
import sys, os; sys.path.insert(0, '.')
from booster_assets import BOOSTER_ASSETS_DIR
print('BOOSTER_ASSETS_DIR :', BOOSTER_ASSETS_DIR)
print('T1_23dof.xml       :', os.path.exists(f'{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml'))
import tasks.mimickit_steering
from booster_deploy.utils.registry import get_task, list_tasks
print('tasks registradas  :', list(list_tasks().keys()))
from tasks.mimickit_steering.controllers import CONTROLLERS
cfg = get_task('t1_mimickit_steering')
cfg.policy.checkpoint_path = os.path.abspath('models/t1_wrturn_s1/policy.pt')
c = CONTROLLERS['explicit'](cfg); c.vel_command.lin_vel_x = 0.5
c.start(); c.update_state()
print('obs dims           :', c.policy.compute_observation().numel())
EOF
```

Esperado: `obs dims : 200`. Se falhar aqui é ambiente — não siga.

### R5 — GUI, linha de base (valida §3 Com GUI)

```sh
MIMICKIT_PATH=$MK DISPLAY=:1 .venv/bin/python \
    tasks/mimickit_steering/teleop.py --pd explicit --vx 0.5 --checkpoint "$CK"
```

O aviso `GLFWError ... Wayland` é inócuo. `--checkpoint` relativo resolve contra o **diretório
da task**, daí o `$PWD` em `$CK`.

### R6 — degrau 0, demanda de torque (valida §4)

```sh
for v in 0.5 1.0; do
  MIMICKIT_PATH=$MK .venv/bin/python scripts/sim_viability.py \
      --vx $v --duration 60 --out /tmp/viab/m_urdf_vx$v.npz
done
```

Esperado: **não cai** em 60 s nas duas; 11 juntas a 0,5 m/s e 12 a 1,0 m/s acima do derated;
`Right_Hip_Roll` em 2,25×.

### R7 — degrau 0b, teto do firmware (valida §4b)

```sh
for v in 0.5 1.0; do
  MIMICKIT_PATH=$MK .venv/bin/python scripts/sim_viability.py \
      --vx $v --effort derated --duration 60
done
```

Esperado: **anda**; rastreio 99,1% a 0,5 e 99,8% a 1,0; a 1,0 m/s cai em **52,8 s**.

### R8 — degrau 1, pior caso (valida §5)

```sh
for v in 0.5 1.0; do
  MIMICKIT_PATH=$MK .venv/bin/python scripts/sim_viability.py \
      --vx $v --degrade zero_root --duration 60
done
```

Esperado: cai em **1,10 s** e **1,00 s**; deslocamento **+0,819 / +0,771 m na direção
comandada** com lateral quase nulo; velocidade a 235% / 122% do comando antes de cair.

Com GUI, que é onde o sub-passo se vê (`Backspace` repete, `Space` + setas andam quadro a
quadro):

```sh
MIMICKIT_PATH=$MK DISPLAY=:1 .venv/bin/python \
    tasks/mimickit_steering/teleop.py --pd explicit --vx 0.5 \
    --degrade zero_root --checkpoint "$CK"
```

### R9 — combinado (valida §4c)

```sh
for v in 0.5 1.0; do
  MIMICKIT_PATH=$MK .venv/bin/python scripts/sim_viability.py \
      --vx $v --degrade zero_root --effort derated --duration 60
done
```

Esperado: **1,07 s** e **0,97 s** — indistinguível de R8. A observabilidade domina.

### R10 — o catálogo do fabricante vs. o código (valida §4d)

```sh
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0,'.')
from booster_deploy.robots.booster import (
    T1_23DOF_CFG, T1_CATALOG_PEAK_TORQUE,
    T1_CATALOG_VELOCITY_LIMIT, T1_CATALOG_KNEE_POINT_VELOCITY)
from tasks.mimickit_steering.mimickit_steering import T1_EFFORT_URDF
for i, n in enumerate(T1_23DOF_CFG.joint_names):
    pk = T1_CATALOG_PEAK_TORQUE[i]
    print(f"{n:<22}{T1_23DOF_CFG.effort_limit[i]:>7.1f}{T1_EFFORT_URDF[i]:>8.1f}"
          f"{pk:>8.1f}{T1_EFFORT_URDF[i]/pk:>8.2f}x{T1_23DOF_CFG.effort_limit[i]/pk:>8.2f}x")
print('vel limit  :', [round(v,2) for v in T1_CATALOG_VELOCITY_LIMIT[10:17]])
print('knee point :', [round(v,2) for v in T1_CATALOG_KNEE_POINT_VELOCITY[10:17]])
print('defaults intactos: velocity_limit =', T1_23DOF_CFG.velocity_limit)
EOF
```

Esperado: joelho URDF 130,5 vs pico 65 → **2,01×**; hip-pitch 1,80×; waist/hip-roll/yaw 1,70×.
E `velocity_limit = None` no cfg — o catálogo não muda default nenhum.

### R11 — `armature` do MJCF = inércia de rotor × gear² (valida §4d)

```sh
X=$(.venv/bin/python -c "from booster_assets import BOOSTER_ASSETS_DIR;print(BOOSTER_ASSETS_DIR)")/robots/T1/T1_23dof.xml
grep -o 'armature="[^"]*"' $X | sort -u
.venv/bin/python -c "
for nm,J,g in (('Neck',18.0,10),('Arm',21.8,36),('Waist/HipRollYaw',76.5,25),
               ('HipPitch',161.7,18),('Knee',196.3,18)):
    print(f'{nm:<18}{J}e-6 * {g}**2 = {J*1e-6*g*g:.7f}')"
```

Esperado: cinco valores do MJCF batem no último dígito — `0.0018`, `0.0282528`, `0.0478125`,
`0.0523908`, `0.0636012`.

### R12 — o que o MJCF não modela (valida §4d)

```sh
for k in frictionloss damping armature; do echo "$k: $(grep -c $k $X)"; done
```

Esperado: `frictionloss: 0`, `damping: 0`, `armature: 23`. Sem atrito seco e sem damping
passivo — o segundo é a armadilha 1 do [05](05-armadilhas-conhecidas.md); o primeiro é algo que
o htwk-gym randomiza em 0–2 Nm e nós não modelamos.

### R13 — velocidade de junta vs. catálogo (valida §4d)

Sai no mesmo relatório de R6/R7, na seção `VELOCIDADE DE JUNTA`. Esperado: a 0,5 m/s
**0/23** juntas passam do peak speed; a 1,0 m/s **7/23** passam, joelho a **1,35×**.

### R14 — a curva T-N, pela primeira vez (valida §4e)

```sh
for v in 0.5 1.0; do
  MIMICKIT_PATH=$MK .venv/bin/python scripts/sim_viability.py \
      --vx $v --effort catalog --tn --duration 60
done
```

`--effort catalog` usa o pico do catálogo como teto; `--tn` liga a queda de torque com a
velocidade (rated → `knee_point_velocity`, peak → `velocity_limit`). É o primeiro teste do
envelope de atuador **como especificado** em vez de estimado.

Com GUI:

```sh
MIMICKIT_PATH=$MK DISPLAY=:1 .venv/bin/python \
    tasks/mimickit_steering/teleop.py --pd explicit --vx 0.5 \
    --effort catalog --tn --checkpoint "$CK"
```

### R15 — varredura completa (valida §4e e as figuras)

```sh
bash scripts/sweep_viability.sh        # ~20 min, 32 rodadas -> /tmp/viab
```

### R16 — figuras

```sh
.venv/bin/python scripts/plot_viability.py --in /tmp/viab --out docs/diagramas
```

Análise das saídas em [09 — Análise gráfica](09-analise-graficos.md).

## Ver também

- [07 — Método comparado: htwk-gym](07-htwk-gym-metodo-comparado.md): como evitar esta lacuna
  por construção no retreino.
- [05 — Armadilhas conhecidas](05-armadilhas-conhecidas.md): itens 2 e 4.
- [06 — Runbook de deploy](06-runbook-deploy-real.md): o procedimento em que qualquer política
  nova vai entrar.
