# 11. Dependências no robô, preflight e a pose de preparo

> Este documento registra as correções feitas após a **primeira tentativa real de deploy da
> `t1_mimickit_steering` no T1** (2026-09-15) e o método para não repetir a classe de falha
> que ela expôs. Complementa o [06 — Runbook](06-runbook-deploy-real.md), que descreve o
> procedimento; aqui está o que validar **antes** de chegar nele.

## 1. O que aconteceu

A tentativa falhou assim:

```
[INFO] Custom mode started, initialized with prepare pose
[INFO] Inference process started
ModuleNotFoundError: No module named 'gymnasium'
[ERROR] Inference process died unexpectedly
```

Uma dependência Python ausente. O detalhe que importa não é *qual* dependência, é **quando**
ela se manifestou: depois de `ChangeMode(kCustom)` e depois da rampa de 500 passos até a pose
de preparo. Ou seja, **o robô já tinha se movido e estava de pé sob os ganhos de preparo**
quando o processo de inferência morreu.

Isso é estrutural, não azar. `scripts/deploy.py` constrói a policy dentro do processo de
inferência, que é forkado no final de `start_rl_gait_conditionally()`
(`booster_robot_controller.py:435-457`). Todo `import` da task — inclusive a cadeia do
MimicKit, que `observation.py:48-66` adia deliberadamente para o primeiro uso — acontece
tarde. Qualquer erro de ambiente, por mais trivial, só aparece com o robô energizado e em pé.

E `deploy.py --list` não protege: `tasks/__init__` engole exceções de import por design, para
que uma task quebrada não derrube o registro das outras. A task aparece na lista e falha no
deploy.

## 2. As três armadilhas de instalação

Além da dependência ausente, a sessão passou perto de três erros piores. Registrados porque
todos os três são intuitivos e todos os três quebram o robô.

### (a) O `requirements.txt` do repositório de treino

```
$ pip install -r requirements.txt        # em ~/tsinghua
ERROR: Could not find a version that satisfies the requirement numpy==2.5.0
```

Falhou na resolução, então **não instalou nada** — sorte. Aquele arquivo pina `numpy==2.5.0`
(exige Python 3.12) e traz `diffusers`, `wandb`, `moviepy`, `pyglet`. Se tivesse resolvido,
teria subido o numpy do sistema para 2.x.

A cadeia de import que a inferência realmente usa é
`anim.mjcf_char_model`, `envs.char_env`, `envs.task_steering_env`, `util.torch_util`, e ela
toca **quatro** pacotes de terceiros: `torch`, `numpy`, `gymnasium`, `pyyaml`. Verificado
carregando os módulos e inspecionando `sys.modules`, não lendo o requirements do MimicKit.

### (b) Criar um venv com outro Python

```
$ python3.11 -m venv env_deploy
Command 'python3.11' not found
```

O impulso é natural — "o problema é a versão do Python" — e está invertido. `rclpy`,
`booster_interface.msg` e `booster_robotics_sdk_python` são extensões C compiladas contra o
CPython 3.10 do sistema (ROS 2 Humble e o build do SDK). Nenhum outro interpretador as
carrega, e **não existe wheel no PyPI para reinstalá-las**. O 3.10 não é uma limitação a
contornar; é o requisito.

Se quiser isolamento, a única forma que funciona herda o site-packages do sistema:

```bash
python3 -m venv --system-site-packages env_deploy
```

### (c) Atualizar o numpy do sistema para 2.x

O `rclpy` do Humble é compilado contra numpy 1.x (o robô traz 1.21.5 em
`/usr/lib/python3/dist-packages`). Subir para 2.x quebra o ABI e derruba a assinatura de
`/low_state` — o deploy inteiro, não só esta task.

**Nenhum dos três é detectável depois.** Cada um se manifesta como um sintoma que aponta para
outro lugar.

## 3. A solução: `requirements-robot.txt` + `scripts/preflight.py`

**`requirements-robot.txt`** lista só o runtime do robô, separa o que já vem na imagem (e não
deve ser reinstalado) do que falta, e carrega as três armadilhas acima como comentários no
ponto onde alguém iria digitar o comando errado.

**`scripts/preflight.py`** importa exatamente o que o deploy importa, na ordem em que o deploy
importa, **sem mover o robô**:

```bash
python3 scripts/preflight.py --task t1_mimickit_steering
python3 scripts/preflight.py --task t1_mimickit_steering --mujoco
```

Verifica versão do Python, isolamento do venv, deps de núcleo, ABI do numpy contra o rclpy, a
pilha ROS/SDK, o registro da task (**relatando a exceção de import que o `--list` engole**), a
cadeia do MimicKit, o asset, o checkpoint (carrega de fato, via `torch.jit.load`), a pose e a
rigidez de preparo, e o estado dos guards. Saída 0/1, para entrar em script.

Quando uma dependência do MimicKit falta, o diagnóstico já traz a correção **e** o aviso de não
instalar o requirements de treino — que é o erro seguinte na sequência natural.

Isto é um passo obrigatório do runbook, antes de energizar.

## 4. A pose de preparo (`prepare_state`)

Achado separado, encontrado ao revisar o que aconteceria se aquele run **tivesse** chegado à
inferência. A `t1_mimickit_steering` não sobrescrevia `prepare_state`, herdando o de
`T1_23DOF_CFG` (`booster.py:341`), que é o de Booster, para a política de Booster. Isso punha
um degrau nos **dois** eixos ao mesmo tempo, no instante da entrega do controle:

| | rampa de preparo (herdada) | o que a política assume |
|---|---|---|
| pose das pernas | `[-0.1, 0, 0, 0.2, -0.1, 0]` | `[-0.2, 0, 0, 0.4, -0.2, 0]` |
| `kp` do joelho | 350 | 80 |

A política de steering emite **alvo absoluto de posição** — ela não corrige a partir da pose em
que recebe o robô, ela comanda a sua. Então o primeiro passo seria um degrau até a pose de
treino, com o robô já de pé.

A primeira correção tratou pose **e ganhos** como o mesmo problema e rampou até a pose de
treino sob os ganhos da própria política, argumentando que assim "nada muda no handover". **Isso
estava errado, e derrubou o robô no teste seguinte** — ver §4a.

A correção correta substitui **só a pose**, herdando `stiffness`/`damping` de
`T1_23DOF_CFG.prepare_state`. `preflight.py` checa as duas coisas, em direções opostas: a pose
tem de bater com a da policy, e os ganhos de preparo têm de ser **rígidos o bastante** (kp ≥ 200
no quadril/joelho), não iguais aos dela.

## 4a. Correção da correção: por que os ganhos de preparo não são os da política

No teste no robô, ao entrar em modo CUSTOM o T1 **afundou lentamente até o chão**, sem parecer
tentar sustentar posição alguma. O diagnóstico inicial deste documento — "se ceder na rampa,
`kp=80` não sustenta o T1 e a política nunca teria funcionado" — estava certo na primeira
metade e **errado na segunda**.

Medido: segurando estaticamente a pose de treino por 3 s, com o teto de torque derated.

| ganho de perna (quadril/joelho) | altura final da base |
|---|---|
| 350 — preparo da Booster | 0,678 m |
| 200 — regime da `locomotion` | 0,676 m |
| **80 — o desta política** | **0,025 m (no chão)** |

Isolando: o tornozelo **não** é a causa — enrijecê-lo sozinho não muda nada. É o ganho de
quadril/joelho. E, revelador: durante todo o colapso **as juntas seguiram seus alvos com erro
de ~1°**. O robô não perdeu rastreamento de junta; ele perdeu equilíbrio. É exatamente a
descrição do operador: "como se o torque fosse baixo e não estivesse tentando manter posição".

**Por que isso não condena a política.** Segurar uma pose e rodar uma política são problemas de
controle diferentes. Um PD sobre alvo fixo carrega o pêndulo invertido inteiro na rigidez das
juntas; a política re-planeja o alvo 30 vezes por segundo e equilibra **ativamente** — e é por
isso que `kp=80` anda perfeitamente em MuJoCo. Não há contradição entre "afunda parada a
kp=80" e "anda a kp=80".

A divisão da Booster — **rígido para ficar de pé, mole para rodar** — é correta, não um
descuido a ser suavizado. A `tasks/locomotion`, que funciona no hardware, faz exatamente o
mesmo: ganhos próprios de 200/50 e rampa sob os 350 herdados.

O degrau de ganho no handover (350 → 80) portanto é **real e fica**. É o mesmo degrau com que a
`locomotion` convive, um fator maior. Ele não é removível por configuração de preparo — tentar
removê-lo por baixo só derruba o robô mais cedo.

## 5. Guard de queda e limitador de taxa

**Guard de queda.** `PolicyCfg.enable_safety_fallback` sempre existiu e sempre valeu `True`;
esta task nunca o lia — uma chave de segurança que se lê como armada e não está ligada a nada.
Portado de `locomotion.py:63-70`: `projected_gravity[2] > fall_gravity_z` (padrão `-0.5`,
o mesmo limiar que `sim_viability.py` já usa para pontuar quedas, de modo que o número da
simulação e o comportamento do robô falem da mesma definição). Lê só a IMU, então vale igual
no hardware e no MuJoCo — o que importa aqui, porque a *observação* não vale: as duas
grandezas de raiz que faltam no hardware também faltariam para qualquer guard, e a atitude do
tronco é o único sinal de queda honesto que o robô tem. É detector, não preventor.

**Limitador de taxa — e o que a medição revelou.** O clip interno do export limita o **valor**
do alvo aos limites de junta do asset; nada limitava a **derivada**. Um salto de curso inteiro
em um passo, através de um PD rígido, é um pico de torque limitado só por `effort_limit` — que
esta task eleva deliberadamente aos máximos do URDF (joelho 130,5 Nm contra os 60 derated do
firmware).

O valor inicial escolhido foi 10 rad/s, com o raciocínio de que seria "folgado para a marcha
nominal". **A medição desmentiu isso**, e o resultado mudou o projeto. Demanda real da
política, rollouts de 30 s com o limitador desligado, pior junta a cada passo:

| `vx` | p50 | p95 | p99 | max |
|---|---|---|---|---|
| 0,5 | 14,3 | 30,8 | 45,5 | 61,1 |
| 1,0 | 14,9 | 29,0 | 37,7 | 45,7 |
| 2,0 | 30,7 | 58,5 | 71,1 | 93,3 |
| 2,5 | 28,7 | 63,4 | 80,4 | 91,7 |

(rad/s; a 30 Hz, 30 rad/s = 1 rad por passo)

A marcha nominal desliza o alvo a **dezenas de rad/s**: o passo **mediano** move a pior junta
em ~0,5 rad (29°). Um alvo de posição tão ativo não deixa vão entre o nominal e o patológico
para um limite de taxa ocupar — o salto de curso inteiro, que é a coisa que valeria a pena
pegar, dá ~75 rad/s, e a marcha já chega a 93.

Efeito de apertar, medido em `vx=1,0` por 30 s:

| `max_target_rate` | passos clipados | rastreamento |
|---|---|---|
| `None` | — | 88,3% |
| 200 / 120 / 60 | 0% | 88,3% (idêntico) |
| 30 | 3% | 86,9% |
| 10 | **82%** | 87,8% |

O default de 10 rad/s clipava **82% de todos os passos**. E note que ele **não falhou**: o robô
andou os 30 s e o rastreamento caiu meros 0,5 ponto. Esse é justamente o perigo — um limite
apertado demais não quebra ruidosamente, ele reescreve em silêncio o laço fechado em que o
checkpoint foi validado, e o run continua parecendo bem-sucedido.

O default passou para **120 rad/s**: acima do envelope medido em toda a faixa comandada (0% de
clipping em qualquer velocidade), pegando só saída completamente fora de família — checkpoint
corrompido, por exemplo. É um **teto de sanidade, não um guard de marcha**, e a doc do campo
diz isso para que ninguém confie nele além disso.

**Bug de precisão encontrado no caminho.** A primeira versão calculava
`prev + clamp(target - prev)` sempre, inclusive sem clipping. Isso é igual a `target` em
aritmética exata e não em ponto flutuante, e o resíduo de ~1e-7 não é inócuo: marcha é caótica,
e num rollout de 30 s ele virou 2 pontos de diferença no rastreamento (0,883 vs 0,905) com
**zero** passos clipados. Corrigido retornando o tensor de entrada quando não há clipping —
um guard desligado precisa ser bit-exato desligado, ou toda medição que ele encosta deriva por
um motivo que ninguém vai achar.

**A lição real é sobre a política, não sobre o limite.** O export do MimicKit não tem termo de
suavidade de ação, então o chatter do alvo está assado no checkpoint. Limitar o transiente de
atuador de verdade significa retreinar com um termo desses — não filtrar aqui. Isso entra na
pauta de retreinamento da [10](10-discussao-proximos-passos.md), ao lado de
`--effort_source deploy`.

## 6. O que isto NÃO resolve

Nada aqui toca a lacuna de observabilidade. No hardware, `booster_robot_controller.update_state`
preenche `root_pos_w` e `root_lin_vel_b` com `np.zeros(3)` (`:244-249`), porque o T1 não tem
sensor para nenhum dos dois. São 4 das 200 dimensões, e `compute_char_obs`
(`envs/char_env.py:412-446`) as usa assim:

- `root_h = root_pos[:, 2:3]` — literalmente a altura, entregue como **0** onde o treino viu
  ~0,69 m. Com σ de treino de 0,12192 m ([08 §6](08-caminho-para-hardware-mimickit.md)), é um
  erro constante de **~5,7σ**. Não é um problema de orçamento de ruído; está fora da
  distribuição.
- `root_vel` — três zeros constantes, enquanto `task_obs` comanda `tar_speed` de até 2,5 m/s.
  A política vê, todo frame, "fui mandada andar e estou perfeitamente parada".

As outras 196 dimensões são invariantes à posição da raiz (`key_pos` é relativo: o
`forward_kinematics` usa `root_pos` e `compute_char_obs` o subtrai de volta, então cancela), o
que confirma a contagem de `controllers.py:130-140`.

Portanto, o estado após estas correções:

| | antes | agora |
|---|---|---|
| falha de dependência | com o robô de pé | em terra, no preflight |
| pose no handover | degrau (meia agachada) | contínua |
| ganho no handover | 350 → 80 | 350 → 80, inerente (§4a) |
| queda | sem detecção | parada controlada |
| transiente de atuador | sem teto | teto de sanidade a 120 rad/s (§5) |
| **`root_h` / `root_vel`** | **zeros** | **zeros** |

As quatro primeiras linhas tornam um teste **sobrevivível e atribuível**. A última é a que
decide se a task **funciona**, e continua aberta — ver
[08 §8](08-caminho-para-hardware-mimickit.md), que estima viabilidade abaixo de 50% por causa
dela.

## 7. Ordem recomendada a partir daqui

1. **`sim_viability.py --degrade zero_root`** — reproduz exatamente a condição do hardware
   (`controllers.py:148-151` zera os mesmos dois campos) dentro do MuJoCo, com custo zero e
   risco zero. Se cai lá, cai no robô. Este passo ainda não foi executado, e é o que deveria
   preceder qualquer teste físico.
2. **Logging no caminho real.** `log_states` só existe no `MujocoController`
   (`mujoco_controller.py:170`). Sem ele, uma queda no robô é inatribuível — arrisca-se o
   hardware e não se aprende nada. Item 6 da [08 §7](08-caminho-para-hardware-mimickit.md).
3. **Teste suspenso no pórtico**, com kill switch ao alcance, e só então.
4. **Chão** só depois da odometria de pé de apoio ([08 §6](08-caminho-para-hardware-mimickit.md)).

Os itens 2, 3 e 5 da [08 §7](08-caminho-para-hardware-mimickit.md) (`tar_speed_min`, `yaw₀` no
`reset()`, logging real) continuam abertos.
