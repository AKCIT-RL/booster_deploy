# 11. Reavaliação: o release upstream de 2026-09-13

> Branch `tsinghua-v2`, sobre `BoosterRobotics/booster_deploy@7bb1462e`.
> O baseline anterior está congelado na branch `tsinghua` e na tag `baseline-docs-09`
> (commit `c463403`) — as figuras de [09](09-analise-graficos.md) continuam reproduzíveis lá.

Os documentos [01](01-arquitetura-comunicacao.md)–[10](10-discussao-proximos-passos.md) foram
escritos contra um fork de `563e7c3` (2025-12-25). Em 2026-09-13 o upstream publicou **um único
commit**, `7bb1462e` (+2199/−823, 46 arquivos): locomotion e dance para o T2, locomotion para o
K1, dance para o T1, suporte a ONNX, e uma reorganização de `robots/` e `tasks/`.

Este documento registra o que o release mudou no estudo. A conclusão curta: **a tese central
sobrevive intacta e ganha suporte**, mas duas premissas de instrumentação precisaram ser
refeitas, e uma atribuição estava errada desde o começo.

## 1. O que o release não mudou (e por que isso importa)

A conclusão de fundo de [08](08-caminho-para-hardware-mimickit.md) e
[09](09-analise-graficos.md) é que **a lacuna de observabilidade domina** — zerar quatro números
derruba o robô em ~1 s, contra dezenas de segundos de diferença entre modelos de atuador.

Essa lacuna **continua exatamente onde estava**:

```python
# booster_deploy/controllers/booster_robot_controller.py:231-234 (no release novo)
self._state_buf[0]["root_pos_w"][:] = np.zeros(...)
self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(...)
```

Continua não havendo odometria no caminho de estado do robô real. A Fig. 3 mede uma coisa que o
release não tocou, e `t1_mimickit_steering` segue sendo sim-only pelo mesmo motivo de antes.

Também seguem válidas as medições de [08](08-caminho-para-hardware-mimickit.md) e
[09](09-analise-graficos.md): **a ordem das juntas do T1 não mudou** (só os nomes viraram
`snake_case` com sufixo `_joint`), e o `MujocoController` indexa `qpos[7:]`/`qvel[6:]` por
posição, não por nome. Os `armature` do `T1_23dof.xml` são os mesmos.

## 2. O que o release mudou no estudo

### 2a. `effort_limit` trocou de significado — o risco mais grave

O release repontou `T1_23DOF_CFG.effort_limit` para os **valores do URDF**
(`7.0 / 38.3 / 68.0 / 98.8 / 130.5 / 73.1 / 17.2`), no lugar dos limites de firmware
(`7 / 18 / 25 / 45 / 60 / 24 / 15`).

O instrumental lia essa constante como a linha de base **`derated`**. Na base nova, `derated`
viraria idêntico a `urdf`: a Fig. 5 continuaria rodando, continuaria plotando, e estaria
comparando um modelo contra ele mesmo — **sem levantar erro algum**.

Correção: os três tetos agora são constantes nomeadas em
`booster_deploy/robots/t1_actuators.py` (`T1_EFFORT_FIRMWARE`, `T1_EFFORT_URDF`,
`T1_CATALOG_PEAK_TORQUE`), com uma asserção em tempo de import que falha se os dois primeiros
coincidirem.

Vale registrar o que o próprio release diz sobre a questão Q1 de
[10](10-discussao-proximos-passos.md): **a Booster promoveu os limites do URDF a configuração
oficial.** Isso enfraquece a leitura de que o firmware clipa nos derated e empurra a
interpretação para o lado `catalog --tn`.

### 2b. A armadilha 1 era do fork, não do framework

[05](05-armadilhas-conhecidas.md) atribuía ao `MujocoController` o `kd = 0`. É do fork: veio do
commit `d232976` deste repositório. O upstream sempre usou `kd = self.robot.joint_damping`, e
continua usando no release.

O `kd = 0` foi mantido em `tsinghua-v2` — é o análogo de hardware contra o qual
[09](09-analise-graficos.md) mede — mas a justificativa registrada no código estava errada: ela
dizia que o amortecimento chegaria pelo XML, e o `T1_23dof.xml` **não declara `dof_damping`
nenhum**. Com `kd = 0`, o T1 roda **sem amortecimento algum** na simulação.

### 2c. O binding Python do SDK deixou de existir

A comunicação com o robô real passou a ser inteiramente ROS 2: `booster_interface.msg` para
estado e comando, e o serviço `booster_rpc_service` (`booster_interface.srv.RpcService`, com
códigos numéricos `_LOC_API_CHANGE_MODE = 2000` e `_LOC_API_GET_STATUS = 2018`) para troca de
modo. Não há mais nenhuma referência a `booster_robotics_sdk_python` no repositório.

Efeito colateral relevante: a verificação de modo que estava **comentada** (a armadilha 9) agora
está implementada e ativa — `_change_robot_mode` confirma via `GET_STATUS` que o
`current_mode` bate com o pedido, com até 20 tentativas.

### 2d. A saída deixou de ser fixa em `kWalking`

`BoosterRobotControllerCfg.exit_mode` (`"walking"` ou `"damping"`) decide o modo pós-execução,
com `--exit-mode` como override de linha de comando, e um abort de segurança que força
`damping`. Ver [06](06-runbook-deploy-real.md#como-abortar).

## 3. A anomalia do tornozelo, resolvida

[08 §4d](08-caminho-para-hardware-mimickit.md) deixou em aberto dois `armature` do
`T1_23dof.xml` que não fechavam em `inércia × gear²`. O release os explica, e por um caminho
inesperado: o README documenta a fórmula de Kd para juntas paralelas, e os ganhos publicados em
`robots/t1.py` a satisfazem exatamente.

Todo o `joint_stiffness`/`joint_damping` do T1 sai de **dois números globais** —
**f_n = 4,0 Hz e zeta = 1,5**. E as duas armaduras de tornozelo são múltiplos exatos da armadura
de um motor só (`26,2e-6 × 36² = 0,0339552`): **2,0×** no pitch (dois motores somando) e
**0,6×** no roll (alavanca diferencial). Os literais `/2` e `/0.6` que o upstream escreve no
amortecimento de tornozelo desfazem esses fatores, e os dois tornozelos caem no mesmo Kd de lado
motor, `2,5601617649`.

A derivação completa, o que isso significa no deploy e como conferir estão em
[06 — Kd de junta paralela vs. Kd de motor](06-runbook-deploy-real.md#kd-de-junta-paralela-vs-kd-de-motor).

**Fica em aberto:** o teto de torque não escala como a armadura (`73,1/17,2 = 4,25` contra
`2/0,6 = 3,33`), então isso ainda não explica o `17,2` do Ankle-Roll que
[09](09-analise-graficos.md) registra como `0,34×` do pico de catálogo.

## 3b. A runlist reproduzida

A runlist R15–R16 foi re-executada inteira sobre a base nova (32 rodadas, nenhuma falha), com as
figuras em [`diagramas/v2/`](diagramas/v2/) ao lado das originais.

```sh
VIAB_OUT=/tmp/viab-v2 bash scripts/sweep_viability.sh
.venv/bin/python scripts/plot_viability.py --in /tmp/viab-v2 --out docs/diagramas/v2
```

**Os números reproduzem o baseline exatamente.** Sobrevivência a 1,0 m/s por modelo de atuador,
contra o publicado em [09](09-analise-graficos.md):

| Modelo | v2 | docs/09 |
|---|---:|---:|
| `urdf` | 60,00 s | 60 s |
| `urdf --tn` | 60,00 s | 60 s |
| `derated` | **52,80 s** | 52,8 s |
| `derated --tn` | **9,60 s** | 9,6 s |
| `catalog` | 60,00 s | 60 s |
| `catalog --tn` | 60,00 s | 60 s |

A varredura de velocidade também: penhasco entre **1,3 e 1,6 m/s**, com 24,87 s / 18,37 s /
4,60 s a 1,6 / 2,0 / 2,5 (docs/09: 24,9 / 18,4 / 4,6), e a série `urdf` plana em 60 s por toda a
faixa. A Fig. 3 segue derrubando o robô em **~1 s** com as quatro dimensões zeradas, contra 60 s
com estado verdadeiro.

Isto é o resultado mais útil da reavaliação: o release **não move nenhum dos números**. As
conclusões de [08](08-caminho-para-hardware-mimickit.md) e [09](09-analise-graficos.md) valem tal
como publicadas — o limite operacional defensável continua **0,5–1,0 m/s**.

Vale dizer o que a reprodução **não** cobre: ela roda com os ganhos do fork e o modelo serial do
tornozelo. Os dois eixos novos abertos pelo release — os ganhos de f_n = 4 Hz / zeta = 1,5 e o
`T1_23dof_parallel.xml` — ainda não foram varridos.

## 4. Veredito das dez armadilhas

Cada item de [05](05-armadilhas-conhecidas.md), verificado contra o código do release.

| # | Armadilha | Veredito |
|---|---|---|
| 1 | `joint_damping` sem efeito no MuJoCo | **Era local do fork.** Upstream usa `kd = joint_damping`; o `kd = 0` veio de `d232976`. Mantido de propósito, com a justificativa corrigida — ver §2b |
| 2 | Sem odometria no robô real | **Persiste.** `root_pos_w`/`root_lin_vel_w` seguem zerados (`booster_robot_controller.py:231-234`) |
| 3 | Referencial de velocidade linear diverge | **Persiste.** A conversão continua explícita só no caminho real |
| 4 | `action_delay_steps` inerte | **Persiste, e é do fork.** O buffer é criado em `run()` e nunca lido no `ctrl_step`. O campo nem existe no upstream |
| 5 | `chute_t1` não inicia | **Persiste.** Segue apontando para `models/paper/` e `motions/paper/`, ausentes e no `.gitignore` |
| 6 | `T1_23DOF_CFG.sim_body_names` vazio | **Corrigida.** Agora populado em `robots/t1.py:108` |
| 7 | `enable_velocity_commands` sem efeito | **Corrigida.** O campo sumiu junto com o código que o continha |
| 8 | `fastdds_profile.xml` ausente | **Corrigida.** A menção saiu do README, e o caminho DDS foi substituído por ROS 2 |
| 9 | Binding Python do SDK defasado | **Resolvida por eliminação.** Não há mais binding Python; a verificação de modo antes comentada está ativa — ver §2c |
| 10 | `EvaluatorCfg` é andaime sem uso | **Persiste.** `registry.py` ainda expõe `register_evaluator`/`get_evaluator`, e nenhuma tarefa os popula |

Três persistem por design, duas são dívida do próprio fork (1 e 4), quatro foram corrigidas, e
uma sumiu com a arquitetura que a causava.

## 5. Documentos afetados

| Doc | Estado |
|---|---|
| [01](01-arquitetura-comunicacao.md), [02](02-modos-e-ciclo-de-vida.md) | **Precisam de revisão maior** — descrevem o caminho DDS/SDK que foi substituído por ROS 2 |
| [03](03-ganhos-pd-sim-vs-real.md) | Emendado: os ganhos do release são um projeto de segunda ordem a f_n = 4 Hz, zeta = 1,5 |
| [04](04-mapa-do-framework.md) | Emendado: `--webots` e `--net` não existem mais |
| [05](05-armadilhas-conhecidas.md) | Emendado com os vereditos de §4 |
| [06](06-runbook-deploy-real.md) | Seção nova de Kd; pré-requisitos, checklist e aborto atualizados |
| [07](07-htwk-gym-metodo-comparado.md) | Sem mudança — a comparação com htwk-gym não depende do release |
| [08](08-caminho-para-hardware-mimickit.md) | §4d marcado como resolvido; números permanecem válidos |
| [09](09-analise-graficos.md) | Nota sobre a origem da coluna `firmware` |
| [10](10-discussao-proximos-passos.md) | Q1 ganhou evidência nova — ver §2a |
| [12](12-politica-vendor-t1-walk.md) | **Novo.** A `t1_walk` da Booster como controle do estudo |

## Ver também

- [06 — Runbook](06-runbook-deploy-real.md): a derivação do Kd de junta paralela.
- [09 — Análise gráfica](09-analise-graficos.md): as figuras do baseline, ainda reproduzíveis
  em `baseline-docs-09`.
