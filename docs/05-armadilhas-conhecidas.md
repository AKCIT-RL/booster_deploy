# 5. Armadilhas conhecidas

> Referência: commit `0dba32b` (branch `tsinghua`). Cada item foi verificado diretamente no
> código — nenhum é especulação.

## 1. `joint_damping` não tem efeito na simulação MuJoCo do T1

**Sintoma:** mudar `RobotCfg.joint_damping` para o T1 não altera em nada o comportamento em
`--mujoco`.

**Causa:** `MujocoController.ctrl_step()` força `kd = 0` no cálculo explícito do PD,
esperando que o amortecimento venha do próprio modelo MJCF via
`mj_model.dof_damping` (comportamento tipo `ImplicitActuator` do IsaacLab):

```python
# booster_deploy/controllers/mujoco_controller.py:213-217
kd = np.zeros_like(kp)
```

O modelo `T1_23dof.xml` **não declara** damping por junta. Resultado: para o T1,
`joint_damping` afeta **só o robô real** — na simulação, o efeito é nulo, a menos que se use
explicitamente um dos controladores alternativos do `mimickit_steering`
(`ExplicitPDController`, que aplica `kd` de fato — ver [03](03-ganhos-pd-sim-vs-real.md)).

**Como lidar:** ao investigar divergência sim2real relacionada a amortecimento no T1, não
assuma que `joint_damping` está "fazendo algo" do lado da simulação padrão. Use os
controladores de `tasks/mimickit_steering/controllers.py` para comparar esquemas
explicitamente.

## 2. Sem odometria no robô real

**Sintoma:** `root_pos_w` e `root_lin_vel_w` são sempre zero quando rodando no robô real.

**Causa:** hard-coded:

```python
# booster_deploy/controllers/booster_robot_controller.py:244-249
self._state_buf[0]["root_pos_w"][:] = np.zeros(3, dtype=np.float32)
self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(3, dtype=np.float32)
```

Não há sensor de odometria integrado a esse caminho de estado (o campo `Odometer` existe no
SDK, em outro tópico, mas não é consumido aqui).

**Como lidar:** qualquer tarefa cuja política dependa de posição/velocidade linear absoluta do
tronco (ex. `t1_mimickit_steering`, que usa isso para orientação de "steering") **não pode
funcionar no robô real como está** — é por isso que essa tarefa é documentada como sim-only.
Antes de tentar levá-la a hardware, seria necessário integrar uma fonte real de odometria.

## 3. Convenção de referencial de velocidade linear diverge entre sim e real

**Sintoma:** `root_lin_vel_b` (velocidade linear no referencial do corpo) é calculada de
formas assimetricamente diferentes nos dois backends.

**Causa:** no MuJoCo, `qvel[:3]` é lido diretamente como se já fosse `root_lin_vel_b`, mas na
verdade é **velocidade no referencial do mundo** (convenção do MuJoCo para corpos livres) —
documentado como uma armadilha conhecida em `tasks/mimickit_steering/observation.py:74-99`. No
robô real, a conversão é feita explicitamente:

```python
# booster_deploy/controllers/booster_robot_controller.py:513-518
self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
    self.robot.data.root_quat_w,
    torch.from_numpy(state["root_lin_vel_w"])...
)
```

**Como lidar:** ao escrever uma nova `Policy`, não assuma que `root_lin_vel_b` do
`MujocoController` está de fato no referencial do corpo sem checar o `update_state()`
específico daquele backend. Isso é uma fonte plausível de divergência sim2real sutil se uma
política nova for portada de/para MuJoCo sem revisar essa conversão.

## 4. `action_delay_steps` está inerte

**Sintoma:** configurar `MujocoControllerCfg.action_delay_steps` não introduz nenhum atraso
observável na ação aplicada.

**Causa:** o buffer é criado em `run()` mas nunca lido nem escrito:

```python
# booster_deploy/controllers/mujoco_controller.py:301-302 (dentro de run())
self._action_buffer = ...  # criado, mas nunca usado no ctrl_step
```

O campo existe na config (`controller_cfg.py:30-33`) — o comentário no próprio config deixa
claro que o valor `0` reproduz o comportamento "sem atraso", mas não há implementação para
valores diferentes de `0`.

**Como lidar:** não confie neste campo para modelar atraso de atuação em experimentos de
robustez sim2real — ele precisaria ser implementado antes de ter efeito.

## 5. `chute_t1` não inicia como está commitado

**Sintoma:** rodar `python3 scripts/deploy.py --task chute_t1` falha ao carregar checkpoint ou
arquivo de movimento.

**Causa:**

```python
# tasks/chute_t1/__init__.py:15-16
checkpoint_path="models/paper/cr7_model_14999.pt"
motion_path="motions/paper/booster_t1_cr7.npz"
```

Ambos os diretórios (`tasks/*/models/`, `tasks/*/motions/`) estão no `.gitignore` do
repositório e não estão presentes localmente.

**Como lidar:** essa tarefa depende de artefatos externos (checkpoint + arquivo de movimento)
que precisam ser obtidos separadamente e colocados manualmente nesses caminhos antes de rodar.

## 6. `T1_23DOF_CFG.sim_body_names` está vazio

**Sintoma:** carregadores de movimento que dependem de nomes de corpo padrão
(`default_motion_body_names`) podem se comportar de forma inesperada para o T1.

**Causa:**

```python
# booster_deploy/robots/booster.py:273-275
sim_body_names: list = []
```

**Como lidar:** qualquer tarefa T1 baseada em `MotionLoader` (ex. o caminho `beyond_mimic`)
depende de o próprio arquivo `.npz` de movimento carregar seus próprios `body_names` — não há
fallback vindo da config do robô.

## 7. `enable_velocity_commands` não tem efeito

**Sintoma:** o campo aparece em `beyond_mimic.py:187` e `chute_t1.py:82`, mas mudar seu valor
não altera se comandos de velocidade são aceitos.

**Causa:** não é um campo de `ControllerCfg` — é lido em lugar nenhum do código. O portão real
que decide se há comando de velocidade é `vel_command is None` em `ControllerCfg`.

**Como lidar:** para habilitar/desabilitar comandos de velocidade de fato, defina (ou remova)
`ControllerCfg.vel_command`, não `enable_velocity_commands`.

## 8. `fastdds_profile.xml` documentado, mas ausente

**Sintoma:** `README.md:94` lista `fastdds_profile.xml` como parte do layout do repositório
("Default FastDDS settings for ROS 2"), mas o arquivo não existe.

**Como lidar:** não assuma que há tuning de QoS/DDS customizado sendo aplicado por esse
caminho — hoje não há. Se ajuste de QoS for necessário (ex. mudar profundidade de fila,
transporte, etc.), o hook correto no SDK é `ChannelFactory::InitWithConfigPath`, mas nada no
repositório o invoca atualmente.

## 9. Binding Python do SDK pode estar defasado do header C++

**Sintoma:** métodos documentados no header C++ do SDK (`GetStatus`, `SwitchGait`,
`UpperBodyCustomControl`, o modo `RobotMode.kSoccer`) podem não existir no binding Python
instalado.

**Causa:** o snapshot do binding Python encontrado no histórico do SDK é mais estreito que o
header C++ atual — é plausível que seja exatamente por isso que a verificação de modo em
`booster_robot_controller.py:326-333` está comentada (ver [02](02-modos-e-ciclo-de-vida.md)).

**Como lidar:** antes de depender de um método específico do `B1LocoClient` em Python, verificar
sua existência no wheel instalado (`python3 -c "from booster_robotics_sdk_python import
B1LocoClient; print(dir(B1LocoClient))"`) em vez de assumir paridade total com o header C++.

## 10. `EvaluatorCfg` é andaime sem uso

**Sintoma:** `EvaluatorCfg` e `register_evaluator`/`_EVALUATOR_REGISTRY` existem no código, mas
nenhuma tarefa os popula, e nada os consome no fluxo de `deploy.py`.

**Como lidar:** não é um mecanismo funcional hoje — ignorar ao procurar como avaliação/métricas
são feitas (ver `SyncedMetrics` para o que de fato está em uso, descrito no
[runbook](06-runbook-deploy-real.md)).

## 11. O default de `LowCmd.cmd_type` é `PARALLEL`, e nada verifica o que chegou

**Sintoma:** o IDL do SDK inicializa o campo como `PARALLEL`
(`../booster_robotics_sdk/include/booster/idl/b1/LowCmd.h:191`), e todos os exemplos do
fabricante usam `PARALLEL` com `SERIAL` comentado uma linha acima. A única coisa que coloca o
deploy no espaço serial é `booster_robot_controller.py:282`. Se essa linha sumir num refactor,
os quatro slots de tornozelo passam a ser interpretados como ângulos de crank — silenciosamente,
sem erro, e com o robô de pé.

Agrava: o handler itera `low_state_msg.motor_state_serial` sem checar o comprimento (`:237`), e
`joint_names` nomeia os slots 15/16/21/22 como `Ankle_Pitch`/`Ankle_Roll`
(`booster_deploy/robots/booster.py:241-248`) sem que nada confirme que o firmware concorda.

**Como lidar:** logar `cmd_type` e `len(motor_state_serial)` uma vez na inicialização, e rodar o
teste de identidade de junta com o robô suspenso antes do primeiro run — ver
[13 §6](13-juntas-serial-vs-parallel.md).

## Ver também

- [03 — Ganhos PD](03-ganhos-pd-sim-vs-real.md)
- [04 — Mapa do framework](04-mapa-do-framework.md)
- [13 — Juntas seriais e paralelas](13-juntas-serial-vs-parallel.md)
