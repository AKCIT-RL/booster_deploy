# 7. Como o htwk-gym resolve a lacuna de observabilidade

> Referências: `booster_deploy` no commit `0dba32b` (branch `tsinghua`); `htwk-gym` no commit
> `cbb4f51`. Todas as afirmações são ancoradas em `arquivo:linha` dos dois repositórios.
>
> **Prova de existência ([12](12-politica-vendor-t1-walk.md)):** a política de caminhada que a
> própria Booster embarca para o T1 (`t1_walk`) segue a mesma prescrição — observação de 720 dims
> feita só de IMU, encoders e ação anterior, com dez frames de histórico e **zero** estado de raiz.
> Não é uma peculiaridade do htwk-gym; é o que o fabricante faz no mesmo robô.

Este documento compara o **método** de treino e deploy do `htwk-gym` (HTWK Robots, derivado do
`booster_gym`) com o que o `booster_deploy` faz hoje — especificamente com a lacuna
identificada na tarefa `t1_mimickit_steering`: quatro dimensões da observação
(`root_h` e as três de `root_vel`) que o simulador preenche e o robô real não tem como
preencher, hoje zeradas em
`booster_deploy/controllers/booster_robot_controller.py:244-249`.

O foco é método, não stack de software. Ambos os projetos rodam o mesmo SDK no mesmo robô.

## Resposta direta

O htwk-gym **não estima** `root_h` e `root_vel` no robô. Ele faz duas coisas diferentes:

1. **Move as duas grandezas para o crítico.** Elas continuam existindo no treino, com ruído
   inclusive, mas vivem em um buffer separado que o ator nunca vê e que é descartado no
   deploy. É actor-critic assimétrico, aplicado exatamente ao caso que nos interessa.
2. **Substitui a informação de fase por um relógio comandado.** O que `root_h`/`root_vel`
   carregariam sobre o ciclo de marcha vem de um `cos`/`sin` de fase que o próprio lado do
   deploy gera a partir do relógio de parede — zero sensores.

O resultado é que a observação do ator tem 47 dimensões e **todas as 47 são produzíveis no
robô real**. A lacuna não é resolvida: ela é evitada por construção.

## 1. A separação ator/crítico

A configuração declara os dois espaços separadamente:

```yaml
# htwk-gym/envs/T1/Base_Walk.yaml
env:
  num_observations: 47        # o que o ator vê
  num_privileged_obs: 14      # o que só o crítico vê
  num_actions: 12
```

E a construção é literalmente dois tensores diferentes:

```python
# htwk-gym/envs/T1/base_walk.py:581-593  — ATOR
self.obs_buf = torch.cat(
    (
        apply_randomization(self.projected_gravity, ...),   # 3
        apply_randomization(self.base_ang_vel, ...),        # 3
        self.commands[:, :3] * commands_scale,              # 3
        (torch.cos(2 * torch.pi * self.gait_process) * ...).unsqueeze(-1),  # 1
        (torch.sin(2 * torch.pi * self.gait_process) * ...).unsqueeze(-1),  # 1
        apply_randomization(self.dof_pos - self.default_dof_pos, ...),      # 12
        apply_randomization(self.dof_vel, ...),                             # 12
        self.actions,                                                       # 12
    ), dim=-1,
)
```

```python
# htwk-gym/envs/T1/base_walk.py:594-603  — CRÍTICO
self.privileged_obs_buf = torch.cat(
    (
        self.base_mass_scaled,                                              # 4  (com xyz + massa)
        apply_randomization(self.base_lin_vel, ...),                        # 3  <-- root_vel
        apply_randomization(self.base_pos[:, 2]
                            - self.terrain.terrain_heights(self.base_pos),
                            ...).unsqueeze(-1),                             # 1  <-- root_h
        self.pushing_forces[:, 0, :] * ...,                                 # 3
        self.pushing_torques[:, 0, :] * ...,                                # 3
    ), dim=-1,
)
```

As duas grandezas exatas que nos bloqueiam — `base_lin_vel` e a altura do tronco acima do
terreno — estão no buffer privilegiado. Elas ajudam o crítico a estimar valor com precisão,
o que acelera e estabiliza o aprendizado, e **nunca entram na rede que roda no robô**.

Note também o que mais está lá: massa e CoM da base, e as forças/torques de empurrão
aplicados naquele instante. São grandezas que nenhum robô consegue medir — e é exatamente
por isso que estão no crítico. A regra implícita é clara: *se o robô não mede, vai para o
crítico.*

### O mesmo `root_h` governa recompensa e terminação

O privilégio não se limita à observação do crítico. A altura do tronco também define quando
o episódio termina:

```python
# htwk-gym/envs/T1/base_walk.py:556
self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) \
                  < self.cfg["rewards"]["terminate_height"]      # 0.45 m
```

e alimenta a recompensa `base_height: -20.` contra `base_height_target: 0.68`. Ou seja:
`root_h` **molda fortemente o comportamento aprendido**, sem nunca ser uma entrada da
política. Essa é a distinção que estava faltando na nossa análise — *informação privilegiada
pode governar o treino inteiro desde que saia pelo gradiente, não pela entrada*.

## 2. O relógio de marcha, que substitui a fase

Duas das 47 dimensões do ator são `cos` e `sin` de uma fase de marcha:

```python
# htwk-gym/envs/T1/base_walk.py:479
self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
```

`gait_frequency` é um **comando** (amostrado em `[1.0, 2.0]` Hz), não uma medição. Logo a fase
é um sinal que o operador gera, e no deploy sai do relógio de parede:

```python
# htwk-gym/deploy/utils/policy.py:35
self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
```

Isso é o ponto de projeto mais elegante do repositório e resolve, por outro caminho, parte do
problema que nos preocupa. Na análise da `mimickit_steering`, `root_h` e `root_vel[z]`
eram apontados como portadores da fase do ciclo de marcha — informação especialmente crítica
para uma política **sem memória**, que não pode derivar fase de um histórico. O htwk-gym
não precisa nem de memória nem de estimador: ele **impõe** a fase em vez de observá-la.

O robô não descobre em que ponto da passada está; ele é informado, e aprende a sincronizar o
corpo com um relógio externo.

### O modo "parado" é explícito e treinado

O clock é multiplicado por `(self.gait_frequency > 1.0e-8)`, e 10% dos ambientes são
comandados a ficar parados (`still_proportion: 0.1`), com `gait_frequency = 0`. No deploy o
mesmo portão existe:

```python
# htwk-gym/deploy/utils/policy.py:41-44
if np.linalg.norm(self.smoothed_commands) < 1e-5:
    self.gait_frequency = 0.0
else:
    self.gait_frequency = self.cfg["policy"]["gait_frequency"]
```

Vale contrastar com a `mimickit_steering`, onde parar é **extrapolação**: o treino amostrou
velocidade em Uniforme(0,5 · 2,5) e `MimicKitSteeringPolicyCfg.tar_speed_min` está declarado
como `0.0`, de modo que o clamp não impede o zero de passar. Aqui, "parado" é um modo
distinto, com sinal próprio e cobertura de treino dedicada.

## 3. As 47 dimensões e por que cada uma sobrevive ao robô

| Bloco | Dims | Origem no robô real |
|---|---|---|
| `projected_gravity` | 3 | IMU roll/pitch — referenciado à gravidade, **não deriva** |
| `base_ang_vel` | 3 | giroscópio, direto |
| `commands` (vx, vy, vyaw) | 3 | operador, **no frame do corpo** |
| `cos`/`sin` da fase | 2 | gerado localmente do relógio |
| `dof_pos - default` | 12 | encoders |
| `dof_vel` | 12 | encoders |
| `actions` | 12 | a própria saída do passo anterior |

Não há posição, não há velocidade linear e **não há yaw absoluto em lugar nenhum**. O comando
de velocidade é um *twist no frame do corpo*, não uma direção no mundo — e o reset de treino
sorteia o yaw inicial uniformemente em `[0, 2π)`, o que força a política a ser indiferente ao
rumo absoluto.

Essa é uma decisão de observabilidade que fica escondida na escolha de parametrização do
comando. A `mimickit_steering` comanda `tar_dir`/`face_dir` como **direções de mundo**, e por
isso depende do yaw absoluto nas 5 dimensões da tarefa — que é a única via pela qual a deriva
da IMU a atinge. O htwk-gym não tem essa via porque nunca pergunta ao robô para onde ele
aponta no mundo.

## 4. Treinar a degradação que se vai enfrentar

Todo canal do ator passa por `apply_randomization` com ruído configurado por canal:

```yaml
# htwk-gym/envs/T1/Base_Walk.yaml
noise:
  gravity:  { range: [0., 0.01], operation: "additive", distribution: "gaussian" }
  ang_vel:  { range: [0., 0.1],  operation: "additive", distribution: "gaussian" }
  dof_pos:  { range: [0., 0.01], operation: "additive", distribution: "gaussian" }
  dof_vel:  { range: [0., 0.1],  operation: "additive", distribution: "gaussian" }
```

O ator nunca vê estado limpo. E a randomização vai além da observação, cobrindo o que de fato
difere entre o modelo e o robô: `friction`, `compliance`, `restitution`, `base_com`,
`base_mass`, `other_com`, `other_mass`, `dof_stiffness`, `dof_damping`, `dof_friction`, além
de *kicks* (impulso na velocidade da base, a cada 2 s) e *pushes* (força/torque sustentados,
a cada 5 s).

### Atuador modelado, não idealizado

O laço de física aplica um PD explícito com atrito seco e saturação:

```python
# htwk-gym/envs/T1/base_walk.py — dentro do laço de decimation
self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) \
              - self.dof_damping * self.dof_vel
friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)   # :448
dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
```

E o atraso de atuação é **randomizado por episódio**, dentro da janela de decimation:

```python
# htwk-gym/envs/T1/base_walk.py:317
self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], ...)
```

Vale registrar a ironia: esse é exatamente o mecanismo que
`MujocoControllerCfg.action_delay_steps` deveria fornecer no `booster_deploy` e que está
inerte — o buffer é criado em `run()` e nunca lido (ver
[05 — Armadilhas conhecidas](05-armadilhas-conhecidas.md), item 4). O htwk-gym trata atraso
de atuação como variável de primeira classe do treino; nós temos o campo e não temos a
implementação.

## 5. O deploy é espelho literal do treino

`deploy/utils/policy.py:47-62` monta o vetor de 47 na mesma ordem do `obs_buf` do treino, e a
ação é reconstruída do mesmo jeito:

```python
# htwk-gym/deploy/utils/policy.py:71-72
self.dof_targets[:] = self.default_dof_pos
self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions
```

Dois detalhes de método que valem nota:

- **A ação é residual sobre a pose padrão**, com `clip_actions: 1.` e `action_scale: 1.`.
  O envelope de falha é limitado por construção — contraste com a `mimickit_steering`, cuja
  saída **é** o alvo absoluto de posição, limitado apenas pelo clamp embutido no módulo
  exportado.
- **A normalização é constante de config, não estatística acumulada.** Os fatores
  (`gravity: 1.0`, `dof_vel: 0.1`, …) vivem no YAML e são lidos pelos dois lados. Não há
  normalizador com média/desvio de treino escondido dentro do `.pt`, como há no
  `t1_steering.pt` da MimicKit. O contrato de observação é legível sem abrir o checkpoint.

## 6. Comparação dos três projetos

| | `htwk-gym` T1/BaseWalk | `booster_deploy` locomotion | `booster_deploy` mimickit_steering |
|---|---|---|---|
| Dims do ator | 47 (1 frame) | 72 × 10 frames | 200 (1 frame) |
| `root_h` / `root_vel` no ator | **não** (crítico) | **não** | **sim** — e zeradas no robô |
| Info privilegiada | buffer separado, 14 dims | não visível no checkpoint | não há separação |
| Fase da marcha | relógio comandado (cos/sin) | implícita, via 10 frames | implícita, via `root_h`/`root_vel` |
| Yaw absoluto | nunca usado | nunca usado | usado nas 5 dims de tarefa |
| Frame do comando | twist no corpo | twist no corpo | direção no mundo |
| Ação | residual, clip ±1 | residual, `scale 0.25` | alvo absoluto |
| "Parado" | modo treinado (10%) | comando zero, coberto | extrapolação (−2,6σ) |
| Ruído na observação | por canal, em todos | não visível no checkpoint | nenhum visível |
| Atraso de atuação | randomizado por episódio | campo inerte | campo inerte |
| Juntas controladas | 12 (pernas) | 21 | 23 |

## 7. O que isso muda na nossa decisão

O htwk-gym confirma a técnica que havíamos identificado (actor-critic assimétrico) e adiciona
uma que não tínhamos considerado (impor a fase em vez de observá-la). Mas **não é um substituto
pronto** para a `mimickit_steering`, e vale ser honesto sobre o porquê:

- O escopo é diferente: 12 DOF de perna contra 23 DOF de corpo inteiro.
- A tarefa é diferente: rastreamento de twist no corpo contra *steering* imitativo derivado de
  captura de movimento, onde a direção-alvo no mundo é parte da definição da tarefa.
- Uma política de marcha com relógio imposto abre mão da liberdade de fase que uma política
  imitativa usa para reproduzir um movimento de referência.

O que ele **estabelece** é que as três portas descritas anteriormente não são igualmente
necessárias:

1. **Estimar** (odometria de pé de apoio) — continua sendo a única porta que serve políticas
   já treinadas, e a única que produz um componente reutilizável. Não fica obsoleta.
2. **Retreinar com observação restrita** — deixa de ser hipótese e passa a ter precedente
   funcionando no mesmo robô, com uma receita completa e legível, incluindo quais grandezas
   mandar para o crítico e quais randomizações aplicar.
3. **Retreinar com o erro do estimador injetado** — continua válida, e o `apply_randomization`
   por canal do htwk-gym é exatamente o gancho onde o modelo de erro entraria.

Se a `mimickit_steering` for retreinada, a mudança mínima que a torna deployável é mover
`root_h` e `root_vel` do `obs_buf` para o buffer privilegiado do crítico, e decidir o que
substitui a informação de fase que elas carregavam — memória (histórico de frames) ou relógio
imposto. A política memoryless atual não sobrevive a nenhuma das duas escolhas sem mudar de
arquitetura.

Independentemente do caminho, duas mudanças do htwk-gym são baratas e se aplicam a qualquer
política nossa: **ruído por canal na observação** e **atraso de atuação randomizado** — sendo
que o segundo já tem campo de config esperando implementação do nosso lado.

## Ver também

- [03 — Ganhos PD](03-ganhos-pd-sim-vs-real.md): por que `kp`/`kd` e o período de controle
  fazem parte da política.
- [05 — Armadilhas conhecidas](05-armadilhas-conhecidas.md): item 4 (`action_delay_steps`
  inerte) e item 2 (ausência de odometria).
- [06 — Runbook de deploy](06-runbook-deploy-real.md): o procedimento operacional em que
  qualquer política nova vai entrar.
