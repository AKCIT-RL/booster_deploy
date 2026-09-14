# 12. A política da Booster (`t1_walk`) como controle do estudo

> Branch `tsinghua-v2`, sobre `BoosterRobotics/booster_deploy@7bb1462e`.
> Reprodução: `scripts/sweep_vendor.sh` (42 rodadas). Figuras e tabelas desta página vêm de
> `/tmp/viab-vendor`.

Tudo em [08](08-caminho-para-hardware-mimickit.md) e [09](09-analise-graficos.md) foi medido numa
única política: a nossa, `t1_mimickit_steering`, treinada em MimicKit. Duas conclusões saíram de
lá — que a lacuna de observabilidade domina, e que o envelope de atuador delimita — e nenhuma
tinha **controle**.

O release `7bb1462e` entrega o controle que faltava: `tasks/locomotion` com `t1_walk.pt`, a
política de caminhada que a própria Booster embarca para o T1, no mesmo framework, no mesmo robô
e sob o mesmo teto de torque.

## 1. O que a política da vendor observa

Lido de `tasks/locomotion/locomotion.py:110-153` e confirmado pela forma do tensor — o TorchScript
aceita entrada 720 e devolve 21 ações:

| Bloco | Dim | Vem de qual sensor no robô real |
|---|---:|---|
| `base_ang_vel` | 3 | giroscópio do IMU |
| `projected_gravity` | 3 | quatérnio do IMU |
| `command` (vx, vy, vyaw) | 3 | comando do operador |
| `dof_pos − default_pos` | 21 | encoders |
| `dof_vel` | 21 | encoders |
| `last_action` | 21 | interno à política |
| **por frame** | **72** | |
| × `actor_obs_history_length = 10` | **720** | |

São 21 juntas, não 23: o pescoço (`aahead_yaw`, `aahead_pitch`) fica de fora, e a política não o
comanda.

**Não há `root_pos_w`. Não há `root_lin_vel_b`.** Cada uma das 72 dimensões por frame é uma
grandeza que o T1 mede de fato. Onde a nossa política precisa de altura do tronco e velocidade
linear absoluta — as 4 dimensões que nenhum sensor de bordo fornece
([05](05-armadilhas-conhecidas.md), armadilha 2) — a da vendor compensa com **histórico**: dez
frames de posição, velocidade e ação passadas, de onde a dinâmica translacional pode ser inferida
sem nunca ser observada diretamente.

Essa é exatamente a prescrição de [07](07-htwk-gym-metodo-comparado.md), e agora ela tem prova de
existência no mesmo robô e no mesmo framework — não é uma escolha de projeto que só o htwk-gym faz.

## 2. `zero_root` é um no-op exato — e isso é o resultado

Rodar `--degrade zero_root` no `t1_walk` zera `root_pos_w` e `root_lin_vel_b` a cada
`update_state`, exatamente como o hardware faz. Para esta política, **não muda nada**:

| vx | `--degrade none` | `--degrade zero_root` | idêntico? |
|---|---:|---:|---|
| 0,5 m/s | 60,00 s | 60,00 s | sim |
| 1,0 m/s | 60,00 s | 60,00 s | sim |

E não "praticamente igual": `elapsed_s`, `root_h_min`, `speed_mean` e `tracking_pct` batem **até o
último dígito** (`root_h_min = 0.6197903744749579` nos dois). Tem de ser assim — nenhuma das 720
entradas depende dessas grandezas.

Contra a nossa política, no mesmo eixo:

| Política | estado verdadeiro | estado zerado (o do robô) |
|---|---:|---:|
| `t1_walk` (Booster) | 60,00 s | **60,00 s** |
| `t1_mimickit_steering` (nossa) | 60,00 s | **1,00 s** |

**Isto reenquadra a Fig. 3 de [09](09-analise-graficos.md).** Ela não mostra que a nossa política é
frágil; mostra que *aquela classe de observação* é inviável no T1. A mesma física, o mesmo robô e o
mesmo teto de torque sustentam por 60 s uma política que não pede o que o robô não mede.

A conclusão é **independente da taxa de controle**: rodando a nossa política a 50 Hz (a taxa da
vendor) em vez dos seus 30 Hz nativos, o colapso sob `zero_root` continua em **1,02 s** contra
1,00 s. Não é artefato de discretização.

## 3. O envelope de atuador: a vendor atravessa a matriz inteira

Mesmos seis modelos da Fig. 5, mesma duração de 60 s:

| Modelo | `t1_walk` @ 0,5 | `t1_walk` @ 1,0 | nossa @ 1,0 (docs/09) |
|---|---:|---:|---:|
| `urdf` | 60,00 s | 60,00 s | 60 s |
| `urdf --tn` | 60,00 s | 60,00 s | 60 s |
| `derated` | 60,00 s | 60,00 s | 52,8 s |
| `derated --tn` | 60,00 s | **60,00 s** | **9,6 s** |
| `catalog` | 60,00 s | 60,00 s | 60 s |
| `catalog --tn` | 60,00 s | 60,00 s | 60 s |

**A política da Booster não cai em nenhum modelo, em nenhuma das duas velocidades** — inclusive sob
`derated --tn`, com 92,7% de rastreamento, que é o modelo onde a nossa dura 9,6 s.

Isso é o oposto do que se poderia esperar. A hipótese natural era: *se a vendor também cair sob
`derated --tn`, esse modelo é pessimista demais.* Ela não cai. Logo `derated --tn` **não é um
envelope impossível** — é um envelope que uma política bem treinada atravessa. Os 9,6 s são
propriedade da **nossa política**, não do hardware.

### 3a. Ressalva que enfraquece o próprio número de 9,6 s

Ao rodar a nossa política a 50 Hz (§5) ela sobrevive 60 s sob `derated --tn`. Como taxa de controle
e taxa de física mudaram juntas, foram separadas:

| controle | física | `derated --tn`, vx=1,0 |
|---:|---:|---:|
| 30 Hz | 120 Hz (nativo) | 9,60 s ✕ |
| 30 Hz | 300 Hz | 25,07 s ✕ |
| 50 Hz | 100 Hz | 5,66 s ✕ |
| 50 Hz | 500 Hz | 60,00 s |

**Quem domina é a taxa de física, não a de controle.** Com o controle fixo em 30 Hz, passar a
física de 120 para 300 Hz leva 9,6 → 25,1 s. É coerente com o mecanismo: o PD explícito e o clip
de torque são recalculados por substep a partir da velocidade no início do substep, então substeps
grossos exageram a demanda de torque e o clip morde mais forte.

Consequência honesta: **o valor de 9,6 s da Fig. 5 é em boa parte artefato de resolução de
integração**, e a faixa "9,6 · 52,8 · 60 s" de [09](09-analise-graficos.md) não deve ser lida como
três leituras do hardware. A *ordenação* dos modelos se mantém (`derated --tn` é o mais severo);
os tempos absolutos a 120 Hz, não.

## 4. Varredura de velocidade: sem fronteira na faixa comandável

`vx_max = 1.0` no `t1_walk`, então a faixa varrida é 0,3 a 1,0 — acima disso seria extrapolação de
comando.

| vx | sobrevivência | rastreamento |
|---:|---:|---:|
| 0,3 | 60,00 s | 0,0% |
| 0,4 | 60,00 s | 76,7% |
| 0,5 | 60,00 s | 85,9% |
| 0,7 | 60,00 s | 92,3% |
| 0,9 | 60,00 s | 99,1% |
| 1,0 | 60,00 s | 99,8% |

Nenhuma queda, com `urdf` ou com `catalog --tn`. Duas leituras:

- **O rastreamento melhora com a velocidade**, ao contrário da nossa política, que fica em 86–103%
  em toda a faixa. A política da vendor é mais precisa justamente onde é mais exigida.
- **A 0,3 m/s ela fica parada** (velocidade 0,000 na segunda metade, tronco estável a 0,620 m).
  É uma zona morta de comando, não uma falha — o robô simplesmente não anda abaixo de ~0,4 m/s.

## 5. Ganhos não são portáveis entre políticas

O release embarca **duas filosofias de ganho** para o mesmo robô:

- `robots/t1.py` traz o conjunto que resolve para **f_n = 4 Hz, ζ = 1,5**
  ([06](06-runbook-deploy-real.md#kd-de-junta-paralela-vs-kd-de-motor)); é o que o
  `t1_motion_tracking` usa, por não sobrescrever nada.
- `T1WalkControllerCfg` **sobrescreve** para 200/5 nas pernas.

Trocando um pelo outro, sem mexer em mais nada:

| vx | ganhos da task (200/5) | padrões de `robots/t1.py` (f_n = 4 Hz) |
|---:|---:|---:|
| 0,5 | 60,00 s | **1,60 s ✕** |
| 1,0 | 60,00 s | **1,32 s ✕** |

A política da Booster, sob os ganhos padrão da Booster, cai em menos de dois segundos. Não é um
defeito de nenhum dos dois conjuntos: uma política de alvo de posição é metade de uma malha
fechada, e o PD é a outra metade — é a mesma razão pela qual a nossa task fixa os próprios ganhos
em vez de herdar os do `t1_walk`.

O que isso acrescenta é a **magnitude**: 60 s contra 1,3 s. Ganhos não são um detalhe de
configuração que se empresta entre tarefas, e o conjunto f_n = 4 Hz de `robots/t1.py` é um
*default de robô*, não um conjunto universalmente seguro.

## 6. Ressalvas

1. **O braço de 50 Hz da nossa política é confundido.** A docstring de
   `mimickit_steering.py:27-33` registra que o checkpoint foi treinado a 30 Hz / 120 Hz. Rodá-lo a
   50 Hz o tira da distribuição de treino, então §3a mede "resolução de integração **e** saída de
   distribuição" juntas. O que §3a estabelece com segurança é que **9,6 s não é um número robusto**;
   não estabelece qual seria o número certo.
2. **Não sabemos o modelo de atuador de treino do `t1_walk`.** É um TorchScript sem metadados. O que
   foi medido é o comportamento dele sob *os nossos* modelos.
3. **Taxas nativas diferentes** (50 Hz/500 Hz contra 30 Hz/120 Hz) tornam qualquer comparação
   direta de tempo de queda entre as duas políticas confundida — por isso §3a existe, e por isso a
   comparação que carrega peso aqui é a de §2, que é estrutural.
4. **Acima de 1,0 m/s é extrapolação** para o `t1_walk`.

## 7. O que isto muda no estudo

1. **A lacuna de observabilidade é a conclusão robusta.** Sobrevive a troca de política (a vendor é
   imune por construção), a troca de taxa (1,00 vs 1,02 s) e a troca de modelo de atuador. É o
   achado que deve orientar qualquer retreino — ver [07](07-htwk-gym-metodo-comparado.md).
2. **A conclusão de envelope de atuador enfraquece.** O número mais dramático da Fig. 5 não
   sobrevive nem à troca de política nem ao refinamento da física. `derated --tn` continua sendo o
   modelo mais severo, mas não é uma parede de hardware.
3. **A Q1 de [10](10-discussao-proximos-passos.md) muda de natureza.** Ela perguntava qual modelo de
   atuador é o robô, assumindo que a resposta definia o envelope operacional. Com a vendor
   atravessando os seis modelos, a pergunta de engenharia mais útil passa a ser *por que a nossa
   política precisa de tanto torque*, e não *quanto torque o robô tem*.
4. **Ganhos entram na lista de coisas que não se herdam** — junto com a taxa de controle, já
   documentada.

## Ver também

- [09 — Análise gráfica](09-analise-graficos.md): as figuras que este documento reenquadra.
- [07 — Método comparado: htwk-gym](07-htwk-gym-metodo-comparado.md): a prescrição de observação
  que o `t1_walk` confirma.
- [11 — Reavaliação do release](11-reavaliacao-release-2026-09.md): o que o release mudou.
