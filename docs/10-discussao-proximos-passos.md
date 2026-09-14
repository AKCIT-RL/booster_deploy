# 10. Pontos de discussão e próximos passos

> Este documento não registra medições — [08](08-caminho-para-hardware-mimickit.md) e
> [09](09-analise-graficos.md) fazem isso. Aqui estão as **perguntas que os resultados abriram**,
> as decisões que dependem delas, e o que é relevante antes, durante e depois de um deploy no
> Booster T1.
>
> Onde tenho recomendação, ela está marcada como tal. Onde a questão está genuinamente aberta,
> está marcada como aberta — e com o que a resolveria.

## Parte I — As quatro questões em aberto

### Q1. Qual modelo de atuador é o robô?

> **Muda de natureza com [12](12-politica-vendor-t1-walk.md).** O `t1_walk` da Booster atravessa os
> **seis** modelos a 0,5 e 1,0 m/s sem cair, inclusive `derated --tn`. E os 9,6 s da nossa política
> são em boa parte artefato de integração (25,1 s com a física a 300 Hz). A pergunta que destrava
> engenharia deixa de ser *quanto torque o robô tem* e passa a ser **por que a nossa política pede
> tanto torque**.

**O que medimos.** A 1,0 m/s, três leituras defensáveis do mesmo hardware dão três resultados:

| leitura | sobrevivência | argumento |
|---|---|---|
| `derated --tn` | **9,6 s** | firmware clipa nos derated **e** o atuador perde torque com velocidade |
| `derated` | **52,8 s** | firmware clipa nos derated; a queda com velocidade já estaria embutida |
| `catalog --tn` | **60 s** | o atuador não passa do pico físico e perde torque com velocidade |

**A pergunta.** Os valores de `T1_EFFORT_FIRMWARE` são um limite *instantâneo* (e então a
curva T-N se soma a eles) ou já são uma **aproximação plana conservadora do envelope inteiro**,
embutindo derating térmico e de velocidade? No segundo caso, aplicar os dois é contagem dupla e
o 9,6 s é artefato.

> **Evidência nova (release `7bb1462e`).** A Booster **substituiu** os limites de firmware por
> os do URDF em `T1_23DOF_CFG.effort_limit` — ou seja, promoveu o teto alto a configuração
> oficial de deploy. Isso não fecha a Q1, mas empurra a leitura para longe do `derated --tn`
> (9,6 s) e na direção do `catalog --tn` (60 s): a linha derated que gerava o pior caso deixou de
> ser o que o fabricante embarca. Ver
> [11 §2a](11-reavaliacao-release-2026-09.md#2a-effort_limit-trocou-de-significado--o-risco-mais-grave).
> A pergunta à Booster continua valendo, e agora tem uma formulação mais direta: *por que o
> `effort_limit` passou a ser o do URDF?*

**Por que importa.** É a diferença entre "anda quase um minuto" e "cai em dez segundos" no
regime que a equipe mais quer usar. Nenhuma decisão de envelope operacional é sólida sem isso.

**O que resolve.** Uma pergunta à Booster: *os limites de operação por junta já consideram a
queda de torque com a velocidade, ou devem ser combinados com a curva T-N do catálogo?* É barato
e destrava o resto.

**Enquanto não resolve:** operar como se fosse o pior caso. A 0,5 m/s as três leituras
concordam (60 s), então o regime suave é seguro sob qualquer delas.

---

### Q2. O ganho de rastreamento com teto menor é real e explorável?

**O que medimos.** Baixar o teto de torque **melhorou** o rastreamento:

| | teto URDF | teto derated |
|---|---|---|
| vx = 0,5 | 96,8% | **99,1%** |
| vx = 1,0 | 88,8% | **99,8%** |

**A hipótese.** Com `kp = 80` e alvos de posição **absolutos** (esta política não usa resíduo
nem `action_scale`), o laço PD estaria sobre-acionando, e o clip funciona como limitador. Se for
isso, o teto mais baixo não é uma restrição tolerada — é **benéfico**.

**Por que é interessante.** Se confirmado, existe um ganho de desempenho grátis disponível
antes de qualquer retreino: reduzir `kp`, ou o teto, ou ambos. E sugere que os ganhos herdados
(`T1_KP` com `kd = 0,0637·kp`) não estão sintonizados para este laço.

**O que resolve.** Uma varredura de `kp` com teto fixo. Se o rastreamento melhorar ao baixar
`kp` *sem* clip, a hipótese do sobre-acionamento se confirma e o clip é só um sintoma.

**Status: aberta, e é a mais barata de todas as quatro.**

---

### Q3. Estimador ou retreino — o que vem primeiro?

**O que medimos.** A observabilidade domina por ~50× (1,0 s contra 60 s). O envelope de atuador
é de segunda ordem e não impede a marcha a 0,5 m/s em nenhum modelo.

Isso **reverteu** a leitura anterior: o estimador voltou a ser caminho crítico, porque é a única
variável que derruba em um segundo e a única com conserto em software.

**A pergunta.** Construir o estimador e tentar salvar `t1_wrturn_s1`, ou aceitar que este
checkpoint não vai a hardware e investir direto no retreino?

**O argumento pelo estimador:** não expira. Serve qualquer política futura, é pré-requisito do
degrau 4, e sem ele não há como medir a velocidade real de caminhada do T1 de forma alguma —
nem para diagnóstico.

**O argumento pelo retreino:** mesmo com estimador perfeito, sobram a estabilidade marginal
(queda aos 52,8 s a 1,0 m/s com estado verdadeiro), a latência do estimador que o treino nunca
viu, e a deriva de yaw. O estimador pode não ser suficiente.

**Recomendação:** construir o estimador **primeiro**, porque ele é pré-requisito da medição que
decide o resto (degrau 3: erro e latência contra a verdade do MuJoCo), e essa medição custa
pouco depois que o módulo existe. Mas orçar o retreino em paralelo, não depois.

---

### Q4. Como se para o robô?

**O que medimos.** O treino amostrou `tar_speed` em Uniforme(0,50 · 2,50). Comandar zero entrega
**−2,6σ**. A política **não tem modo parado** — não é uma limitação de configuração, é ausência
de cobertura de treino.

**A pergunta operacional.** Um deploy real precisa de três transições: começar a andar, andar, e
**parar sem cair**. A terceira não existe hoje. Hoje a saída é `ChangeMode(kWalking)`, que
devolve o controle ao controlador nativo — mas sem rampa de torque
([02](02-modos-e-ciclo-de-vida.md)), com o último `LowCmd` valendo até o firmware processar.

**Opções:**

1. **Nunca comandar parado.** Operar sempre em ≥ 0,5 m/s e sair pela troca de modo. Simples,
   mas significa que o robô está andando em todo momento sob a política.
2. **Blend de saída.** Interpolar da última saída da política para a pose de preparo com os
   ganhos de preparo, em ~1 s, antes de trocar de modo. É o espelho da rampa de entrada que já
   existe, e é a que eu faria.
3. **Cobrir parado no retreino** — a solução certa, mas é retreino.

**Status: aberta, e é a que mais fácil passa despercebida até o dia do deploy.**

## Parte II — O que ainda não está na bancada

Quatro degradações reais que **nenhum experimento nosso cobre**, em ordem de facilidade:

| degradação | evidência de que importa | custo de simular |
|---|---|---|
| **atrito seco de junta** | `frictionloss = 0` nas 23 juntas do MJCF; htwk-gym randomiza 0–2 Nm | baixo — é um termo no `ctrl_step` |
| **atraso de atuação** | `action_delay_steps` existe e está inerte (armadilha 4 do [05](05-armadilhas-conhecidas.md)); htwk-gym randomiza por episódio | baixo |
| **latência de observação** | inerente à odometria de perna; o treino nunca viu | baixo |
| **deriva de yaw** | corrompe as 5 dims de steering; sem referência absoluta na IMU | baixo — uma rampa no yaw |

Atrito seco merece destaque: é a explicação mais comum para "andava em simulação e ficou lento
no robô", e é o único dos quatro que altera a **energia** do sistema em vez de só a informação.
Adicionar um degrau para ele é barato e pode explicar parte da margem que hoje atribuímos a
outras coisas.

## Parte III — Pré-deploy

### Portões em simulação

Proposta de critério de aceitação, aplicável a **qualquer** política antes de ir a hardware —
não só a esta:

1. Sobrevive 60 s com o **estado degradado ao que o robô entrega** (`--degrade zero_root`, ou o
   estimador real quando existir).
2. Sobrevive 60 s no **envelope de atuador pessimista** (`--effort derated --tn`) na faixa
   operacional pretendida.
3. O guard de queda dispara antes do robô chegar ao chão, nas duas condições acima.
4. As velocidades de junta ficam abaixo do **rated speed** do catálogo na faixa pretendida.

Hoje `t1_wrturn_s1` falha 1 e 2. Isso é informação, não veredito — mas deveria ser explícito
antes de energizar.

### Correções que não dependem do robô

Da §7 de [08](08-caminho-para-hardware-mimickit.md), em ordem de valor: guard de queda,
limitador de taxa em `dof_targets`, `prepare_state` no `.replace()`, `yaw₀` no `reset()`,
**logging de estado no caminho real** (hoje inexistente).

O logging merece insistência: sem ele, uma queda no robô é **inatribuível**. Não se distingue
"a política pediu besteira" de "o hardware não entregou" — que é exatamente a distinção que
todos estes documentos passaram a investigar em simulação.

### Manifesto de rodada

Emitido no início: task, commits de `booster_deploy` e do repo de treino, sha256 do checkpoint,
`kp`/`kd`/`effort`/`default_joint_pos`/`prepare_state` resolvidos, `policy_dt`/`decimation`,
versões de firmware e SDK, modelo de atuador usado na validação. Com `.replace()` e
`__post_init__` mexendo em campos, o config efetivo não é óbvio a partir do fonte.

### Envelope operacional

Da varredura: **0,5–1,0 m/s**, onde todos os modelos de atuador concordam. Sem `vy` e sem ré —
as estatísticas mostram `tar_dir ≡ face_dir` em todo o treino, então qualquer comando lateral é
regime inédito.

## Parte IV — Durante o deploy

### O que monitorar ao vivo

| sinal | esperado | o que significa divergir |
|---|---|---|
| `policy_step` | 30 Hz | inferência não cabe no orçamento; o laço passa a comandar espaçado |
| `low_state_handler` | ~500 Hz | perda de amostras ou thread travada — e o estado congela silenciosamente |
| `tau_est` por junta | abaixo do derated | saturação real; comparar com a Fig. 1 de [09](09-analise-graficos.md) |
| altura estimada do tronco | ~0,69 m | o estimador é o único sinal de que a política está recebendo verdade |
| `projected_gravity[2]` | ≈ −1 | o guard de queda; é a grandeza mais confiável do robô |

As duas primeiras já existem via `SyncedMetrics`, mas só são impressas **no fim**
([06](06-runbook-deploy-real.md)). Ao vivo, seriam mais úteis.

### Abortos e suas latências

Vale mapear explicitamente, porque cada um tem um tempo diferente e a Fig. 3 de
[09](09-analise-graficos.md) mostra que há **cerca de um segundo** entre o início da divergência
e o chão:

- guard de queda em `projected_gravity` — dentro de um passo de política (33 ms)
- limitador de taxa em `dof_targets` — imediato, mas só pega saída absurda
- abort por estado obsoleto — precisa existir; hoje `synced_state.read()` devolve o último valor
  para sempre
- `Ctrl-C` → `kWalking` — depende do operador, mais a latência do RPC
- `kDamping` pelo controle remoto — o mais rápido, e solta o robô

Um segundo é pouco para reação humana. Os abortos automáticos são os que contam.

### Autoridade em estágios

As duas confirmações do operador que já existem acontecem **antes** da política rodar. Falta um
envelope de comando para as primeiras rodadas: limitar `vx`/`vyaw` a uma fração do
`VelocityCommandCfg`, que hoje permite `vx_max = 2.5` — exatamente onde a varredura mede queda
em 4,6 s sob o modelo físico.

## Parte V — Pós-deploy

### Atribuição

A pergunta a responder depois de cada rodada, bem ou mal sucedida: **o robô se comportou como a
simulação previu?** Isso exige que os dois lados gravem as mesmas grandezas. Hoje o lado da
simulação grava (`log_states`) e o do robô não.

### Realimentar o modelo

Cada rodada em hardware produz números que melhoram a simulação, e essa é a parte que costuma
não acontecer:

- `tau_est` medido → confirma ou refuta qual modelo de atuador é o robô (**resolve a Q1
  empiricamente**, se a pergunta à Booster não for respondida)
- velocidade de junta medida → onde a curva T-N realmente morde
- discrepância entre alvo e posição em regime → estimativa de atrito seco
- deriva de yaw em 60 s parado → parametriza a degradação que falta na bancada
- diferença entre a velocidade estimada e a integrada → qualidade da detecção de contato

Cada um desses fecha uma das lacunas da Parte II. Um deploy que falha mas gera esses números
vale mais que um que anda e não grava nada.

### Critério de sucesso

Vale combinar antes, porque "andou" é ambíguo. Proposta: sobreviver 60 s a 0,5 m/s com
rastreamento dentro de ±15% do que a simulação prevê para o mesmo modelo de atuador, sem o guard
de queda disparar.

## Parte VI — Discussão de retreinamento

### O que é consenso técnico

Da comparação com o htwk-gym ([07](07-htwk-gym-metodo-comparado.md)), três mudanças não têm
contra-argumento sério:

1. **`root_h` e `root_vel` vão para o crítico**, nunca para o ator. Custa mudar qual tensor vai
   para qual rede, não a tarefa nem a recompensa.
2. **Treinar contra os limites de torque corretos** — e agora "corretos" significa o catálogo do
   fabricante, não o URDF, que é **não-físico** (joelho 130,5 contra pico real de 65). Com a
   curva T-N, se a Q1 disser que ela se aplica.
3. **Randomizar observação e atraso de atuação.** `apply_randomization` por canal do htwk-gym é
   o gancho; nós temos o campo `action_delay_steps` e nenhuma implementação.

### O que é decisão de projeto, não consenso

**A fonte da fase da marcha.** Tirando `root_h`/`root_vel` do ator, a política perde a
informação de fase do ciclo. Três substitutos, e a escolha não é óbvia:

| opção | custo | risco |
|---|---|---|
| histórico de frames | muda a arquitetura (a política é memoryless hoje) e o contrato do módulo exportado | baixo — é o que a `locomotion` já faz com sucesso neste robô |
| relógio de marcha imposto | elegante, zero sensores, dá `gait_frequency` como controle | **briga com o motion prior** do MCWAMP, a menos que as referências sejam alinhadas em fase |
| estimador como entrada | mantém a observação | exige treinar com o modelo de erro do estimador, senão só move o gap |

**Recomendação:** histórico de frames, por ter precedente funcionando no mesmo robô e não
conflitar com o prior. Mas é a decisão que mais merece discussão, porque muda a arquitetura.

**O espaço de comando.** A steering comanda direções de **mundo** (`tar_dir`, `face_dir`), e é
só por isso que a deriva de yaw a atinge. A `locomotion` e o htwk-gym comandam twist no corpo e
são estruturalmente imunes. Migrar o espaço de comando resolveria a deriva de yaw de uma vez —
mas descaracteriza a tarefa de steering, que é imitativa e definida em direções de mundo.

**A lacuna de strafe.** As estatísticas do normalizador mostram `tar_dir` e `face_dir` com
distribuições idênticas: coincidiram em **todo** o treino. A política nunca aprendeu a andar numa
direção olhando para outra. Mas `steering_command` calcula `heading = face_heading + atan2(vy, vx)`
e o config expõe `vy_max = 1.0`. Ou o treino cobre isso, ou o comando não deveria estar exposto.

**Cobertura de parado.** Ver Q4. O htwk-gym trata como modo explícito com 10% dos ambientes e um
portão no relógio. Aqui não existe.

### Uma pergunta de escopo

Vale perguntar se `mimickit_steering` é a formulação certa para caminhar no T1, ou se a tarefa
de caminhada deveria seguir o caminho `locomotion`/htwk-gym (twist no corpo, resíduo sobre pose
padrão, histórico, crítico privilegiado) e a MimicKit ficar para movimentos que **precisam** de
imitação — chute, gestos, transições.

As duas formulações resolvem problemas diferentes. Usar a imitativa para andar em linha reta
paga o custo de observabilidade e de espaço de comando sem usar o que ela tem de melhor.

## Prioridades sugeridas

Ordenadas por (destrava outras coisas) × (custo baixo):

1. **Perguntar à Booster sobre a Q1.** Um e-mail. Destrava o envelope operacional inteiro.
2. **Varredura de `kp` (Q2).** Uma tarde. Pode dar desempenho grátis antes de qualquer retreino.
3. **Correções independentes** da Parte III. Pequenas, e melhoram os próprios experimentos.
4. **Degraus de atrito seco e latência** (Parte II). Baratos, e fecham lacunas que hoje
   atribuímos a outras causas.
5. **Estimador** (degrau 2) e sua medição de erro (degrau 3).
6. **Diagnóstico no robô** quando houver robô — pode encurtar o 5.
7. **Retreino**, com as decisões da Parte VI tomadas explicitamente e não por omissão.

## Ver também

- [08 — Caminho para hardware](08-caminho-para-hardware-mimickit.md): os números e a runlist.
- [09 — Análise gráfica](09-analise-graficos.md): as figuras e o que elas mostram.
- [07 — Método comparado: htwk-gym](07-htwk-gym-metodo-comparado.md): a receita de referência.
- [06 — Runbook de deploy](06-runbook-deploy-real.md): o procedimento operacional atual.
