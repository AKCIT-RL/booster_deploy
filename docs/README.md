# Documentação: sim2real do `booster_deploy`

Esta pasta documenta como o `booster_deploy` conecta uma política treinada em simulação ao
robô real Booster T1/K1, com foco em três perguntas: qual protocolo de comunicação é usado,
como funciona a troca de modos do robô, e onde/como os ganhos do controlador PD são
configurados em simulação (MuJoCo) versus no robô real.

Todas as afirmações técnicas nestes documentos são ancoradas em `arquivo:linha` do código,
referentes ao commit `0dba32b` (branch `tsinghua`). Trechos derivados de strings/símbolos do
binário pré-compilado do SDK (`libbooster_robotics_sdk.a`), em vez do código-fonte, estão
marcados explicitamente como tal.

## Índice

1. [Arquitetura de comunicação](01-arquitetura-comunicacao.md) — ROS 2/DDS pub-sub vs. RPC:
   os dois planos de comunicação e quem usa cada um.
2. [Modos e ciclo de vida](02-modos-e-ciclo-de-vida.md) — `RobotMode`, o handshake de entrada
   e saída do modo CUSTOM, e por que ele existe.
3. [Ganhos PD: simulação vs. robô real](03-ganhos-pd-sim-vs-real.md) — onde `kp`/`kd` entram
   em cada backend e como parametrizá-los por tarefa.
4. [Mapa do framework](04-mapa-do-framework.md) — abstrações, arquitetura de dois processos,
   ordenação de juntas, taxas de controle, tarefas registradas.
5. [Armadilhas conhecidas](05-armadilhas-conhecidas.md) — comportamentos verificados no
   código que divergem do que se esperaria.
6. [Runbook de deploy no robô real](06-runbook-deploy-real.md) — procedimento operacional e
   checklist de segurança.
7. [Método comparado: htwk-gym](07-htwk-gym-metodo-comparado.md) — como o htwk-gym evita a
   lacuna de observabilidade (`root_h`/`root_vel`) que bloqueia a `t1_mimickit_steering`.
8. [Caminho para hardware: mimickit_steering](08-caminho-para-hardware-mimickit.md) — inventário
   do que o T1 expõe, escada de viabilidade em simulação, envelope de atuador e runlist de
   validação.
9. [Análise gráfica dos experimentos](09-analise-graficos.md) — as cinco figuras dos degraus 0
   a 1 e o que elas mostram que as tabelas escondiam.
10. [Pontos de discussão e próximos passos](10-discussao-proximos-passos.md) — as questões em
    aberto, o que é relevante pré/durante/pós deploy, e a pauta de retreinamento.
11. [Dependências, preflight e handover](11-dependencias-e-preflight.md) — o que a primeira
    tentativa real de deploy no T1 quebrou, as três armadilhas de instalação no robô
    (Python 3.10, numpy/ABI, requirements de treino), e as correções de continuidade do
    handover, guard de queda e limitador de taxa.
12. [Guia de teste no robô real](12-guia-teste-robo-real.md) — a sequência concreta para a
    `t1_mimickit_steering`, da validação em sim ao run no pórtico, com o que observar em cada
    ponto e uma tabela de troubleshooting.
13. [Juntas seriais e paralelas](13-juntas-serial-vs-parallel.md) — o mecanismo de tornozelo do
    T1, os dois espaços de coordenadas do firmware, o que a simulação não modela e o contrato
    de espaço de ação entre treino e deploy.

## TL;DR das três perguntas centrais

**A comunicação é puramente ROS 2, ou há RPC?**
É híbrida, sobre o mesmo transporte Fast DDS. O streaming de controle (`LowCmd`/`LowState`,
tópicos `/joint_ctrl` e `/low_state`) é pub/sub puro, falado via `rclpy` (não usa as classes
pub/sub do SDK). A troca de modo, movimentação de alto nível e consultas de status passam por
um RPC caseiro do SDK (`B1LocoClient` → `RpcClient`), que por baixo dos panos também é
implementado como dois tópicos DDS (`RpcReqMsg`/`RpcRespMsg`) correlacionados por UUID — não é
um serviço ROS 2 nem DDS-RPC nativo. Ver [01](01-arquitetura-comunicacao.md).

**O deploy roda em modo CUSTOM?**
Sim. O portal do robô aguarda confirmação do operador, publica um `LowCmd` segurando a pose
medida com os ganhos de "prepare", chama `ChangeMode(kCustom)` via RPC, executa uma rampa de
500 passos (2 ms cada) até a pose de preparo, aguarda uma segunda confirmação e só então
inicia o processo de inferência da política. Na saída, o modo volta para `kWalking`.
Ver [02](02-modos-e-ciclo-de-vida.md).

**Os ganhos do PD são parametrizáveis no script de deploy, ou ficam na plataforma do robô?**
São parametrizáveis no config Python da tarefa — não há nada a configurar na plataforma do
robô. No robô real, `kp`/`kd` são enviados a cada comando (50 Hz) dentro do próprio `MotorCmd`
e o PD roda nas placas dos motores. Em MuJoCo, o PD é explícito em Python, com `kd` forçado a
zero (esperando amortecimento passivo do MJCF, que o modelo do T1 não declara — ver
[05](05-armadilhas-conhecidas.md)). Existem dois conjuntos de ganhos por robô:
`prepare_state.stiffness/damping` (só na entrada em CUSTOM) e
`joint_stiffness/joint_damping` (regime normal da política), ambos definidos em
`booster_deploy/robots/booster.py` e sobrescritíveis por tarefa. Ver
[03](03-ganhos-pd-sim-vs-real.md).
