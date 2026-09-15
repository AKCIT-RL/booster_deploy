# 6. Runbook: deploy no robô real

> Referência: commit `0dba32b` (branch `tsinghua`). Este runbook complementa
> [02 — Modos e ciclo de vida](02-modos-e-ciclo-de-vida.md); leia aquele documento para
> entender *por que* cada passo existe.

## Pré-requisitos (`README.md:6-13`)

| Item | Requisito |
|---|---|
| Firmware Booster | `>= v1.4` |
| Python | 3.10+ (já instalado no robô) |
| ROS 2 | Humble, com `/low_state` + `/joint_ctrl` disponíveis (já instalado no robô) |
| SDK | `booster_robotics_sdk` compilado **com binding Python** (`-DBUILD_PYTHON_BINDING=ON`) |
| Placa | No T1 Standard Edition, o deploy deve rodar na **placa de movimento**, não na de percepção (`README.md:48`) |

## Checklist antes de energizar

- [ ] **`python3 scripts/preflight.py --task <TASK_NAME>` sai com código 0.** Sem FAIL.
      Este passo é obrigatório e não move o robô. `deploy.py` constrói a policy dentro do
      processo de inferência, forkado **depois** de o robô entrar em modo CUSTOM e rampar
      até a pose de preparo — então uma dependência ausente se manifesta com o robô já de
      pé, e `--list` não avisa (`tasks/__init__` engole erros de import por design). Foi
      exatamente assim que a primeira tentativa real falhou; ver
      [11 — Dependências e preflight](11-dependencias-e-preflight.md).
- [ ] Tarefa já validada em `--mujoco` (sim2sim) com o mesmo checkpoint que será usado no
      robô.
- [ ] Robô fisicamente suspenso ou apoiado com segurança para o primeiro teste de uma tarefa
      nova (a rampa de preparo e a política podem produzir movimento inesperado).
- [ ] Área ao redor do robô livre de obstáculos e pessoas na trajetória de movimento previsto.
- [ ] Acesso rápido a um kill switch físico ou ao controle remoto que dispara `kDamping`,
      independente do processo de deploy.
- [ ] Robô já está em modo `Prepare` (postura bipedal de apoio) antes de iniciar o script —
      não é o script que coloca o robô em `Prepare`.
- [ ] `--net` aponta para a interface de rede correta (padrão `127.0.0.1`, para execução
      local na própria placa do robô).

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
  robô é colocado em `RobotMode.kWalking` ao final
  (`booster_robot_controller.py:459-461`).
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
