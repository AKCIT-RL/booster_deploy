#!/usr/bin/env bash
# Grava um "take": um run do deploy.py com tudo o que é preciso para entendê-lo
# depois. Guia completo: docs/18-takes.md.
#
#   scripts/take.sh [argumentos do deploy.py...]
#
#   exemplo (robô):
#     scripts/take.sh --task t1_mimickit_steering_meas \
#         --checkpoint ../../models/<ckpt>.pt --teleop keyboard --safe-vx 0.3
#
#   exemplo (ensaio no MuJoCo, sem ROS):
#     scripts/take.sh --mujoco --task t1_mimickit_steering_meas \
#         --checkpoint ../../models/<ckpt>.pt --pd explicit --effort derated --tn \
#         --teleop keyboard
#
# O que faz, em ordem:
#   1. pergunta o nome do take (ou usa $TAKE_NAME);
#   2. cria takes/<AAAA-MM-DD>/<HHMMSS>_<nome>/ e grava meta.txt (data,
#      máquina, commit, arquivos modificados, comando);
#   3. no robô, sobe scripts/log_deploy.py em segundo plano (/low_state e
#      /joint_ctrl -> log.npz + log.txt);
#   4. roda o deploy.py dentro de `script` (deploy.txt guarda toda a saída do
#      terminal, e o teclado do teleop continua funcionando) com
#      --log-dir apontando para o take (cmd.csv: vx, vy, vyaw e heading
#      comandados a cada passo; events.csv: modos e damping);
#   5. quando o deploy termina, encerra o gravador, pede as observações e as
#      guarda em notes.md, e completa o meta.txt (código de saída, md5 do
#      checkpoint, duração).
#
# Variáveis: TAKE_NAME (pula a pergunta), PYTHON (interpretador; default
# .venv/bin/python se existir, senão python3), TAKES_DIR (default takes/).

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
TAKES_DIR="${TAKES_DIR:-$REPO/takes}"

# ---- argumentos do deploy que o take precisa conhecer --------------------
TASK="t1_mimickit_steering_meas"
CKPT_ARG=""
MUJOCO=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --task) TASK="${args[$((i + 1))]:-$TASK}" ;;
        --task=*) TASK="${args[$i]#--task=}" ;;
        --checkpoint) CKPT_ARG="${args[$((i + 1))]:-}" ;;
        --checkpoint=*) CKPT_ARG="${args[$i]#--checkpoint=}" ;;
        --mujoco) MUJOCO=1 ;;
        --log-dir|--log-dir=*)
            echo "[take] ERRO: nao passe --log-dir; o take.sh define a pasta" >&2
            exit 1 ;;
    esac
done
if ! printf '%s\n' "${args[@]}" | grep -qx -- "--task"; then
    if ! printf '%s\n' "${args[@]}" | grep -q -- "^--task="; then
        args=(--task "$TASK" "${args[@]}")
    fi
fi

if [ ! -t 0 ]; then
    echo "[take] ERRO: precisa de um terminal interativo (via ssh use 'ssh -t')" >&2
    exit 1
fi

# ---- 1. nome --------------------------------------------------------------
NAME="${TAKE_NAME:-}"
while [ -z "$NAME" ]; do
    read -r -p "[take] nome do take (ex.: damping_portico, vx03_reto): " NAME
done
SAFE_NAME="$(printf '%s' "$NAME" | tr ' /' '__' | tr -cd 'A-Za-z0-9._-')"
[ -z "$SAFE_NAME" ] && SAFE_NAME="take"

DAY="$(date +%F)"
DIR="$TAKES_DIR/$DAY/$(date +%H%M%S)_$SAFE_NAME"
mkdir -p "$DIR" || exit 1
echo "[take] pasta: $DIR"

# ---- 2. meta --------------------------------------------------------------
T_START="$(date +%s)"
{
    echo "take:      $NAME"
    echo "inicio:    $(date '+%F %T %z')"
    echo "maquina:   $(hostname)  usuario: $(id -un)"
    echo "modo:      $([ $MUJOCO -eq 1 ] && echo mujoco || echo robo)"
    echo "task:      $TASK"
    echo "commit:    $(git rev-parse HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
    echo "comando:   $PYTHON scripts/deploy.py ${args[*]} --log-dir $DIR"
    echo "modificados (vazio = codigo identico ao commit):"
    git status --short --untracked-files=no 2>/dev/null | sed 's/^/    /'
} > "$DIR/meta.txt"

# ---- 3. gravador (só no robô) ----------------------------------------------
LOGGER_PID=""
if [ $MUJOCO -eq 0 ]; then
    if "$PYTHON" -c "import rclpy, booster_interface.msg" 2>/dev/null; then
        "$PYTHON" scripts/log_deploy.py --task "$TASK" --out "$DIR/log" \
            > "$DIR/logger.txt" 2>&1 &
        LOGGER_PID=$!
        sleep 1.5
        if ! kill -0 "$LOGGER_PID" 2>/dev/null; then
            echo "[take] AVISO: o log_deploy.py morreu ao iniciar:" >&2
            tail -n 5 "$DIR/logger.txt" >&2
            LOGGER_PID=""
        else
            echo "[take] gravador /low_state + /joint_ctrl rodando (pid $LOGGER_PID)"
        fi
    else
        echo "[take] AVISO: rclpy/booster_interface indisponiveis (faltou o"
        echo "       'source /opt/booster/BoosterRos2Interface/install/setup.bash'?)."
        echo "       Sem log.npz; cmd.csv, events.csv e deploy.txt continuam."
    fi
    if [ -z "$LOGGER_PID" ]; then
        read -r -p "[take] continuar sem o gravador? [s/N] " ok
        case "$ok" in s|S|y|Y) ;; *) echo "[take] abortado"; rm -rf "$DIR"; exit 1 ;; esac
        echo "gravador:  NAO (log.npz ausente)" >> "$DIR/meta.txt"
    fi
fi

stop_logger() {
    [ -z "$LOGGER_PID" ] && return
    # um segundo a mais pega o fim do movimento (a queda em damping, a
    # transicao para kWalking) depois que o deploy ja saiu
    sleep 1
    kill -TERM "$LOGGER_PID" 2>/dev/null
    for _ in $(seq 1 100); do
        kill -0 "$LOGGER_PID" 2>/dev/null || break
        sleep 0.1
    done
    kill -0 "$LOGGER_PID" 2>/dev/null && kill -KILL "$LOGGER_PID" 2>/dev/null
    wait "$LOGGER_PID" 2>/dev/null
    LOGGER_PID=""
}
# se o proprio take.sh for interrompido, nao deixar o gravador orfao
trap 'stop_logger' EXIT

# ---- 4. deploy --------------------------------------------------------------
CMD=("$PYTHON" scripts/deploy.py "${args[@]}" --log-dir "$DIR")
CMD_STR="$(printf '%q ' "${CMD[@]}")"
echo "[take] iniciando: $CMD_STR"
echo
# Ctrl-C vai para o deploy.py (dentro do pty do `script`); o take.sh segue
trap ':' INT
script -q -f -e -c "$CMD_STR" "$DIR/deploy.txt"
RC=$?
trap - INT
stty sane 2>/dev/null
echo
echo "[take] deploy.py terminou (codigo $RC)"

stop_logger

# ---- 5. observações e fechamento -------------------------------------------
echo
echo "[take] observacoes do take: o que aconteceu, o que mudou, o que observar."
echo "       Termine com uma linha contendo so '.' (ou Ctrl-D)."
{
    echo "# $NAME"
    echo
    echo "- data: $(date '+%F %T')"
    echo "- pasta: $DIR"
    echo "- codigo de saida do deploy: $RC"
    echo
    echo "## Observacoes"
    echo
} > "$DIR/notes.md"
while IFS= read -r -p "> " line; do
    [ "$line" = "." ] && break
    printf '%s\n' "$line" >> "$DIR/notes.md"
done
echo

# checkpoint efetivo: o caminho que a policy imprimiu, senao o --checkpoint
CKPT_PATH="$(tr -d '\r' < "$DIR/deploy.txt" 2>/dev/null \
    | sed -n 's/^\[mimickit_steering\] policy: //p' | tail -n 1)"
if [ -z "$CKPT_PATH" ] && [ -n "$CKPT_ARG" ]; then
    for base in "" tasks/*/; do
        if [ -f "$base$CKPT_ARG" ]; then CKPT_PATH="$base$CKPT_ARG"; break; fi
    done
fi
{
    echo "fim:       $(date '+%F %T %z')  ($(( $(date +%s) - T_START )) s)"
    echo "saida:     $RC"
    if [ -n "$CKPT_PATH" ] && [ -f "$CKPT_PATH" ]; then
        echo "checkpoint: $CKPT_PATH"
        echo "md5:       $(md5sum "$CKPT_PATH" | cut -d' ' -f1)"
    else
        echo "checkpoint: (nao resolvido; ver deploy.txt)"
    fi
} >> "$DIR/meta.txt"

echo "[take] arquivos:"
ls -la "$DIR" | tail -n +2 | sed 's/^/    /'
[ -f "$DIR/log.txt" ] && echo "[take] relatorio do gravador: $DIR/log.txt"
echo "[take] pronto: $DIR"
