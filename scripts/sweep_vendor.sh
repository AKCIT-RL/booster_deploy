# Booster's own t1_walk policy through the same axes as scripts/sweep_viability.sh,
# plus a rate-control arm on our policy. See docs/12.
cd /home/jgabriel/Projects/Humanoids/booster_deploy || exit 1
export MIMICKIT_PATH=${MIMICKIT_PATH:-../dev_tsinghua/MimicKit/mimickit}
VIAB_OUT="${VIAB_OUT:-/tmp/viab-vendor}"
mkdir -p "$VIAB_OUT"

WALK="tasks/locomotion/robots/t1/models/t1_walk.pt"
OURS="models/t1_wrturn_s1/policy.pt"

run() {  # run <tag> <extra-args...>
  tag="$1"; shift
  .venv/bin/python scripts/sim_viability.py --duration 60 "$@" \
      --json "${VIAB_OUT}/${tag}.json" --out "${VIAB_OUT}/${tag}.npz" >/dev/null 2>&1
  echo "  ${tag} -> $([ -f "${VIAB_OUT}/${tag}.json" ] && echo ok || echo FALHOU)"
}
walk() { tag="$1"; shift; run "$tag" --task t1_walk --checkpoint "$WALK" "$@"; }

echo "[1/5] t1_walk: matriz de modelos de atuador"
for v in 0.5 1.0; do
  walk "w_urdf_vx${v}"       --vx $v --effort urdf
  walk "w_urdf_tn_vx${v}"    --vx $v --effort urdf    --tn
  walk "w_derated_vx${v}"    --vx $v --effort derated
  walk "w_derated_tn_vx${v}" --vx $v --effort derated --tn
  walk "w_catalog_vx${v}"    --vx $v --effort catalog
  walk "w_catalog_tn_vx${v}" --vx $v --effort catalog --tn
done

echo "[2/5] t1_walk: observabilidade (esperado: no-op exato)"
for v in 0.5 1.0; do
  walk "w_obs_none_vx${v}" --vx $v --effort urdf --degrade none
  walk "w_obs_zero_vx${v}" --vx $v --effort urdf --degrade zero_root
done

echo "[3/5] t1_walk: varredura de velocidade (vx_max=1.0; acima e extrapolacao)"
for v in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  walk "w_sweep_urdf_vx${v}"       --vx $v --effort urdf
  walk "w_sweep_catalog_tn_vx${v}" --vx $v --effort catalog --tn
done

echo "[4/5] t1_walk: ganhos da task (200/5) vs padrao de robots/t1.py (f_n=4Hz)"
for v in 0.5 1.0; do
  walk "w_gains_task_vx${v}"    --vx $v --effort urdf --gains task
  walk "w_gains_default_vx${v}" --vx $v --effort urdf --gains default
done

echo "[5/5] nossa politica a 50 Hz (braco de controle de taxa -- ver ressalva em docs/12)"
for v in 0.5 1.0; do
  run "r50_urdf_vx${v}"       --checkpoint "$OURS" --vx $v --effort urdf    --policy-dt 0.02 --decimation 10
  run "r50_derated_tn_vx${v}" --checkpoint "$OURS" --vx $v --effort derated --tn --policy-dt 0.02 --decimation 10
  run "r50_zero_vx${v}"       --checkpoint "$OURS" --vx $v --effort urdf --degrade zero_root --policy-dt 0.02 --decimation 10
done
echo SWEEP_VENDOR_DONE
