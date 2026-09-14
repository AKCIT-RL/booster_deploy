cd /home/jgabriel/Projects/Humanoids/booster_deploy || exit 1
export MIMICKIT_PATH=${MIMICKIT_PATH:-../dev_tsinghua/MimicKit/mimickit}
# Override to keep a run from overwriting an earlier one (e.g. the docs/09 baseline).
VIAB_OUT="${VIAB_OUT:-/tmp/viab}"
mkdir -p "$VIAB_OUT"
run() {  # run <tag> <extra-args...>
  tag="$1"; shift
  .venv/bin/python scripts/sim_viability.py --duration 60 "$@" \
      --json "${VIAB_OUT}/${tag}.json" --out "${VIAB_OUT}/${tag}.npz" >/dev/null 2>&1
  echo "  ${tag} -> $([ -f "${VIAB_OUT}/${tag}.json" ] && echo ok || echo FALHOU)"
}
echo "[1/3] matriz de modelos de atuador (estado verdadeiro)"
for v in 0.5 1.0; do
  run "m_urdf_vx${v}"       --vx $v --effort urdf
  run "m_urdf_tn_vx${v}"    --vx $v --effort urdf    --tn
  run "m_derated_vx${v}"    --vx $v --effort derated
  run "m_derated_tn_vx${v}" --vx $v --effort derated --tn
  run "m_catalog_vx${v}"    --vx $v --effort catalog
  run "m_catalog_tn_vx${v}" --vx $v --effort catalog --tn
done
echo "[2/3] estado zerado (matriz de observabilidade)"
for v in 0.5 1.0; do
  run "z_urdf_vx${v}"    --vx $v --effort urdf    --degrade zero_root
  run "z_derated_vx${v}" --vx $v --effort derated --degrade zero_root
done
echo "[3/3] varredura de velocidade"
for v in 0.3 0.5 0.7 1.0 1.3 1.6 2.0 2.5; do
  run "s_catalog_tn_vx${v}" --vx $v --effort catalog --tn
  run "s_urdf_vx${v}"       --vx $v --effort urdf
done
echo SWEEP_DONE
