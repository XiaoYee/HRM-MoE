#!/usr/bin/env bash
set -euo pipefail
set -x

repo_dir="${HRM_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8}"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

cd "${repo_dir}"

python - <<'PY'
import importlib.util
import subprocess
import sys

missing = [name for name in ("safetensors", "transformers", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "transformers", "pyyaml"])
PY

python conversion/export_dcp_to_hf_safetensors.py \
  --ckpt-path "${HRM_EXPORT_CKPT_PATH:?}" \
  --ckpt-epoch "${HRM_EXPORT_CKPT_EPOCH:?}" \
  --out-dir "${HRM_EXPORT_OUT_DIR:?}" \
  ${HRM_EXPORT_EXTRA_ARGS:-}
