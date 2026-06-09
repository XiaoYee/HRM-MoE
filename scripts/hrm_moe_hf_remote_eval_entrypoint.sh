#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HRM_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8}"
cd "${repo_dir}"

log_dir="${repo_dir}/rjob_logs"
mkdir -p "${log_dir}" || true
log_file="${log_dir}/${MLP_TASK_NAME:-hrm_moe_hf_remote}_${RJOB_TASK_INDEX:-0}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${log_file}") 2>&1

echo "[hf_remote_eval] repo_dir=${repo_dir}"
echo "[hf_remote_eval] date=$(date --iso-8601=seconds)"
echo "[hf_remote_eval] cuda_visible=${CUDA_VISIBLE_DEVICES:-}"

if [[ "${HRM_BOOTSTRAP:-1}" == "1" ]]; then
  python - <<'PY'
import importlib.util
import subprocess
import sys

packages = {
    "transformers": "transformers",
    "safetensors": "safetensors",
    "datasets": "datasets",
    "tqdm": "tqdm",
    "math_verify": "math-verify",
}
missing = [pip_name for module_name, pip_name in packages.items() if importlib.util.find_spec(module_name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY
fi

model_id="${HRM_HF_REMOTE_MODEL_ID:-Xiaoye08/HRM-MoE}"
mode="${HRM_HF_REMOTE_EVAL_MODE:-usage}"
output_dir="${HRM_HF_REMOTE_EVAL_OUTPUT_DIR:-${repo_dir}/rjob_logs/hrm_moe_hf_remote_eval}"
max_new_tokens="${HRM_HF_REMOTE_MAX_NEW_TOKENS:-256}"
num_gpus="${MLP_WORKER_GPU:-}"

if [[ -z "${num_gpus}" ]]; then
  num_gpus="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi

mkdir -p "${output_dir}"

case "${mode}" in
  usage)
    python scripts/hrm_moe_hf_remote_eval.py \
      --mode usage \
      --model-id "${model_id}" \
      --output-dir "${output_dir}" \
      --max-new-tokens "${max_new_tokens}" \
      --device cuda
    ;;
  gsm8k)
    echo "[hf_remote_eval] running full GSM8k with ${num_gpus} shards"
    pids=()
    for gpu_idx in $(seq 0 "$((num_gpus - 1))"); do
      (
        export CUDA_VISIBLE_DEVICES="${gpu_idx}"
        python scripts/hrm_moe_hf_remote_eval.py \
          --mode gsm8k \
          --model-id "${model_id}" \
          --output-dir "${output_dir}" \
          --max-new-tokens "${max_new_tokens}" \
          --shard-index "${gpu_idx}" \
          --shard-count "${num_gpus}" \
          --device cuda
      ) &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    python scripts/hrm_moe_hf_remote_eval.py \
      --mode aggregate \
      --model-id "${model_id}" \
      --output-dir "${output_dir}"
    ;;
  *)
    echo "unknown HRM_HF_REMOTE_EVAL_MODE=${mode}" >&2
    exit 2
    ;;
esac

echo "[hf_remote_eval] done date=$(date --iso-8601=seconds)"
