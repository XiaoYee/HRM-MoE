#!/usr/bin/env bash
set -euo pipefail
set -x

repo_dir="${HRM_REPO_DIR:-${repo_dir:-/mnt/shared-storage-user/quxiaoye/HRM-Text}}"
mode="${HRM_MODE:-${mode:-env_check}}"
config_name="${HRM_CONFIG_NAME:-${config_name:-cfg_pretrain}}"
data_path="${HRM_DATA_PATH:-${data_path:-}}"
checkpoint_path="${HRM_CHECKPOINT_PATH:-${checkpoint_path:-}}"
resume_from="${HRM_RESUME_FROM:-${resume_from:-}}"
arch_size="${HRM_ARCH_SIZE:-${arch_size:-}}"
global_batch_size="${HRM_GLOBAL_BATCH_SIZE:-${global_batch_size:-}}"
extra_args="${HRM_EXTRA_ARGS:-${extra_args:-}}"
bootstrap="${HRM_BOOTSTRAP:-${bootstrap:-0}}"
sample_before_train="${HRM_SAMPLE_BEFORE_TRAIN:-${sample_before_train:-0}}"

cd "${repo_dir}"

log_dir="${HRM_LOG_DIR:-${repo_dir}/rjob_logs}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${MLP_TASK_NAME:-${mode}}_${RJOB_TASK_INDEX:-0}_${NODE_RANK:-0}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${log_file}") 2>&1

export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-${repo_dir}/.hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${repo_dir}/wandb}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "HRM log file: ${log_file}"

if [[ "${bootstrap}" == "1" || "${bootstrap}" == "true" ]]; then
  python -m pip install --no-cache-dir \
    -i "${PIP_INDEX_URL:-http://mirrors.i.h.pjlab.org.cn/pypi/simple/}" \
    --trusted-host "${PIP_TRUSTED_HOST:-mirrors.i.h.pjlab.org.cn}" \
    -r docker/requirements/runtime_overlay.txt
fi

run_env_check() {
  python - <<'PY'
import importlib
import os
import sys

import torch

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))

required = os.environ.get(
    "HRM_REQUIRED_IMPORTS",
    "flash_attn_3 hydra omegaconf pydantic transformers wandb coolname numpy numba yaml safetensors",
).split()
optional = os.environ.get("HRM_OPTIONAL_IMPORTS", "datasets lm_eval math_verify vllm").split()

for name in required:
    importlib.import_module(name)
    print("import_ok", name)

for name in optional:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print("optional_missing", name, repr(exc))
    else:
        print("optional_import_ok", name)

print("NODE_COUNT", os.environ.get("NODE_COUNT"))
print("NODE_RANK", os.environ.get("NODE_RANK"))
print("MASTER_ADDR", os.environ.get("MASTER_ADDR"))
print("PROC_PER_NODE", os.environ.get("PROC_PER_NODE"))
PY
}

append_if_set() {
  local -n append_target_array="$1"
  local key="$2"
  local value="$3"

  if [[ -n "${value}" ]]; then
    append_target_array+=("${key}=${value}")
  fi
}

build_train_args() {
  local -n target_array="$1"

  if [[ "${mode}" == "sft" ]]; then
    target_array+=(--config-name cfg_sft)
  elif [[ "${config_name}" != "cfg_pretrain" ]]; then
    target_array+=(--config-name "${config_name}")
  fi

  if [[ -n "${arch_size}" ]]; then
    target_array+=("arch/size@arch=${arch_size}")
  fi

  append_if_set target_array "data.path" "${data_path}"
  append_if_set target_array "global_batch_size" "${global_batch_size}"

  if [[ -n "${checkpoint_path}" ]]; then
    target_array+=("+checkpoint_path=${checkpoint_path}")
  fi

  if [[ "${mode}" == "sft" ]]; then
    if [[ -z "${resume_from}" || -z "${data_path}" || -z "${checkpoint_path}" ]]; then
      echo "SFT requires resume_from, data_path, and checkpoint_path." >&2
      exit 2
    fi
    target_array+=("resume_from=${resume_from}")
    if [[ -n "${weights_only_resume_from_ema:-}" ]]; then
      target_array+=("weights_only_resume_from_ema=${weights_only_resume_from_ema}")
    fi
  fi

  if [[ -n "${extra_args}" ]]; then
    # Hydra overrides should not contain spaces. Use this for simple key=value args.
    read -r -a parsed_extra_args <<< "${extra_args}"
    target_array+=("${parsed_extra_args[@]}")
  fi
}

run_torchrun() {
  local train_args=()
  build_train_args train_args

  local nproc_per_node="${PROC_PER_NODE:-8}"

  if [[ -n "${NODE_COUNT:-}" && -n "${NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]]; then
    torchrun \
      --nnodes="${NODE_COUNT}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --nproc_per_node="${nproc_per_node}" \
      pretrain.py "${train_args[@]}"
  else
    torchrun --nproc_per_node="${nproc_per_node}" pretrain.py "${train_args[@]}"
  fi
}

prepare_sample_if_requested() {
  if [[ "${sample_before_train}" != "1" && "${sample_before_train}" != "true" ]]; then
    return
  fi

  if [[ -z "${data_path}" ]]; then
    echo "sample_before_train requires data_path/SAMPLED_OUTPUT." >&2
    exit 2
  fi

  DATA_PREP_STAGE=sample \
  SAMPLED_OUTPUT="${data_path}" \
  ANALYTICS_PATH="${ANALYTICS_PATH:-${repo_dir}/data_prep_logs/show_analytics_${MLP_TASK_NAME:-pretrain}_${RJOB_TASK_INDEX:-0}_${NODE_RANK:-0}.md}" \
    bash scripts/hrm_prepare_data_entrypoint.sh
}

case "${mode}" in
  env_check)
    run_env_check
    ;;
  prepare_data)
    exec bash scripts/hrm_prepare_data_entrypoint.sh
    ;;
  pretrain|sft)
    if [[ "${mode}" == "pretrain" ]]; then
      prepare_sample_if_requested
    fi
    run_torchrun
    ;;
  *)
    echo "Unknown HRM_MODE: ${mode}" >&2
    exit 2
    ;;
esac
