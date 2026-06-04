#!/usr/bin/env bash
set -euo pipefail
set -x

repo_dir="${HRM_REPO_DIR:-${repo_dir:-/mnt/shared-storage-user/quxiaoye/HRM-Text}}"
bootstrap="${HRM_BOOTSTRAP:-${bootstrap:-1}}"
test_script="${HRM_MOE_EQUIV_SCRIPT:-scripts/test_moe_ep_distributed_equivalence.py}"
distributed="${HRM_MOE_EQUIV_DISTRIBUTED:-1}"
extra_args="${HRM_EXTRA_ARGS:-${extra_args:-}}"

cd "${repo_dir}"

log_dir="${HRM_LOG_DIR:-${repo_dir}/rjob_logs}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${MLP_TASK_NAME:-moe_equiv}_${RJOB_TASK_INDEX:-0}_${NODE_RANK:-0}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${log_file}") 2>&1

echo "HRM MoE equivalence log file: ${log_file}"

if [[ "${bootstrap}" == "1" || "${bootstrap}" == "true" ]]; then
  python -m pip install --no-cache-dir \
    -i "${PIP_INDEX_URL:-http://mirrors.i.h.pjlab.org.cn/pypi/simple/}" \
    --trusted-host "${PIP_TRUSTED_HOST:-mirrors.i.h.pjlab.org.cn}" \
    -r docker/requirements/runtime_overlay.txt
fi

export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -n "${extra_args}" ]]; then
  read -r -a parsed_extra_args <<< "${extra_args}"
  for assignment in "${parsed_extra_args[@]}"; do
    export "${assignment}"
  done
fi

if [[ "${distributed}" == "1" || "${distributed}" == "true" ]]; then
  nproc_per_node="${PROC_PER_NODE:-8}"
  if [[ -n "${NODE_COUNT:-}" && -n "${NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]]; then
    torchrun \
      --nnodes="${NODE_COUNT}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --nproc_per_node="${nproc_per_node}" \
      "${test_script}"
  else
    torchrun --nproc_per_node="${nproc_per_node}" "${test_script}"
  fi
else
  python "${test_script}"
fi
