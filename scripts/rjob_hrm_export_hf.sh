#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-env_check}"
job_name="${job_name:-hrm-moe-export-hf}"
num_gpus="${num_gpus:-0}"
memory="${memory:-200000}"
cpu="${cpu:-32}"
auto_restart="${auto_restart:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
rdma="${rdma:-false}"
host_network="${host_network:-false}"
gang_start="${gang_start:-false}"
share_host_shm="${share_host_shm:-false}"
repo_dir="${repo_dir:-/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8}"
entrypoint="${entrypoint:-scripts/hrm_export_hf_entrypoint.sh}"
bootstrap="${bootstrap:-0}"

export mode job_name num_gpus memory cpu auto_restart WANDB_MODE rdma host_network gang_start share_host_shm
export repo_dir entrypoint bootstrap
export HRM_EXPORT_CKPT_PATH="${ckpt_path:-${HRM_EXPORT_CKPT_PATH:-}}"
export HRM_EXPORT_CKPT_EPOCH="${ckpt_epoch:-${HRM_EXPORT_CKPT_EPOCH:-4}}"
export HRM_EXPORT_OUT_DIR="${out_dir:-${HRM_EXPORT_OUT_DIR:-/mnt/shared-storage-user/quxiaoye/hf_hrm_moe_export}}"
export HRM_EXPORT_EXTRA_ARGS="${export_extra_args:-${HRM_EXPORT_EXTRA_ARGS:-}}"

exec bash "${script_dir}/rjob_hrm_common.sh"
