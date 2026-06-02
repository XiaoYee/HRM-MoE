#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-env_check}"
job_name="${job_name:-hrm-env-check}"
num_gpus="${num_gpus:-0}"
auto_restart="${auto_restart:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
memory="${memory:-256000}"
cpu="${cpu:-32}"
rdma="${rdma:-false}"

if [[ "${num_gpus}" == "0" ]]; then
  host_network="${host_network:-false}"
  gang_start="${gang_start:-false}"
  share_host_shm="${share_host_shm:-false}"
fi

export mode job_name num_gpus auto_restart WANDB_MODE memory cpu rdma host_network gang_start share_host_shm
exec bash "${script_dir}/rjob_hrm_common.sh"
