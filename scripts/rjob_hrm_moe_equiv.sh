#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-moe_equiv}"
num_gpus="${num_gpus:-8}"
job_name="${job_name:-hrm-moe-equiv}"
entrypoint="${entrypoint:-scripts/hrm_moe_equiv_entrypoint.sh}"
moe_equiv_script="${moe_equiv_script:-scripts/test_moe_ep_distributed_equivalence.py}"
moe_equiv_distributed="${moe_equiv_distributed:-1}"
auto_restart="${auto_restart:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
memory="${memory:-256000}"
cpu="${cpu:-32}"
rdma="${rdma:-false}"
gang_start="${gang_start:-false}"

export mode num_gpus job_name entrypoint moe_equiv_script moe_equiv_distributed
export auto_restart WANDB_MODE memory cpu rdma gang_start
exec bash "${script_dir}/rjob_hrm_common.sh"
