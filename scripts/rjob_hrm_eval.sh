#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-eval}"
num_gpus="${num_gpus:-1}"
job_name="${job_name:-hrm-eval-${num_gpus}g}"
memory="${memory:-256000}"
cpu="${cpu:-16}"
auto_restart="${auto_restart:-false}"
rdma="${rdma:-false}"
host_network="${host_network:-true}"
gang_start="${gang_start:-false}"
share_host_shm="${share_host_shm:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
bootstrap="${bootstrap:-${HRM_BOOTSTRAP:-1}}"
entrypoint="${entrypoint:-scripts/hrm_eval_entrypoint.sh}"
eval_config="${eval_config:-${HRM_EVAL_CONFIG:-/mnt/shared-storage-user/quxiaoye/HRM-Text/evaluation/config/hrm_benchmarking.yaml}}"
eval_workdir="${eval_workdir:-${HRM_EVAL_WORKDIR:-/mnt/shared-storage-user/quxiaoye/data_io/tokenizer}}"
export HF_ENDPOINT="${hf_endpoint:-${HRM_HF_ENDPOINT:-https://huggingface.co}}"
export HF_HUB_ETAG_TIMEOUT="${hf_hub_etag_timeout:-${HF_HUB_ETAG_TIMEOUT:-60}}"
export HF_HUB_DOWNLOAD_TIMEOUT="${hf_hub_download_timeout:-${HF_HUB_DOWNLOAD_TIMEOUT:-120}}"

if [[ -z "${ckpt_path:-}" ]]; then
  echo "Usage: ckpt_path=/path/to/checkpoints/run_dir [ckpt_epoch=1] [run_only='[GSM8k,MATH]'] bash scripts/rjob_hrm_eval.sh" >&2
  exit 2
fi

cluster_proxy_default="http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
cluster_no_proxy_default="localhost,127.0.0.1,::1,.pjlab.org.cn,.svc,.svc.pjlab.local"
export RJOB_HTTP_PROXY="${rjob_http_proxy:-${RJOB_HTTP_PROXY:-${http_proxy:-${cluster_proxy_default}}}}"
export RJOB_HTTPS_PROXY="${rjob_https_proxy:-${RJOB_HTTPS_PROXY:-${https_proxy:-${RJOB_HTTP_PROXY}}}}"
export RJOB_NO_PROXY="${rjob_no_proxy:-${RJOB_NO_PROXY:-${cluster_no_proxy_default}}}"

export mode num_gpus job_name memory cpu auto_restart rdma host_network gang_start share_host_shm WANDB_MODE bootstrap entrypoint
export ckpt_path ckpt_epoch ckpt_use_ema eval_config run_only batch_size eval_extra_args eval_workdir
exec bash "${script_dir}/rjob_hrm_common.sh"
