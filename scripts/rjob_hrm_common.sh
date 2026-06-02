#!/usr/bin/env bash
set -euo pipefail
set -x

mode="${mode:-${HRM_MODE:-env_check}}"
num_gpus="${num_gpus:-8}"
job_name="${job_name:-hrm-${mode}-${num_gpus}g}"
gpu_group="${gpu_group:-moe_gpu}"
namespace="${namespace:-ailab-moe}"
repo_dir="${repo_dir:-/mnt/shared-storage-user/quxiaoye/HRM-Text}"
image="${image:-registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194}"
memory="${memory:-1200000}"
cpu="${cpu:-128}"
auto_restart="${auto_restart:-true}"
private_machine="${private_machine:-group}"
dry_run="${dry_run:-false}"
rdma="${rdma:-true}"
host_network="${host_network:-true}"
gang_start="${gang_start:-true}"
share_host_shm="${share_host_shm:-true}"
bootstrap="${bootstrap:-1}"

if (( num_gpus == 0 )) && [[ "${mode}" != "env_check" && "${mode}" != "prepare_data" ]]; then
  echo "num_gpus=0 is only supported for env_check and prepare_data." >&2
  exit 2
fi

if (( num_gpus == 0 )); then
  num_nodes=1
  gpu_per_replica=0
elif (( num_gpus % 8 == 0 )); then
  num_nodes=$((num_gpus / 8))
  gpu_per_replica=8
else
  echo "num_gpus must be 0 or a multiple of 8 for this rjob launcher." >&2
  exit 2
fi

rjob_env=(
  -e "DISTRIBUTED_JOB=true"
  -e "HRM_REPO_DIR=${repo_dir}"
  -e "HRM_MODE=${mode}"
  -e "HRM_CONFIG_NAME=${config_name:-cfg_pretrain}"
  -e "HRM_DATA_PATH=${data_path:-}"
  -e "HRM_CHECKPOINT_PATH=${checkpoint_path:-}"
  -e "HRM_RESUME_FROM=${resume_from:-}"
  -e "HRM_ARCH_SIZE=${arch_size:-}"
  -e "HRM_GLOBAL_BATCH_SIZE=${global_batch_size:-}"
  -e "HRM_EXTRA_ARGS=${extra_args:-}"
  -e "HRM_BOOTSTRAP=${bootstrap}"
  -e "HRM_SAMPLE_BEFORE_TRAIN=${sample_before_train:-${HRM_SAMPLE_BEFORE_TRAIN:-}}"
  -e "DATA_PREP_STAGE=${DATA_PREP_STAGE:-}"
  -e "DATA_IO_DIR=${DATA_IO_DIR:-}"
  -e "TOKENIZED_PATH=${TOKENIZED_PATH:-}"
  -e "SAMPLED_OUTPUT=${SAMPLED_OUTPUT:-}"
  -e "ANALYTICS_PATH=${ANALYTICS_PATH:-}"
  -e "DATA_EPOCHS=${DATA_EPOCHS:-}"
  -e "CONTEXT_SIZE=${CONTEXT_SIZE:-}"
  -e "DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-}"
  -e "TOKENIZER_WORKERS=${TOKENIZER_WORKERS:-}"
  -e "TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-}"
  -e "TOKENIZER_IMPL=${TOKENIZER_IMPL:-}"
  -e "AUTO_INSTALL_RUST=${AUTO_INSTALL_RUST:-}"
  -e "HRM_RUST_HOME=${HRM_RUST_HOME:-}"
  -e "RUSTUP_HOME=${RUSTUP_HOME:-}"
  -e "CARGO_HOME=${CARGO_HOME:-}"
  -e "RUSTUP_DIST_SERVER=${RUSTUP_DIST_SERVER:-}"
  -e "RUSTUP_UPDATE_ROOT=${RUSTUP_UPDATE_ROOT:-}"
  -e "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-}"
  -e "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
  -e "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}"
  -e "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
  -e "WANDB_MODE=${WANDB_MODE:-online}"
)

if [[ -n "${RJOB_HTTP_PROXY:-}" ]]; then
  rjob_env+=(-e "http_proxy=${RJOB_HTTP_PROXY}" -e "HTTP_PROXY=${RJOB_HTTP_PROXY}")
fi

if [[ -n "${RJOB_HTTPS_PROXY:-}" ]]; then
  rjob_env+=(-e "https_proxy=${RJOB_HTTPS_PROXY}" -e "HTTPS_PROXY=${RJOB_HTTPS_PROXY}")
fi

if [[ -n "${RJOB_NO_PROXY:-}" ]]; then
  rjob_env+=(-e "no_proxy=${RJOB_NO_PROXY}" -e "NO_PROXY=${RJOB_NO_PROXY}")
fi

if [[ -n "${weights_only_resume_from_ema:-}" ]]; then
  rjob_env+=(-e "weights_only_resume_from_ema=${weights_only_resume_from_ema}")
fi

rjob_submit_args=()
if [[ "${dry_run}" == "true" ]]; then
  rjob_submit_args+=(--dry-run true)
fi

rjob_resource_args=()
if [[ "${rdma}" == "true" ]]; then
  rjob_resource_args+=(
    --custom-resources rdma/mlnx_shared=8
    --custom-resources mellanox.com/mlnx_rdma=1
  )
fi

rjob submit \
  "${rjob_submit_args[@]}" \
  --name="${job_name}" \
  --gpu="${gpu_per_replica}" --memory="${memory}" --cpu="${cpu}" \
  --charged-group="${gpu_group}" \
  --namespace "${namespace}" \
  --private-machine="${private_machine}" \
  -P "${num_nodes}" \
  --image "${image}" \
  --mount=gpfs://gpfs1/quxiaoye:/mnt/shared-storage-user/quxiaoye \
  --mount=gpfs://gpfs1/moegroup:/mnt/shared-storage-user/moegroup \
  --mount=gpfs://gpfs2/moegroup2:/mnt/shared-storage-user/moegroup2 \
  --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
  --host-network="${host_network}" \
  --gang-start="${gang_start}" \
  --share-host-shm="${share_host_shm}" \
  "${rjob_resource_args[@]}" \
  --auto-restart="${auto_restart}" \
  "${rjob_env[@]}" \
  -- bash -lc "cd ${repo_dir} && bash scripts/hrm_entrypoint.sh"
