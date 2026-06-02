#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-prepare_data}"
stage="${stage:-${DATA_PREP_STAGE:-check}}"
job_name="${job_name:-hrm-data-${stage}}"
auto_restart="${auto_restart:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
tokenizer_impl="${tokenizer_impl:-${TOKENIZER_IMPL:-auto}}"

case "${stage}" in
  check)
    num_gpus="${num_gpus:-0}"
    memory="${memory:-128000}"
    cpu="${cpu:-16}"
    rdma="${rdma:-false}"
    host_network="${host_network:-false}"
    gang_start="${gang_start:-false}"
    share_host_shm="${share_host_shm:-false}"
    ;;
  download_cleaned)
    num_gpus="${num_gpus:-0}"
    memory="${memory:-128000}"
    cpu="${cpu:-16}"
    rdma="${rdma:-false}"
    host_network="${host_network:-false}"
    gang_start="${gang_start:-false}"
    share_host_shm="${share_host_shm:-false}"
    ;;
  tokenize)
    num_gpus="${num_gpus:-0}"
    memory="${memory:-512000}"
    if [[ "${tokenizer_impl}" == "python" ]]; then
      cpu="${cpu:-48}"
    else
      cpu="${cpu:-5}"
    fi
    rdma="${rdma:-false}"
    host_network="${host_network:-false}"
    gang_start="${gang_start:-false}"
    share_host_shm="${share_host_shm:-false}"
    ;;
  sample)
    num_gpus="${num_gpus:-0}"
    memory="${memory:-512000}"
    cpu="${cpu:-64}"
    rdma="${rdma:-false}"
    host_network="${host_network:-false}"
    gang_start="${gang_start:-false}"
    share_host_shm="${share_host_shm:-false}"
    ;;
  all)
    num_gpus="${num_gpus:-8}"
    memory="${memory:-768000}"
    cpu="${cpu:-96}"
    rdma="${rdma:-false}"
    host_network="${host_network:-true}"
    gang_start="${gang_start:-true}"
    share_host_shm="${share_host_shm:-true}"
    ;;
esac

export mode job_name num_gpus memory cpu auto_restart rdma host_network gang_start share_host_shm WANDB_MODE
export DATA_PREP_STAGE="${stage}"
export DATA_IO_DIR="${DATA_IO_DIR:-/mnt/shared-storage-user/quxiaoye/data_io}"
hrm_repo_dir="${repo_dir:-${HRM_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text}}"
export TOKENIZED_PATH="${TOKENIZED_PATH:-${hrm_repo_dir}/data_tokenized_bpe_65k}"
export DATA_EPOCHS="${epochs:-${DATA_EPOCHS:-4}}"
export CONTEXT_SIZE="${context_size:-${CONTEXT_SIZE:-4097}}"
export SAMPLED_OUTPUT="${sampled_output:-${SAMPLED_OUTPUT:-${hrm_repo_dir}/data_sampled_bpe_65k_e${DATA_EPOCHS}_ctx${CONTEXT_SIZE}}}"
export ANALYTICS_PATH="${analytics_path:-${ANALYTICS_PATH:-${hrm_repo_dir}/data_prep_logs/show_analytics_e${DATA_EPOCHS}_ctx${CONTEXT_SIZE}.md}}"
export DOWNLOAD_WORKERS="${download_workers:-${DOWNLOAD_WORKERS:-16}}"
export TOKENIZER_WORKERS="${tokenizer_workers:-${TOKENIZER_WORKERS:-16}}"
export TOKENIZER_BATCH_SIZE="${tokenizer_batch_size:-${TOKENIZER_BATCH_SIZE:-8192}}"
export TOKENIZER_IMPL="${tokenizer_impl}"
export AUTO_INSTALL_RUST="${auto_install_rust:-${AUTO_INSTALL_RUST:-1}}"
export HRM_RUST_HOME="${hrm_rust_home:-${HRM_RUST_HOME:-/mnt/shared-storage-user/quxiaoye/.hrm-rust}}"
export RUSTUP_HOME="${rustup_home:-${RUSTUP_HOME:-${HRM_RUST_HOME}/rustup}}"
export CARGO_HOME="${cargo_home:-${CARGO_HOME:-${HRM_RUST_HOME}/cargo}}"
export RUSTUP_DIST_SERVER="${rustup_dist_server:-${RUSTUP_DIST_SERVER:-https://mirrors.tuna.tsinghua.edu.cn/rustup}}"
export RUSTUP_UPDATE_ROOT="${rustup_update_root:-${RUSTUP_UPDATE_ROOT:-https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup}}"
export CARGO_BUILD_JOBS="${cargo_build_jobs:-${CARGO_BUILD_JOBS:-4}}"
export HF_ENDPOINT="${hf_endpoint:-${HRM_HF_ENDPOINT:-https://huggingface.co}}"
export HF_HUB_ETAG_TIMEOUT="${hf_hub_etag_timeout:-${HF_HUB_ETAG_TIMEOUT:-60}}"
export HF_HUB_DOWNLOAD_TIMEOUT="${hf_hub_download_timeout:-${HF_HUB_DOWNLOAD_TIMEOUT:-120}}"
cluster_proxy_default="http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
cluster_no_proxy_default="localhost,127.0.0.1,::1,.pjlab.org.cn,.svc,.svc.pjlab.local"
export RJOB_HTTP_PROXY="${rjob_http_proxy:-${RJOB_HTTP_PROXY:-${http_proxy:-${cluster_proxy_default}}}}"
export RJOB_HTTPS_PROXY="${rjob_https_proxy:-${RJOB_HTTPS_PROXY:-${https_proxy:-${RJOB_HTTP_PROXY}}}}"
export RJOB_NO_PROXY="${rjob_no_proxy:-${RJOB_NO_PROXY:-${cluster_no_proxy_default}}}"

exec bash "${script_dir}/rjob_hrm_common.sh"
