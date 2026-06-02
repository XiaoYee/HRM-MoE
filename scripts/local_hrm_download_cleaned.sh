#!/usr/bin/env bash
set -euo pipefail

data_io_dir="${DATA_IO_DIR:-/mnt/shared-storage-user/quxiaoye/data_io}"
cleaned_repo="${CLEANED_REPO:-sapientinc/HRM-Text-data-io-cleaned-20260515}"
download_workers="${download_workers:-${DOWNLOAD_WORKERS:-4}}"
cluster_proxy_default="http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
cluster_no_proxy_default="localhost,127.0.0.1,::1,.pjlab.org.cn,.svc,.svc.pjlab.local"

# Do not inherit the login shell's hf-mirror endpoint by default; this Hub
# client can list repos through the mirror but fails on file downloads.
export HF_ENDPOINT="${hf_endpoint:-${HRM_HF_ENDPOINT:-https://huggingface.co}}"
export HF_HUB_ETAG_TIMEOUT="${hf_hub_etag_timeout:-${HF_HUB_ETAG_TIMEOUT:-60}}"
export HF_HUB_DOWNLOAD_TIMEOUT="${hf_hub_download_timeout:-${HF_HUB_DOWNLOAD_TIMEOUT:-120}}"

# Tmux sessions may inherit stale localhost proxies. Use the cluster proxy for
# external downloads and keep internal PJLab services direct.
unset ALL_PROXY all_proxy
export HTTP_PROXY="${local_http_proxy:-${HTTP_PROXY_OVERRIDE:-${cluster_proxy_default}}}"
export HTTPS_PROXY="${local_https_proxy:-${HTTPS_PROXY_OVERRIDE:-${HTTP_PROXY}}}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${local_no_proxy:-${NO_PROXY_OVERRIDE:-${cluster_no_proxy_default}}}"
export no_proxy="${NO_PROXY}"

mkdir -p "${data_io_dir}"
cd "${data_io_dir}"

hf download "${cleaned_repo}" \
  --repo-type dataset \
  --local-dir "${data_io_dir}" \
  --include 'data/**' 'data_clustered/**' \
  --max-workers "${download_workers}"
