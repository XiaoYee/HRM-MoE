#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HRM_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text}"
download_session="${DOWNLOAD_SESSION:-hrm_data_download}"
download_log="${DOWNLOAD_LOG:-}"
tokenize_job_name="${TOKENIZE_JOB_NAME:-hrm-data-tokenize}"
poll_seconds="${POLL_SECONDS:-300}"

if [[ -z "${download_log}" ]]; then
  download_log="$(cat "${repo_dir}/local_data_prep_logs/tmux_download_cleaned.logpath")"
fi

cd "${repo_dir}"

while tmux has-session -t "${download_session}" 2>/dev/null; do
  sleep "${poll_seconds}"
done

if ! grep -q 'download_exit=0' "${download_log}"; then
  echo "download_not_successful: ${download_log}" >&2
  exit 1
fi

stage=tokenize \
job_name="${tokenize_job_name}" \
auto_restart=true \
bash scripts/rjob_hrm_prepare_data.sh
