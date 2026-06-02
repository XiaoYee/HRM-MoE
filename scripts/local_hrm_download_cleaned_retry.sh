#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HRM_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text}"
sleep_seconds="${RETRY_SLEEP_SECONDS:-900}"
max_attempts="${MAX_ATTEMPTS:-0}"
attempt=1

cd "${repo_dir}"

while true; do
  echo "download_attempt=${attempt} $(date)"
  if bash scripts/local_hrm_download_cleaned.sh; then
    echo "download_exit=0"
    exit 0
  fi

  status=$?
  echo "download_attempt_failed=${attempt} status=${status} $(date)" >&2
  if (( max_attempts > 0 && attempt >= max_attempts )); then
    echo "download_exit=${status}"
    exit "${status}"
  fi

  attempt=$((attempt + 1))
  sleep "${sleep_seconds}"
done
