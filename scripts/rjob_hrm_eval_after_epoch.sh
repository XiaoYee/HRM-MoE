#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${repo_dir:-/mnt/shared-storage-user/quxiaoye/HRM-Text}"
ckpt_path="${ckpt_path:-${HRM_EVAL_CKPT_PATH:-}}"
epoch="${epoch:-${ckpt_epoch:-${HRM_EVAL_CKPT_EPOCH:-1}}}"
num_gpus="${num_gpus:-8}"
batch_size="${batch_size:-16}"
required_carry_count="${required_carry_count:-16}"
poll_interval="${poll_interval:-300}"
stable_checks="${stable_checks:-3}"
stable_interval="${stable_interval:-60}"
force_submit="${force_submit:-false}"
submit_dry_run="${submit_dry_run:-${dry_run:-false}}"
entrypoint="${entrypoint:-scripts/hrm_eval_fanout_entrypoint.sh}"
eval_config="${eval_config:-${HRM_EVAL_CONFIG:-${repo_dir}/evaluation/config/hrm_benchmarking.yaml}}"

if [[ -z "${ckpt_path}" ]]; then
  echo "Usage: ckpt_path=/path/to/checkpoint/run [ckpt_epoch=1] bash scripts/rjob_hrm_eval_after_epoch.sh" >&2
  exit 2
fi

if [[ ! "${epoch}" =~ ^[0-9]+$ ]] || (( epoch < 1 )); then
  echo "epoch/ckpt_epoch must be a positive integer." >&2
  exit 2
fi

if [[ ! "${num_gpus}" =~ ^[0-9]+$ ]] || (( num_gpus < 1 || num_gpus > 8 )); then
  echo "num_gpus must be an integer from 1 to 8 for eval." >&2
  exit 2
fi

if [[ ! "${required_carry_count}" =~ ^[0-9]+$ ]] || (( required_carry_count < 1 )); then
  echo "required_carry_count must be a positive integer." >&2
  exit 2
fi

if [[ ! "${stable_checks}" =~ ^[0-9]+$ ]] || (( stable_checks < 1 )); then
  echo "stable_checks must be a positive integer." >&2
  exit 2
fi

if [[ ! "${stable_interval}" =~ ^[0-9]+$ ]] || (( stable_interval < 1 )); then
  echo "stable_interval must be a positive integer." >&2
  exit 2
fi

if [[ ! "${poll_interval}" =~ ^[0-9]+$ ]] || (( poll_interval < 1 )); then
  echo "poll_interval must be a positive integer." >&2
  exit 2
fi

sanitize_name() {
  local value="$1"
  value="${value//[^[:alnum:]._-]/-}"
  printf "%s" "${value}"
}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S %z)T] %s\n' -1 "$*"
}

ckpt_base="$(basename "${ckpt_path}")"
ckpt_tag="$(sanitize_name "${ckpt_base#hrm-pretrain-}")"
ckpt_tag="${ckpt_tag:0:32}"
job_name="${job_name:-hrm-eval-e${epoch}-all-${num_gpus}g-${ckpt_tag}}"
marker_dir="${marker_dir:-${repo_dir}/rjob_logs}"
marker_file="${marker_file:-${marker_dir}/.${job_name}.submitted}"
checkpoint_dir="${ckpt_path}/fsdp2_epoch_${epoch}"

checkpoint_ready_once() {
  [[ -f "${ckpt_path}/all_config.yaml" ]] || return 1
  [[ -f "${ckpt_path}/train_metadata.yaml" ]] || return 1
  [[ -d "${checkpoint_dir}" ]] || return 1
  [[ -f "${checkpoint_dir}/.metadata" ]] || return 1

  local carry_count
  carry_count="$(find "${ckpt_path}" -maxdepth 1 -type f -name "carry_epoch_${epoch}.*.pt" 2>/dev/null | wc -l)"
  (( carry_count >= required_carry_count )) || return 1
}

checkpoint_fingerprint() {
  {
    find "${checkpoint_dir}" -type f -printf 'ckpt/%P\t%s\t%T@\n' 2>/dev/null | sort
    find "${ckpt_path}" -maxdepth 1 -type f -name "carry_epoch_${epoch}.*.pt" -printf 'carry/%f\t%s\t%T@\n' 2>/dev/null | sort
  } | sha256sum | awk '{print $1}'
}

wait_for_stable_checkpoint() {
  local last_fingerprint=""
  local stable_count=0
  local carry_count=0

  while true; do
    if checkpoint_ready_once; then
      last_fingerprint=""
      stable_count=0

      while checkpoint_ready_once; do
        local fingerprint
        fingerprint="$(checkpoint_fingerprint)"
        carry_count="$(find "${ckpt_path}" -maxdepth 1 -type f -name "carry_epoch_${epoch}.*.pt" 2>/dev/null | wc -l)"

        if [[ "${fingerprint}" == "${last_fingerprint}" ]]; then
          stable_count=$((stable_count + 1))
        else
          last_fingerprint="${fingerprint}"
          stable_count=1
        fi

        log "Checkpoint epoch ${epoch} candidate stable ${stable_count}/${stable_checks}; carry files: ${carry_count}/${required_carry_count}."
        if (( stable_count >= stable_checks )); then
          return 0
        fi

        sleep "${stable_interval}"
      done

      log "Checkpoint epoch ${epoch} changed while checking stability; resuming polling."
    else
      carry_count="$(find "${ckpt_path}" -maxdepth 1 -type f -name "carry_epoch_${epoch}.*.pt" 2>/dev/null | wc -l || true)"
      log "Waiting for ${checkpoint_dir}/.metadata and ${required_carry_count} carry files; currently carry files: ${carry_count}."
    fi

    sleep "${poll_interval}"
  done
}

submit_eval() {
  local submit_env=(
    "ckpt_path=${ckpt_path}"
    "ckpt_epoch=${epoch}"
    "num_gpus=${num_gpus}"
    "job_name=${job_name}"
    "batch_size=${batch_size}"
    "entrypoint=${entrypoint}"
    "eval_config=${eval_config}"
  )

  if [[ -n "${run_only:-}" ]]; then
    submit_env+=("run_only=${run_only}")
  fi

  if [[ -n "${eval_extra_args:-}" ]]; then
    submit_env+=("eval_extra_args=${eval_extra_args}")
  fi

  if [[ -n "${eval_max_parallel:-}" ]]; then
    submit_env+=("eval_max_parallel=${eval_max_parallel}")
  fi

  if [[ -n "${eval_data_dir:-${HRM_EVAL_DATA_DIR:-}}" ]]; then
    submit_env+=("eval_data_dir=${eval_data_dir:-${HRM_EVAL_DATA_DIR:-}}")
  fi

  if [[ -n "${ckpt_use_ema:-}" ]]; then
    submit_env+=("ckpt_use_ema=${ckpt_use_ema}")
  fi

  if [[ "${submit_dry_run}" == "true" ]]; then
    submit_env+=("dry_run=true")
  fi

  log "Submitting eval job ${job_name} for ${ckpt_path} epoch ${epoch} on ${num_gpus} GPU(s)."
  env "${submit_env[@]}" bash "${script_dir}/rjob_hrm_eval.sh"
}

mkdir -p "${marker_dir}"

if [[ -f "${marker_file}" ]]; then
  log "Marker already exists; eval was already submitted: ${marker_file}"
  exit 0
fi

log "Watching checkpoint root: ${ckpt_path}"
log "Target epoch: ${epoch}; eval job: ${job_name}; entrypoint: ${entrypoint}"

if [[ "${force_submit}" == "true" ]]; then
  log "force_submit=true; skipping checkpoint wait."
else
  wait_for_stable_checkpoint
fi

submit_eval

if [[ "${submit_dry_run}" == "true" ]]; then
  log "submit_dry_run=true; not writing marker."
else
  {
    printf 'submitted_at=%(%Y-%m-%d %H:%M:%S %z)T\n' -1
    printf 'job_name=%s\n' "${job_name}"
    printf 'ckpt_path=%s\n' "${ckpt_path}"
    printf 'ckpt_epoch=%s\n' "${epoch}"
    printf 'num_gpus=%s\n' "${num_gpus}"
    printf 'batch_size=%s\n' "${batch_size}"
    printf 'entrypoint=%s\n' "${entrypoint}"
  } > "${marker_file}.tmp"
  mv "${marker_file}.tmp" "${marker_file}"
  log "Wrote submission marker: ${marker_file}"
fi
