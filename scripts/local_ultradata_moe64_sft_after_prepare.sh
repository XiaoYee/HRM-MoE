#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MOE_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8}"
data_path="${SFT_DATA_PATH:-/mnt/shared-storage-user/quxiaoye/HRM-Text/data_ultradata_sft_2605_hrm_sft_e5_ctx4097}"
resume_from="${RESUME_FROM:-${repo_dir}/checkpoints/hrm-moe32g-sm16-06050339}"
checkpoint_path="${SFT_CHECKPOINT_PATH:-${repo_dir}/checkpoints/hrm-moe64x8-ultradata-sft-0607}"
job_name="${SFT_JOB_NAME:-hrm-moe64-sft-ultra0607}"
num_gpus="${SFT_NUM_GPUS:-32}"
arch_size="${SFT_ARCH_SIZE:-XL_moe64x8_grouped_triton}"
epochs="${SFT_EPOCHS:-5}"
target_resume_epoch="${SFT_RESUME_EPOCH:-4}"
global_batch_size="${SFT_GLOBAL_BATCH_SIZE:-262144}"
required_carry_count="${REQUIRED_CARRY_COUNT:-32}"
poll_seconds="${POLL_SECONDS:-300}"
stability_seconds="${STABILITY_SECONDS:-30}"
marker_dir="${MARKER_DIR:-${repo_dir}/rjob_logs}"
sft_max_attempts="${SFT_MAX_ATTEMPTS:-3}"

mkdir -p "${marker_dir}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >&2
}

fingerprint_path() {
  local path="$1"
  find "${path}" -type f -printf '%P %s %T@\n' 2>/dev/null | sort
}

wait_stable_path() {
  local path="$1"
  local first second
  first="$(fingerprint_path "${path}")"
  sleep "${stability_seconds}"
  second="$(fingerprint_path "${path}")"
  [[ "${first}" == "${second}" && -n "${second}" ]]
}

job_state() {
  local name="$1"
  local output
  output="$(rjob get job "${name}" 2>&1 || true)"
  if [[ -z "${output}" ]]; then
    echo "MISSING"
    return 0
  fi
  if grep -qiE ": (Succeeded|Completed|Success)" <<<"${output}"; then
    echo "SUCCEEDED"
  elif grep -qiE ": (Failed|Error|Terminated|Killed|Stopped)" <<<"${output}" || grep -qiE "failed': [1-9]" <<<"${output}"; then
    echo "FAILED"
  elif grep -qiE ": (Running|Pending|STARTING|Starting|Created)" <<<"${output}"; then
    echo "RUNNING"
  else
    echo "UNKNOWN"
  fi
}

validate_sft_data() {
  local path="$1"
  python - "${path}" "${epochs}" "${global_batch_size}" "${num_gpus}" <<'PY'
import json
import sys
from pathlib import Path
from numpy.lib import format as npy_format

path = Path(sys.argv[1])
epochs = int(sys.argv[2])
global_batch_size = int(sys.argv[3])
num_gpus = int(sys.argv[4])

def npy_header(file: Path):
    with file.open("rb") as handle:
        version = npy_format.read_magic(handle)
        shape, fortran_order, dtype = npy_format._read_array_header(handle, version)
    return tuple(shape), str(dtype)

required_top = [
    "metadata.json",
    "prepare_summary.json",
    "tokenizer.json",
    "tokenizer_info.json",
    "tokens.npy",
    "inst_start.npy",
    "inst_len.npy",
    "resp_start.npy",
    "resp_len.npy",
]
missing = [name for name in required_top if not (path / name).is_file()]
if missing:
    raise SystemExit(f"missing_top={missing}")

metadata = json.loads((path / "metadata.json").read_text())
summary = json.loads((path / "prepare_summary.json").read_text())
tokenizer_info = json.loads((path / "tokenizer_info.json").read_text())

if metadata.get("source") != "openbmb/UltraData-SFT-2605":
    raise SystemExit(f"unexpected_source={metadata.get('source')!r}")
if metadata.get("max_seq_len") != 4097:
    raise SystemExit(f"unexpected_max_seq_len={metadata.get('max_seq_len')!r}")
if metadata.get("num_samples") != summary.get("num_samples"):
    raise SystemExit("metadata/summary num_samples mismatch")
local_batch_size = global_batch_size // num_gpus
required_local_batch_size = int(metadata["max_seq_len"]) - 1
if global_batch_size % num_gpus != 0:
    raise SystemExit(f"global_batch_size_not_divisible global_batch_size={global_batch_size} num_gpus={num_gpus}")
if local_batch_size < required_local_batch_size:
    raise SystemExit(
        "local_batch_too_small "
        f"global_batch_size={global_batch_size} num_gpus={num_gpus} "
        f"local_batch_size={local_batch_size} required_local_batch_size={required_local_batch_size}"
    )

condition_mapping = tokenizer_info.get("condition_mapping") or {}
for key in ("direct", "synth", "cot"):
    if key not in condition_mapping:
        raise SystemExit(f"missing_condition_mapping={key}")

row_shape, row_dtype = npy_header(path / "inst_start.npy")
if len(row_shape) != 1 or row_shape[0] <= 0 or row_dtype != "int64":
    raise SystemExit(f"bad_inst_start_header shape={row_shape} dtype={row_dtype}")

for name in ("inst_len", "resp_start", "resp_len"):
    shape, dtype = npy_header(path / f"{name}.npy")
    if shape != row_shape or dtype != "int64":
        raise SystemExit(f"bad_{name}_header shape={shape} dtype={dtype}")

tokens_shape, tokens_dtype = npy_header(path / "tokens.npy")
if len(tokens_shape) != 1 or tokens_shape[0] <= 0 or tokens_dtype != "int32":
    raise SystemExit(f"bad_tokens_header shape={tokens_shape} dtype={tokens_dtype}")

for epoch in range(epochs):
    epoch_dir = path / f"epoch_{epoch}"
    if not epoch_dir.is_dir():
        raise SystemExit(f"missing_epoch_dir={epoch_dir}")
    for name in ("inst_start", "inst_len", "resp_start", "resp_len"):
        shape, dtype = npy_header(epoch_dir / f"{name}.npy")
        if shape != row_shape or dtype != "int64":
            raise SystemExit(f"bad_epoch_array epoch={epoch} name={name} shape={shape} dtype={dtype}")

print(
    "sft_data_ok "
    f"samples={metadata.get('num_samples')} "
    f"tokens={metadata.get('num_tokens')} "
    f"max_sample_len={metadata.get('max_sample_len')} "
    f"local_batch_size={local_batch_size} "
    f"skipped_long={summary.get('skipped_long')}"
)
PY
}

latest_stable_resume_epoch() {
  python - "${resume_from}" "${required_carry_count}" "${target_resume_epoch}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = int(sys.argv[2])
target = int(sys.argv[3]) if sys.argv[3] else None

def is_complete(epoch: int) -> bool:
    ckpt = root / f"fsdp2_epoch_{epoch}"
    if not ckpt.is_dir() or not (ckpt / ".metadata").is_file():
        return False
    carry = list(root.glob(f"carry_epoch_{epoch}.*.pt"))
    return len(carry) >= required

if target is not None:
    if is_complete(target):
        print(target)
    raise SystemExit(0)

epochs = []
for ckpt in root.glob("fsdp2_epoch_*"):
    if not ckpt.is_dir() or not (ckpt / ".metadata").is_file():
        continue
    match = re.fullmatch(r"fsdp2_epoch_(\d+)", ckpt.name)
    if not match:
        continue
    epoch = int(match.group(1))
    if is_complete(epoch):
        epochs.append(epoch)
if epochs:
    print(max(epochs))
PY
}

latest_sft_epoch() {
  local path="$1"
  python - "${path}" "${required_carry_count}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = int(sys.argv[2])
epochs = []
for ckpt in root.glob("fsdp2_epoch_*"):
    if not ckpt.is_dir() or not (ckpt / ".metadata").is_file():
        continue
    match = re.fullmatch(r"fsdp2_epoch_(\d+)", ckpt.name)
    if not match:
        continue
    epoch = int(match.group(1))
    carry = list(root.glob(f"carry_epoch_{epoch}.*.pt"))
    if len(carry) >= required:
        epochs.append(epoch)
if epochs:
    print(max(epochs))
PY
}

sft_complete() {
  local path="$1"
  local epoch
  epoch="$(latest_sft_epoch "${path}" || true)"
  if [[ "${epoch}" == "${epochs}" ]]; then
    wait_stable_path "${path}/fsdp2_epoch_${epochs}" || return 1
    return 0
  fi
  return 1
}

wait_for_sft_data() {
  while true; do
    if [[ -f "${data_path}/prepare_summary.json" ]]; then
      if wait_stable_path "${data_path}" && validate_sft_data "${data_path}"; then
        return 0
      fi
      log "SFT data exists but is not stable/valid yet: ${data_path}"
    else
      local files size
      files="$(find "${data_path}" -type f 2>/dev/null | wc -l || true)"
      size="$(du -sh "${data_path}" 2>/dev/null | awk '{print $1}' || true)"
      log "waiting_sft_data path=${data_path} files=${files} size=${size:-missing}"
    fi
    sleep "${poll_seconds}"
  done
}

wait_for_resume_epoch() {
  local epoch
  while true; do
    epoch="$(latest_stable_resume_epoch || true)"
    if [[ -n "${epoch}" ]]; then
      if wait_stable_path "${resume_from}/fsdp2_epoch_${epoch}"; then
        echo "${epoch}"
        return 0
      fi
      log "resume checkpoint epoch ${epoch} exists but is still changing"
    else
      log "waiting_resume_checkpoint path=${resume_from} target_epoch=${target_resume_epoch:-latest}"
    fi
    sleep "${poll_seconds}"
  done
}

submit_sft() {
  local resume_epoch="$1"
  local attempt="$2"
  local attempt_checkpoint_path="$3"
  local name="${job_name}"
  local marker
  if (( attempt > 1 )); then
    name="${job_name}-r${attempt}"
  fi
  marker="${marker_dir}/${name}.submitted"

  if [[ -f "${marker}" ]]; then
    local existing_state
    existing_state="$(job_state "${name}")"
    if [[ "${existing_state}" == "RUNNING" || "${existing_state}" == "SUCCEEDED" ]]; then
      log "already_submitted job=${name} state=${existing_state} marker=${marker}"
      echo "${name}"
      return 0
    fi
    log "stale_marker_ignored job=${name} state=${existing_state} marker=${marker}"
  fi

  cd "${repo_dir}"
  local extra_args
  extra_args="epochs=${epochs} checkpoint_interval=1 log_interval=5 compile_train_batch=false fsdp_wrap_moe_experts=true allow_compile_moe=false +resume_epoch=${resume_epoch}"

  log "dry_run job=${name} resume_epoch=${resume_epoch} checkpoint_path=${attempt_checkpoint_path}"
  dry_run=true \
  repo_dir="${repo_dir}" \
  job_name="${name}" \
  num_gpus="${num_gpus}" \
  arch_size="${arch_size}" \
  resume_from="${resume_from}" \
  data_path="${data_path}" \
  checkpoint_path="${attempt_checkpoint_path}" \
  global_batch_size="${global_batch_size}" \
  weights_only_resume_from_ema=true \
  moe_triton_autotune=1 \
  moe_triton_sm_margin=16 \
  WANDB_MODE=offline \
  extra_args="${extra_args}" \
    bash scripts/rjob_hrm_sft.sh | tee "${marker}.dry_run" >&2

  log "submit job=${name} resume_epoch=${resume_epoch} checkpoint_path=${attempt_checkpoint_path}"
  repo_dir="${repo_dir}" \
  job_name="${name}" \
  num_gpus="${num_gpus}" \
  arch_size="${arch_size}" \
  resume_from="${resume_from}" \
  data_path="${data_path}" \
  checkpoint_path="${attempt_checkpoint_path}" \
  global_batch_size="${global_batch_size}" \
  weights_only_resume_from_ema=true \
  moe_triton_autotune=1 \
  moe_triton_sm_margin=16 \
  WANDB_MODE=offline \
  extra_args="${extra_args}" \
    bash scripts/rjob_hrm_sft.sh | tee "${marker}" >&2
  echo "${name}"
}

monitor_sft_job() {
  local name="$1"
  local attempt_checkpoint_path="$2"
  while true; do
    if sft_complete "${attempt_checkpoint_path}"; then
      log "sft_complete job=${name} checkpoint_path=${attempt_checkpoint_path} epoch=${epochs}"
      echo "${attempt_checkpoint_path}" > "${marker_dir}/${job_name}.latest_success_path"
      return 0
    fi
    local state latest_epoch
    state="$(job_state "${name}")"
    latest_epoch="$(latest_sft_epoch "${attempt_checkpoint_path}" || true)"
    log "sft_status job=${name} state=${state} latest_epoch=${latest_epoch:-none} checkpoint_path=${attempt_checkpoint_path}"
    if [[ "${state}" == "FAILED" || "${state}" == "SUCCEEDED" || "${state}" == "MISSING" ]]; then
      return 1
    fi
    sleep "${poll_seconds}"
  done
}

log "watching data_path=${data_path}"
log "repo_dir=${repo_dir}"
log "resume_from=${resume_from}"
log "target_resume_epoch=${target_resume_epoch:-latest}"
wait_for_sft_data
resume_epoch="$(wait_for_resume_epoch)"
attempt=1
while (( attempt <= sft_max_attempts )); do
  attempt_checkpoint_path="${checkpoint_path}"
  if (( attempt > 1 )); then
    attempt_checkpoint_path="${checkpoint_path}-r${attempt}"
  fi
  if sft_complete "${attempt_checkpoint_path}"; then
    log "sft_already_complete checkpoint_path=${attempt_checkpoint_path}"
    exit 0
  fi
  submitted_job="$(submit_sft "${resume_epoch}" "${attempt}" "${attempt_checkpoint_path}")"
  if monitor_sft_job "${submitted_job}" "${attempt_checkpoint_path}"; then
    log "sft_finished job=${submitted_job} checkpoint_path=${attempt_checkpoint_path}"
    exit 0
  fi
  log "sft_attempt_failed job=${submitted_job} attempt=${attempt}/${sft_max_attempts}"
  attempt=$((attempt + 1))
done

log "sft_failed_after_retries attempts=${sft_max_attempts} base_checkpoint_path=${checkpoint_path}"
exit 6
