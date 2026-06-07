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
global_batch_size="${SFT_GLOBAL_BATCH_SIZE:-32768}"
required_carry_count="${REQUIRED_CARRY_COUNT:-32}"
poll_seconds="${POLL_SECONDS:-300}"
stability_seconds="${STABILITY_SECONDS:-30}"
marker_dir="${MARKER_DIR:-${repo_dir}/rjob_logs}"
marker="${marker_dir}/${job_name}.submitted"

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

validate_sft_data() {
  local path="$1"
  python - "${path}" "${epochs}" <<'PY'
import json
import sys
from pathlib import Path
from numpy.lib import format as npy_format

path = Path(sys.argv[1])
epochs = int(sys.argv[2])

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
    f"skipped_long={summary.get('skipped_long')}"
)
PY
}

latest_stable_resume_epoch() {
  python - "${resume_from}" "${required_carry_count}" <<'PY'
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
      log "waiting_resume_checkpoint path=${resume_from}"
    fi
    sleep "${poll_seconds}"
  done
}

submit_sft() {
  local resume_epoch="$1"
  if [[ -f "${marker}" ]]; then
    log "already_submitted marker=${marker}"
    cat "${marker}"
    return 0
  fi

  cd "${repo_dir}"
  local extra_args
  extra_args="epochs=${epochs} checkpoint_interval=1 log_interval=5 compile_train_batch=false fsdp_wrap_moe_experts=true allow_compile_moe=false resume_epoch=${resume_epoch}"

  log "dry_run job=${job_name} resume_epoch=${resume_epoch}"
  dry_run=true \
  repo_dir="${repo_dir}" \
  job_name="${job_name}" \
  num_gpus="${num_gpus}" \
  arch_size="${arch_size}" \
  resume_from="${resume_from}" \
  data_path="${data_path}" \
  checkpoint_path="${checkpoint_path}" \
  global_batch_size="${global_batch_size}" \
  weights_only_resume_from_ema=true \
  moe_triton_autotune=1 \
  moe_triton_sm_margin=16 \
  WANDB_MODE=offline \
  extra_args="${extra_args}" \
    bash scripts/rjob_hrm_sft.sh | tee "${marker}.dry_run"

  log "submit job=${job_name} resume_epoch=${resume_epoch}"
  repo_dir="${repo_dir}" \
  job_name="${job_name}" \
  num_gpus="${num_gpus}" \
  arch_size="${arch_size}" \
  resume_from="${resume_from}" \
  data_path="${data_path}" \
  checkpoint_path="${checkpoint_path}" \
  global_batch_size="${global_batch_size}" \
  weights_only_resume_from_ema=true \
  moe_triton_autotune=1 \
  moe_triton_sm_margin=16 \
  WANDB_MODE=offline \
  extra_args="${extra_args}" \
    bash scripts/rjob_hrm_sft.sh | tee "${marker}"
}

log "watching data_path=${data_path}"
log "repo_dir=${repo_dir}"
log "resume_from=${resume_from}"
wait_for_sft_data
resume_epoch="$(wait_for_resume_epoch)"
submit_sft "${resume_epoch}"
