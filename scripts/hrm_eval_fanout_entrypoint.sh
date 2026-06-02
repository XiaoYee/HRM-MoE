#!/usr/bin/env bash
set -euo pipefail
set -x

repo_dir="${HRM_REPO_DIR:-${repo_dir:-/mnt/shared-storage-user/quxiaoye/HRM-Text}}"
ckpt_path="${HRM_EVAL_CKPT_PATH:-${ckpt_path:-}}"
ckpt_epoch="${HRM_EVAL_CKPT_EPOCH:-${ckpt_epoch:-}}"
ckpt_use_ema="${HRM_EVAL_CKPT_USE_EMA:-${ckpt_use_ema:-}}"
eval_config="${HRM_EVAL_CONFIG:-${eval_config:-${repo_dir}/evaluation/config/hrm_benchmarking.yaml}}"
run_only="${HRM_EVAL_RUN_ONLY:-${run_only:-}}"
batch_size="${HRM_EVAL_BATCH_SIZE:-${batch_size:-}}"
eval_extra_args="${HRM_EVAL_EXTRA_ARGS:-${eval_extra_args:-}}"
bootstrap="${HRM_BOOTSTRAP:-${bootstrap:-1}}"
data_io_dir="${DATA_IO_DIR:-/mnt/shared-storage-user/quxiaoye/data_io}"
eval_workdir="${HRM_EVAL_WORKDIR:-${eval_workdir:-${data_io_dir}/tokenizer}}"
max_parallel="${HRM_EVAL_MAX_PARALLEL:-${eval_max_parallel:-}}"

if [[ -z "${ckpt_path}" ]]; then
  echo "Usage: ckpt_path=/path/to/checkpoint_dir bash scripts/rjob_hrm_eval.sh" >&2
  exit 2
fi

if [[ "${eval_config}" != /* ]]; then
  eval_config="${repo_dir}/${eval_config}"
fi

cd "${repo_dir}"

log_dir="${HRM_LOG_DIR:-${repo_dir}/rjob_logs}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${MLP_TASK_NAME:-hrm-eval-fanout}_${RJOB_TASK_INDEX:-0}_${NODE_RANK:-0}_$(date +%Y%m%d_%H%M%S).log"
bench_log_dir="${log_dir}/${MLP_TASK_NAME:-hrm-eval-fanout}_${RJOB_TASK_INDEX:-0}_${NODE_RANK:-0}_bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${bench_log_dir}"
exec > >(tee -a "${log_file}") 2>&1

export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${repo_dir}/.hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${repo_dir}/wandb}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

echo "HRM fanout eval log file: ${log_file}"
echo "HRM fanout eval benchmark logs: ${bench_log_dir}"
echo "HRM fanout eval checkpoint: ${ckpt_path}"

if [[ "${bootstrap}" == "1" || "${bootstrap}" == "true" ]]; then
  python -m pip install --no-cache-dir \
    -i "${PIP_INDEX_URL:-http://mirrors.i.h.pjlab.org.cn/pypi/simple/}" \
    --trusted-host "${PIP_TRUSTED_HOST:-mirrors.i.h.pjlab.org.cn}" \
    -r "${repo_dir}/docker/requirements/runtime_overlay.txt"
fi

if [[ ! -d "${eval_workdir}" ]]; then
  echo "Eval workdir ${eval_workdir} does not exist; falling back to ${repo_dir}." >&2
  eval_workdir="${repo_dir}"
fi

mapfile -t benchmarks < <(python - "${eval_config}" "${run_only}" <<'PY'
import sys
import yaml

config_path, run_only = sys.argv[1], sys.argv[2].strip()
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

names = [item["name"] for item in cfg["benchmarks"]]
if run_only:
    if run_only.startswith("[") and run_only.endswith("]"):
        requested = [x.strip().strip("'\"") for x in run_only[1:-1].split(",") if x.strip()]
    else:
        requested = [run_only]
    requested_set = set(requested)
    unknown = [name for name in requested if name not in names]
    if unknown:
        raise ValueError(f"Unknown benchmark(s) in run_only: {unknown}")
    names = [name for name in names if name in requested_set]

for name in names:
    print(name)
PY
)

if (( ${#benchmarks[@]} == 0 )); then
  echo "No benchmarks selected." >&2
  exit 2
fi

gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if ! [[ "${gpu_count}" =~ ^[0-9]+$ ]] || (( gpu_count < 1 )); then
  echo "No CUDA devices available for evaluation." >&2
  exit 1
fi

if [[ -z "${max_parallel}" ]]; then
  max_parallel="${gpu_count}"
fi

if ! [[ "${max_parallel}" =~ ^[0-9]+$ ]] || (( max_parallel < 1 )); then
  echo "eval_max_parallel/HRM_EVAL_MAX_PARALLEL must be a positive integer." >&2
  exit 2
fi

if (( max_parallel > gpu_count )); then
  max_parallel="${gpu_count}"
fi

echo "Selected benchmarks: ${benchmarks[*]}"
echo "CUDA device count: ${gpu_count}; max parallel workers: ${max_parallel}"

run_benchmark() {
  local bench="$1"
  local gpu_id="$2"
  local bench_log="${bench_log_dir}/${bench}.log"
  local eval_args=(
    "config=${eval_config}"
    "ckpt_path=${ckpt_path}"
    "run_only=[${bench}]"
  )

  if [[ -n "${ckpt_epoch}" ]]; then
    eval_args+=("ckpt_epoch=${ckpt_epoch}")
  fi

  if [[ -n "${ckpt_use_ema}" ]]; then
    eval_args+=("ckpt_use_ema=${ckpt_use_ema}")
  fi

  if [[ -n "${batch_size}" ]]; then
    eval_args+=("generation_config.batch_size=${batch_size}")
  fi

  if [[ -n "${eval_extra_args}" ]]; then
    read -r -a parsed_eval_extra_args <<< "${eval_extra_args}"
    eval_args+=("${parsed_eval_extra_args[@]}")
  fi

  (
    cd "${eval_workdir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m evaluation.main "${eval_args[@]}"
  ) > >(tee -a "${bench_log}" | sed -u "s/^/[${bench}] /") 2> >(tee -a "${bench_log}" >&2)
}

active_jobs=0
pids=()
pid_names=()
next_gpu=0
failed=0

for bench in "${benchmarks[@]}"; do
  gpu_id="${next_gpu}"
  next_gpu=$(((next_gpu + 1) % gpu_count))

  run_benchmark "${bench}" "${gpu_id}" &
  pids+=("$!")
  pid_names+=("${bench}")
  active_jobs=$((active_jobs + 1))

  if (( active_jobs >= max_parallel )); then
    if ! wait "${pids[0]}"; then
      echo "Benchmark ${pid_names[0]} failed." >&2
      failed=1
    fi
    pids=("${pids[@]:1}")
    pid_names=("${pid_names[@]:1}")
    active_jobs=$((active_jobs - 1))
  fi
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Benchmark ${pid_names[$i]} failed." >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  exit 1
fi

echo "All selected benchmarks finished."
