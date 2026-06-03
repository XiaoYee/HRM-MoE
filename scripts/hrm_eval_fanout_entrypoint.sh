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
export HRM_EVAL_DATA_DIR="${HRM_EVAL_DATA_DIR:-}"

echo "HRM fanout eval log file: ${log_file}"
echo "HRM fanout eval benchmark logs: ${bench_log_dir}"
echo "HRM fanout eval checkpoint: ${ckpt_path}"
echo "HRM fanout eval data dir: ${HRM_EVAL_DATA_DIR:-<online datasets>}"

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

eval_args=(
  "config=${eval_config}"
  "ckpt_path=${ckpt_path}"
  "fanout.output_dir=${bench_log_dir}"
)

if [[ -n "${ckpt_epoch}" ]]; then
  eval_args+=("ckpt_epoch=${ckpt_epoch}")
fi

if [[ -n "${ckpt_use_ema}" ]]; then
  eval_args+=("ckpt_use_ema=${ckpt_use_ema}")
fi

if [[ -n "${run_only}" ]]; then
  eval_args+=("run_only=${run_only}")
fi

if [[ -n "${batch_size}" ]]; then
  eval_args+=("generation_config.batch_size=${batch_size}")
fi

if [[ -n "${max_parallel}" ]]; then
  eval_args+=("fanout.num_workers=${max_parallel}")
fi

if [[ -n "${eval_extra_args}" ]]; then
  # Hydra overrides should not contain spaces. Use this for simple key=value args.
  read -r -a parsed_eval_extra_args <<< "${eval_extra_args}"
  eval_args+=("${parsed_eval_extra_args[@]}")
fi

cd "${eval_workdir}"
python -m evaluation.sharded_main "${eval_args[@]}"
