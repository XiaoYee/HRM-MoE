#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-sft}"
config_name="${config_name:-cfg_sft}"
num_gpus="${num_gpus:-8}"
job_name="${job_name:-hrm-sft-${num_gpus}g}"
arch_size="${arch_size:-XL}"

if [[ -z "${resume_from:-}" || -z "${data_path:-}" || -z "${checkpoint_path:-}" ]]; then
  echo "Usage: resume_from=/path/to/pretrain data_path=/path/to/sft_data checkpoint_path=/path/to/out bash scripts/rjob_hrm_sft.sh" >&2
  exit 2
fi

export mode config_name num_gpus job_name arch_size resume_from data_path checkpoint_path
exec bash "${script_dir}/rjob_hrm_common.sh"
