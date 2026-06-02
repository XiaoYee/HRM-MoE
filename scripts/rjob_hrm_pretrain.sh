#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${mode:-pretrain}"
config_name="${config_name:-cfg_pretrain}"
num_gpus="${num_gpus:-16}"
job_name="${job_name:-hrm-pretrain-${num_gpus}g}"
arch_size="${arch_size:-XL}"
data_path="${data_path:-/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097}"

export mode config_name num_gpus job_name arch_size data_path
exec bash "${script_dir}/rjob_hrm_common.sh"
