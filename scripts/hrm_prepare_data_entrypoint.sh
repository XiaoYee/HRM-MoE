#!/usr/bin/env bash
set -euo pipefail
set -x

data_io_dir="${DATA_IO_DIR:-/mnt/shared-storage-user/quxiaoye/data_io}"
stage="${DATA_PREP_STAGE:-${stage:-check}}"
cleaned_repo="${CLEANED_REPO:-sapientinc/HRM-Text-data-io-cleaned-20260515}"
hrm_repo_dir="${HRM_REPO_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text}"
tokenized_path="${TOKENIZED_PATH:-${hrm_repo_dir}/data_tokenized_bpe_65k}"
sampled_output="${SAMPLED_OUTPUT:-${sampled_output:-/dev/shm/sampled}}"
epochs="${DATA_EPOCHS:-${epochs:-4}}"
context_size="${CONTEXT_SIZE:-4097}"
download_workers="${DOWNLOAD_WORKERS:-16}"
tokenizer_workers="${TOKENIZER_WORKERS:-16}"
tokenizer_batch_size="${TOKENIZER_BATCH_SIZE:-8192}"
analytics_path="${ANALYTICS_PATH:-${hrm_repo_dir}/data_prep_logs/show_analytics_e${epochs}_ctx${context_size}.md}"
auto_install_rust="${AUTO_INSTALL_RUST:-1}"
tokenizer_impl="${TOKENIZER_IMPL:-auto}"
hrm_rust_home="${HRM_RUST_HOME:-/mnt/shared-storage-user/quxiaoye/.hrm-rust}"
export RUSTUP_HOME="${RUSTUP_HOME:-${hrm_rust_home}/rustup}"
export CARGO_HOME="${CARGO_HOME:-${hrm_rust_home}/cargo}"
export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://mirrors.tuna.tsinghua.edu.cn/rustup}"
export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export PATH="${CARGO_HOME}/bin:${PATH}"

log_dir="${DATA_PREP_LOG_DIR:-/mnt/shared-storage-user/quxiaoye/HRM-Text/data_prep_logs}"
mkdir -p "${log_dir}"
log_file="${log_dir}/prepare_${stage}_${RJOB_TASK_INDEX:-0}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${log_file}") 2>&1
echo "Data prep log file: ${log_file}"

cd "${data_io_dir}"

python -m pip install --no-cache-dir \
  -i "${PIP_INDEX_URL:-http://mirrors.i.h.pjlab.org.cn/pypi/simple/}" \
  --trusted-host "${PIP_TRUSTED_HOST:-mirrors.i.h.pjlab.org.cn}" \
  -r requirements.txt

ensure_cargo() {
  if command -v cargo >/dev/null 2>&1; then
    cargo --version
    rustc --version || true
    return
  fi

  if [[ "${auto_install_rust}" != "1" ]]; then
    echo "missing_cargo: Rust/Cargo is required for tokenization." >&2
    return 3
  fi

  if command -v curl >/dev/null 2>&1; then
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}"
    tmp_dir="$(mktemp -d)"
    curl -L --retry 5 --retry-delay 3 \
      "${RUSTUP_UPDATE_ROOT}/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init" \
      -o "${tmp_dir}/rustup-init"
    chmod +x "${tmp_dir}/rustup-init"
    "${tmp_dir}/rustup-init" -y --no-modify-path --profile minimal --default-toolchain stable
    rm -rf "${tmp_dir}"
    cargo --version
    rustc --version || true
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "missing_cargo: apt-get is unavailable; use an image with Rust/Cargo." >&2
    return 3
  fi

  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    cargo \
    pkg-config \
    rustc

  cargo --version
  rustc --version || true
}

run_rust_tokenizer() {
  ensure_cargo
  if [[ -d "${tokenized_path}" ]]; then
    find "${tokenized_path}" -name ".*.tmp" -delete
  fi
  (
    cd tokenizer
    cargo run --release --bin tokenizer -- \
      ../data ../data_clustered \
      --tokenizer-path ../trained_tokenizers/bpe/tokenizer.json \
      -o "${tokenized_path}"
  )
}

check_env() {
  python - <<'PY'
import importlib
for name in ["datasets", "huggingface_hub", "numpy", "omegaconf", "pydantic", "pyarrow", "tokenizers", "yaml", "tqdm"]:
    importlib.import_module(name)
    print("import_ok", name)
PY
  if command -v cargo >/dev/null 2>&1; then
    cargo --version
    rustc --version || true
  else
    echo "cargo_missing_python_tokenizer_fallback_enabled"
  fi
  test -f trained_tokenizers/bpe/tokenizer.json
}

download_cleaned() {
  hf download "${cleaned_repo}" \
    --repo-type dataset \
    --local-dir "${data_io_dir}" \
    --include 'data/**' 'data_clustered/**' \
    --max-workers "${download_workers}"
}

tokenize() {
  if [[ ! -d data || ! -d data_clustered ]]; then
    echo "Missing cleaned data directories. Run stage=download_cleaned first." >&2
    exit 4
  fi
  if [[ "${tokenizer_impl}" != "python" ]] && command -v cargo >/dev/null 2>&1; then
    run_rust_tokenizer
  elif [[ "${tokenizer_impl}" == "rust" ]]; then
    run_rust_tokenizer
  else
    python "${hrm_repo_dir}/scripts/tokenize_data_io_python.py" \
      "${data_io_dir}/data" "${data_io_dir}/data_clustered" \
      --tokenizer-path "${data_io_dir}/trained_tokenizers/bpe/tokenizer.json" \
      --workers "${tokenizer_workers}" \
      --batch-size "${tokenizer_batch_size}" \
      -o "${tokenized_path}"
  fi
}

sample_tokenized() {
  if [[ ! -f "${tokenized_path}/tokenizer_info.json" ]]; then
    echo "Missing tokenized data at ${tokenized_path}. Run stage=tokenize first." >&2
    exit 5
  fi
  python sample_tokenized.py \
    "tokenized_path=${tokenized_path}" \
    "output_path=${sampled_output}" \
    "epochs=${epochs}" \
    "context_size=${context_size}" \
    "prefix_config_path=${data_io_dir}/prefix_config.yaml" \
    > "${analytics_path}"

  python - <<PY
from pathlib import Path
required = [
    "metadata.json",
    "tokens.npy",
    "epoch_0/inst_start.npy",
    "epoch_0/inst_len.npy",
    "epoch_0/resp_start.npy",
    "epoch_0/resp_len.npy",
]
root = Path(${sampled_output@Q})
missing = [p for p in required if not (root / p).exists()]
if missing:
    raise SystemExit(f"missing sampled outputs: {missing}")
print("sampled_output_ok", root)
PY
}

case "${stage}" in
  check)
    check_env
    ;;
  download_cleaned)
    download_cleaned
    ;;
  tokenize)
    check_env
    tokenize
    ;;
  sample)
    sample_tokenized
    ;;
  all)
    download_cleaned
    check_env
    tokenize
    sample_tokenized
    ;;
  *)
    echo "Unknown DATA_PREP_STAGE: ${stage}" >&2
    exit 2
    ;;
esac
