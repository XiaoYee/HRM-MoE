# HRM-Text Agent Guide

## Project Positioning

This repository is the working checkout for HRM-Text training on the rjob
cluster. It is not a generic language-model training template.

Use these files as the primary source of truth:

- `README.md` for the HRM-Text workflow and supported training modes
- `docker/Dockerfile` for the tested CUDA/PyTorch/FlashAttention environment
- `config/cfg_pretrain.yaml` for default pretraining settings
- `config/cfg_sft.yaml` for full-parameter SFT settings
- `pretrain.py` for the actual Hydra and FSDP2 training entrypoint

## Version Control

All agent code changes in this repository must be managed with git:

- Work on a named branch instead of leaving substantial changes only in the
  worktree.
- Run `git status --short --branch` before and after edits.
- Stage only intentional code, script, and documentation changes; never stage
  data, checkpoints, W&B outputs, or rjob logs.
- Commit completed changes with a concise message after validation. If a change
  is intentionally left uncommitted, record why in the user update.
- After completing cluster, training, data-prep, or evaluation work, update this
  AGENTS.md with durable lessons learned, reusable commands, and new pitfalls
  before committing. Treat this as part of the done criteria, not optional
  cleanup.
- Record all future HRM experiment results and operational outcomes in
  `docs/hrm_eval_results.md`. Training launches, eval jobs, benchmark
  summaries, failed attempts, root-cause notes, cleanup actions, resource
  settings, follow-up AIME/MMLU-Pro results, and notable reruns should be
  appended there rather than scattered across separate ad hoc notes. Write new
  experiment records in Chinese; keep benchmark names, metric keys, paths, and
  command fragments in their original spelling when that is clearer.
- Do not edit entrypoint scripts while an rjob is still reading them from shared
  storage. Prefer committing fixes, then launching a fresh rjob from that commit.

## Environment Model

Do not treat the local `.venv` as the training environment. HRM training depends
on Hopper-class GPUs, CUDA 12.8-era PyTorch, and `flash_attn_3`, so meaningful
training and import validation should run through rjob unless the task is
explicitly CPU-only.

The source-of-truth environment is the HRM image from the README:

```bash
sapientai/hrm-text:latest
```

The current rjob default uses the internal XTuner image plus a lightweight HRM
Python overlay because this cluster cannot currently pull the public Docker Hub
image:

```bash
registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194
```

If an internal HRM image is available, override `image=... bootstrap=0` when
launching rjob.

Build an internal image on a machine with Docker daemon access:

```bash
IMAGE_NAME=registry.h.pjlab.org.cn/ailab-moe-moe_gpu/hrm-text \
PUSH=1 bash scripts/build_hrm_image.sh
```

## Rjob Workflow

Use the HRM rjob wrappers under `scripts/`:

- `scripts/rjob_hrm_env_check.sh` checks the container imports and can request
  either CPU-only (`num_gpus=0`) or one 8-GPU node (`num_gpus=8`)
- `scripts/rjob_hrm_pretrain.sh` launches `pretrain.py` with pretraining config
- `scripts/rjob_hrm_sft.sh` launches `pretrain.py --config-name cfg_sft`
- `scripts/rjob_hrm_eval.sh` launches evaluation; with the default entrypoint it
  runs `python -m evaluation.main` on one GPU, and with
  `entrypoint=scripts/hrm_eval_fanout_entrypoint.sh num_gpus=8` it runs sharded
  8-card evaluation
- `scripts/rjob_hrm_eval_after_epoch.sh` watches a checkpoint directory until a
  target epoch is complete and stable, then submits an eval rjob
- `scripts/rjob_hrm_prepare_data.sh` runs HRM pretraining data preparation
- `scripts/rjob_hrm_common.sh` owns rjob resources, image, mounts, and RDMA flags
- `scripts/hrm_entrypoint.sh` owns container-side env vars and `torchrun`
- `scripts/hrm_eval_entrypoint.sh` owns container-side evaluation setup
- `scripts/hrm_eval_fanout_entrypoint.sh` shards each selected benchmark's
  prompts across available GPUs and aggregates metrics after generation
- `scripts/hrm_prepare_data_entrypoint.sh` owns `data_io` download, tokenization,
  and stratified sampling
- `scripts/build_hrm_image.sh` builds and optionally pushes an internal HRM image
- `docker/requirements/runtime_overlay.txt` is the bootstrap overlay for the
  internal fallback image; it intentionally excludes `torch`, `flash_attn_3`,
  and `vllm`
- `scripts/tokenize_data_io_python.py` is a no-Cargo fallback that writes the
  same tokenized layout as `data_io/tokenizer`
- `scripts/compare_tokenized_outputs.py` compares Rust and Python tokenized
  outputs for small-sample parity before trusting the Python fallback
- Set `tokenizer_workers=<N>` on `stage=tokenize` when using the Python
  fallback; the container may report a low `os.cpu_count()` despite a larger
  rjob CPU request.
- Prefer the official Rust `data_io/tokenizer` for full data prep. A shared
  Rust toolchain is installed at `/mnt/shared-storage-user/quxiaoye/.hrm-rust`,
  and the wrapper passes `CARGO_HOME`, `RUSTUP_HOME`, and mirror URLs into
  rjob. `tokenizer_impl=auto` uses Rust only when `cargo` is already present;
  it falls back to Python instead of auto-installing Rust. Use
  `tokenizer_impl=rust bootstrap=0 cpu=5` for tokenization so the
  Rust tokenizer sees only four worker threads; the official tokenizer holds a
  file's tokenized output in memory before writing, so exposing dozens of CPUs
  can over-parallelize large parquet files.

These wrappers follow the same pattern as the reference XTuner project:
thin experiment wrappers set `job_name`, `num_gpus`, and training overrides,
then delegate to a common rjob launcher and one container entrypoint.

Common overrides:

```bash
dry_run=true bash scripts/rjob_hrm_env_check.sh
job_name=hrm-env-check-cpu bash scripts/rjob_hrm_env_check.sh
job_name=hrm-env-check-gpu num_gpus=8 bash scripts/rjob_hrm_env_check.sh
stage=check bash scripts/rjob_hrm_prepare_data.sh
stage=download_cleaned bash scripts/rjob_hrm_prepare_data.sh
stage=tokenize bash scripts/rjob_hrm_prepare_data.sh
stage=tokenize tokenizer_impl=rust bootstrap=0 cpu=5 bash scripts/rjob_hrm_prepare_data.sh
stage=sample epochs=4 bash scripts/rjob_hrm_prepare_data.sh
python scripts/compare_tokenized_outputs.py /path/to/rust_tokenized /path/to/python_tokenized
num_gpus=8 arch_size=L global_batch_size=172032 extra_args='lr=2.5e-4' bash scripts/rjob_hrm_pretrain.sh
num_gpus=16 arch_size=XL bash scripts/rjob_hrm_pretrain.sh
resume_from=/path/to/pretrain data_path=/path/to/sft_data checkpoint_path=/path/to/out bash scripts/rjob_hrm_sft.sh
ckpt_path=/path/to/checkpoints/run_dir run_only='[GSM8k,MATH]' bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 batch_size=16 bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 batch_size=16 entrypoint=scripts/hrm_eval_fanout_entrypoint.sh bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 batch_size=16 bash scripts/rjob_hrm_eval_after_epoch.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 batch_size=16 entrypoint=scripts/hrm_eval_fanout_entrypoint.sh eval_config=/mnt/shared-storage-user/quxiaoye/HRM-Text/evaluation/config/hrm_mmlu_pro_benchmarking.yaml eval_data_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet hf_datasets_offline=1 bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 entrypoint=scripts/hrm_eval_fanout_entrypoint.sh eval_config=/mnt/shared-storage-user/quxiaoye/HRM-Text/evaluation/config/hrm_maj_vote_benchmarking.yaml eval_data_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet hf_datasets_offline=1 bash scripts/rjob_hrm_eval.sh
```

Default rjob settings are intentionally close to the reference project:

- `gpu_group=moe_gpu`
- `namespace=ailab-moe`
- `gpu=8` per replica for training
- one replica per 8 GPUs for training
- env-check wrapper lowers CPU/memory and disables RDMA because it only
  validates imports
- eval wrapper defaults to one GPU, no RDMA, no gang start, and a separate
  entrypoint so it does not touch a running training entrypoint
- 8-card eval uses one eval replica with eight GPUs; the fanout entrypoint keeps
  all selected GPUs busy by splitting each benchmark's prompt list across GPUs
  instead of assigning one benchmark per GPU
- host network, gang start, RDMA resources
- mounts for `quxiaoye`, `moegroup`, `moegroup2`, and `intern7shared`
- data-prep wrapper defaults `bootstrap=0` because it installs `data_io`
  requirements itself and does not need the training runtime overlay

Preserve these hardcoded cluster paths unless the task is specifically about
changing cluster environment setup.

## Data Expectations

Pretraining expects sampled tokenized data at `data.path`, defaulting to
`/dev/shm/sampled` in `config/data/hlm.yaml`. The directory must contain:

- `metadata.json`
- `tokens.npy`
- `epoch_<n>/inst_start.npy`
- `epoch_<n>/inst_len.npy`
- `epoch_<n>/resp_start.npy`
- `epoch_<n>/resp_len.npy`

The companion `data_io` checkout lives at:

```bash
/mnt/shared-storage-user/quxiaoye/data_io
```

Keep HRM's tokenized pretraining output under this repository:

```bash
/mnt/shared-storage-user/quxiaoye/HRM-Text/data_tokenized_bpe_65k
```

Keep HRM's default sampled pretraining output under this repository too:

```bash
/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097
```

The default data-prep path uses the official cleaned dataset from Hugging Face,
then tokenizes it with `data_io/tokenizer` when Rust/Cargo is available or the
Python fallback otherwise, then samples it to the persistent HRM-Text path:

```bash
stage=all bash scripts/rjob_hrm_prepare_data.sh
```

The data-prep wrapper intentionally defaults `HF_ENDPOINT` to
`https://huggingface.co`; the login environment's `hf-mirror.com` setting can
list repos but fails on actual file downloads with the installed Hub client.
Override with `hf_endpoint=...` only after confirming `hf download` works.
Data-prep rjobs also pass the cluster HTTP(S) proxy into the container because
compute nodes otherwise cannot reach Hugging Face file downloads. Keep PJLab
internal domains in `NO_PROXY` so pip can still use the internal mirror. CPU
rjobs cannot use host networking on this platform; if a download must run under
rjob with host networking, explicitly request an 8-GPU task with
`num_gpus=8 host_network=true`.
If CPU rjobs cannot use host networking, use
`scripts/local_hrm_download_cleaned.sh` from the login machine to download the
cleaned dataset onto shared storage, then resume the rjob flow at
`stage=tokenize`. `scripts/local_hrm_after_download.sh` can watch the download
tmux session and submit tokenization automatically after a successful download.

Do not default to `/dev/shm/sampled` for reusable experiments. `/dev/shm` is
node-local and disappears when the rjob exits, so a separate training rjob may
land on another node and miss the sampled data. Prefer the persistent sampled
path above and pass it as `data_path` for training. Only use `/dev/shm/sampled`
for short same-container debugging when the user explicitly asks.

For multi-node training, generate the persistent sampled dataset once with the
same `epochs` and `context_size` as training, then mount/read that shared path
from all nodes. Re-sampling per node is deterministic but wastes startup time
and can hide node-local path mistakes.

SFT data is prepared with:

```bash
python scripts/prepare_sft_data.py \
  --train input.jsonl \
  --tokenizer /path/to/tokenizer.json \
  --output /path/to/sft_data \
  --epochs 5
```

`--epochs` must match the training config's `epochs`.

## Training Guardrails

- Do not submit a long pretraining or SFT job unless the user explicitly asks.
- Use `dry_run=true` on a wrapper when changing launcher options.
- Run CPU-only `scripts/rjob_hrm_env_check.sh` before using a new image, then
  run `num_gpus=8 bash scripts/rjob_hrm_env_check.sh` before the first real
  training job when GPU quota is available.
- rjob stdout/stderr is also written to `rjob_logs/` by `hrm_entrypoint.sh`;
  inspect that directory if `rjob logs` is empty or flaky.
- Keep checkpoints on shared storage when training with more than one node.
- Keep cleaned `data_io` outputs and HRM tokenized outputs on shared storage;
  keep reusable sampled data on shared storage under HRM-Text, not in
  `/dev/shm`.
- Pass Hydra overrides directly through `extra_args` or the wrapper-specific
  env vars instead of editing default configs for one-off experiments.
- For SFT, require `resume_from`, `data_path`, and `checkpoint_path`.
- For evaluation, require `ckpt_path`; pass benchmark subsets with
  `run_only='[GSM8k,MATH]'` and lower memory use with `batch_size=16`.
  The HRM evaluation entrypoint defaults `eval_workdir` to
  `/mnt/shared-storage-user/quxiaoye/data_io/tokenizer` because current
  checkpoints store the BPE tokenizer path as `../trained_tokenizers/...`.
- For all-benchmark 8-card evaluation, prefer
  `entrypoint=scripts/hrm_eval_fanout_entrypoint.sh num_gpus=8`; it data-shards
  every benchmark across GPUs and avoids the low-utilization pattern of one GPU
  per benchmark.
- When waiting for a training epoch before evaluation, use
  `scripts/rjob_hrm_eval_after_epoch.sh` or equivalent logic. Checkpoint
  readiness means `fsdp2_epoch_<N>/.metadata` exists, all expected
  `carry_epoch_<N>.<rank>.pt` files exist, and file sizes/mtimes have stayed
  stable across repeated checks.
- W&B is used by `pretrain.py`; make sure credentials are available in the job
  environment before long runs.

## Cluster Lessons Learned

- Tokenization and sampling are different stages. The expensive one-time
  tokenizer output is `data_tokenized_bpe_65k`; `sample_tokenized.py` only
  builds `tokens.npy`, `metadata.json`, and `epoch_*` arrays for training.
- The Python tokenizer fallback must be treated as an emergency path until it
  passes a small-sample parity check against the Rust tokenizer with
  `scripts/compare_tokenized_outputs.py`. The persistent data used here was
  produced by the official Rust tokenizer, not the fallback.
- The official Rust tokenizer should run with `tokenizer_impl=rust bootstrap=0
  cpu=5`; exposing many CPUs can over-parallelize parquet files and blow up
  memory because the tokenizer holds each tokenized file before writing.
- Rust/Cargo is installed in `/mnt/shared-storage-user/quxiaoye/.hrm-rust` with
  a Cargo mirror. Keep `CARGO_HOME`, `RUSTUP_HOME`, and mirror env vars in rjob.
- Tokenized outputs created by rjob may be root-owned. If local `mv` fails with
  permission denied, use a short CPU rjob to move/verify the directory instead
  of copying hundreds of GB.
- Do not edit an entrypoint script while an rjob is still executing it from
  shared storage. Bash can keep reading the mounted script as it runs; a data
  prep job can finish `sampled_output_ok` and still fail at script EOF if the
  file changed underneath it. Treat the data as usable only after validating the
  sampled files, and use a fresh rjob for the edited script.
- The login node may fail to `np.load(..., mmap_mode="r")` very large GPFS
  `.npy` files with `OSError: [Errno 19] No such device`, even though the data
  prep rjob successfully used `open_memmap` on a compute node. For local
  validation, parse `.npy` headers or check files structurally; verify compute
  mmap behavior through rjob.
- In bash helpers that use `local -n`, do not name the nameref variable the
  same as the caller's variable. `local -n target_array=target_array` creates a
  circular name reference and silently drops Hydra overrides such as
  `data.path`, causing training to fall back to `/dev/shm/sampled`.
- The internal XTuner fallback image has a `flash_attn_interface`
  `_flash_attn_backward` wrapper that uses the older keyword `causal` instead
  of `is_causal` and requires `softmax_scale`; the observed signature also
  accepts `window_size`, `softcap`, `deterministic`, and `sm_margin`. Keep
  `models/flash_attention_prefixlm_v2.py` compatible with both APIs and pass an
  explicit `q.shape[-1] ** -0.5` scale through forward and backward so the two
  API paths cannot diverge silently.
- CPU rjobs cannot rely on host networking here. Downloads that need host
  network should either run locally on the login node or request an 8-GPU rjob
  with `host_network=true`; sampling/tokenization should not need host network.
- `hf-mirror.com` can list repos but has failed for actual cleaned-data
  downloads with the installed Hub client. Use `https://huggingface.co` unless
  a download test proves otherwise.
- `rjob logs` requires a target type, e.g. `rjob logs job <name>` or
  `rjob logs replica <replica>`. In STARTING it can return empty-log errors;
  also inspect `rjob_logs/` and `data_prep_logs/`.
- For first 8-card bring-up, use README's L-size settings:
  `arch_size=L global_batch_size=172032 extra_args='lr=2.5e-4'`. XL is the
  two-node reference default.
- Use `WANDB_MODE=offline` for bring-up if credentials are uncertain. Switch to
  online only when the run is meant to be recorded.
- The persistent sample build for
  `/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097`
  produced 18 files and about 673G: `tokens.npy`, `metadata.json`, and four
  epoch directories. Token writing took about 47 minutes, epoch index generation
  about 43 seconds, and report generation about 6 seconds on the observed CPU
  rjob.
- `num_gpus=8` only allocates eight GPUs for an eval rjob; it does not make a
  single `evaluation.main` process use all eight. Avoid the naive strategy of
  one benchmark per GPU for full eval because benchmark sizes vary and the job
  leaves GPUs idle after short benchmarks finish. Use prompt-level sharding
  within each benchmark and aggregate generations before computing metrics.
- Epoch-gated eval watchers should write idempotency markers under `rjob_logs/`
  rather than under checkpoint directories. Checkpoint directories are often
  root-owned because training rjobs create them.
- For long-lived local watchers, run them in tmux and tee logs into
  `rjob_logs/`. If you change watcher submission logic, restart the watcher;
  if you only change the future eval entrypoint path it will be picked up by the
  rjob when the watcher eventually submits.
- Always `dry_run=true` a new rjob eval launch shape before leaving a watcher to
  submit it automatically.
- Local watchdogs that run under `set -u` should use `bash -c`, not `bash -lc`.
  On the login node, the login-shell profile has referenced `ZSH_VERSION`
  without a default and can make a watcher exit before it writes its own log.
- Keep auto-submitted eval job names short, e.g. `hrm-eval-e1-0602`. The rjob
  client combines the job name with `generated-task-0` for a Kubernetes label,
  so descriptive checkpoint-derived names can exceed the 63-character label
  limit even before the job is submitted.
- Full benchmark eval rjobs should not depend on live Hugging Face Hub dataset
  resolution. Compute-node `datasets.load_dataset(...)` has failed here with
  proxy errors even when direct file downloads from the login node worked.
  Download the required benchmark parquet files to shared storage with
  `scripts/download_eval_data.py`, then launch eval with
  `eval_data_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet`
  so `HRM_EVAL_DATA_DIR` is passed into the rjob.
- When using local parquet eval data, setting `HF_DATASETS_OFFLINE=1` is useful
  because it proves benchmark construction is not secretly falling back to Hub
  access. Missing local mappings should fail fast before model inference.
- Python 3.12 evaluates stricter `typing` generics at import time. Use
  `Generator[yield_type, send_type, return_type]`; a two-argument
  `Generator[...]` annotation can kill all eval shards before inference starts.
- Current HRM checkpoints may store tokenizer paths like
  `../trained_tokenizers/bpe/tokenizer.json`. Eval runs from
  `/mnt/shared-storage-user/quxiaoye/data_io/tokenizer` so this resolves to the
  shared `data_io/trained_tokenizers` directory. Resolve the path before calling
  `AutoTokenizer.from_pretrained`, and pass the tokenizer directory rather than
  the raw `tokenizer.json` file path.
- `torch.compile` can fail on eval prefill/decode with FA3 custom ops because
  Dynamo fake tensors do not support `flash_attn_3.fwd.default` in this image.
  Keep inference compile optional with `HRM_EVAL_TORCH_COMPILE=1`; default to
  eager execution for bring-up and correctness runs.
- Local `py_compile` can fail because rjobs create root-owned `__pycache__`
  files under the repo. Use a writable temporary prefix, e.g.
  `PYTHONPYCACHEPREFIX=/tmp/hrm_pycache_check python -m py_compile ...`, rather
  than changing ownership of generated cache files during debugging.
- MMLU-Pro and AIME are not part of `evaluation/config/hrm_benchmarking.yaml`.
  Use `evaluation/config/hrm_mmlu_pro_benchmarking.yaml` for MMLU-Pro and
  `evaluation/config/hrm_maj_vote_benchmarking.yaml` for AIME majority voting.
  MMLU-Pro local data is `TIGER-Lab/MMLU-Pro/data/test-00000-of-00001.parquet`;
  AIME25 local data is `math-ai/aime25/test.jsonl`, so the local eval loader
  must use the `json` dataset builder for AIME instead of assuming every cached
  benchmark file is parquet.
- For 64x8 MoE speed work, precision is a hard gate. Run local equivalence and
  the CUDA/bfloat16 rjob gate before treating a kernel, dispatch, or FSDP/EP
  change as usable. Do not trade away router fp32 softmax, top-k=8, routing
  weight normalization, or index-add aggregation semantics for speed.
- The current fastest safe 64x8 MoE path is grouped Triton expert compute with
  `HRM_MOE_TRITON_AUTOTUNE=1` and `HRM_MOE_TRITON_SM_MARGIN=8` or `16`.
  `HRM_MOE_TRITON_BLOCK_M=64` and `SM_MARGIN=32` were slower in same-shape
  smoke runs. Keep the wrapper env parsing tolerant of empty string values
  because rjob launchers pass unset env vars explicitly.
- CUTLASS `grouped_gemm` passed the CUDA/bfloat16 equivalence gate in the
  XTuner fallback image, but the training smoke was much slower than Triton
  because backward/wgrad dominated. Do not switch the default MoE backend to
  CUTLASS without a new precision gate and a faster same-shape smoke.
- Do not use `torch.compile` for MoE `train_batch` by default. The force-compile
  smoke caused graph breaks/recompilation and very large optimizer-step times.
  Keep MoE compile disabled unless a fresh gate proves both correctness and
  speed.
- Expert Parallel all-to-all equivalence is not sufficient. The custom
  autograd all-to-all path passed distributed correctness, but multi-step
  training still stalled on the second backward. EP changes need a multi-step
  training gate, not just a forward/backward equivalence script.
- Keep the MoE dispatch `torch.sort(..., stable=True)` unless a new same-shape
  smoke proves otherwise. A default-sort experiment passed numerical
  equivalence but slowed the Triton path, so it was reverted.
- Use `HRM_MOE_PROFILE=1` only for diagnosis. It inserts CUDA events and
  synchronizes phase boundaries, so the absolute step time is slower than a
  normal smoke; use it for breakdown, not headline speed. The current Triton
  64x8 profile shows forward dispatch/combine and forward grouped GEMMs are
  more promising than backward grouped GEMM. Backward `grad_input_gemm` and
  `grad_weight_gemm` were small in the measured steady step.
- Do not replace `flat_token_idx = arange(...).repeat_interleave(top_k)` with
  `sort_idx // top_k` just because it is algebraically equivalent. That
  token-index experiment passed local and CUDA/bfloat16 equivalence but slowed
  the same-shape Triton smoke, so it was reverted.
- Do not move the routing weight multiply before the expert down projection in
  HRM MoE just because it is algebraically equivalent. The pre-weight attempt
  passed local CPU/fp32 equivalence but failed the CUDA/bfloat16 gate on router
  gradients (`hrm-moe-preweight-eq-06050027`), with max absolute difference
  0.1640625 against `atol=5e-2`. MoE precision is sensitive to bf16 rounding
  placement; keep routing weights applied after the down projection unless a
  new fused forward/backward kernel passes the full CUDA/bfloat16 gradient
  gate.
- GitHub directions for future MoE speed work: MegaBlocks supports continuing
  grouped-GEMM/block-sparse work; Tutel and DeepEP are more relevant for
  expert-parallel communication and parameter/communication isolation; use
  TorchTitan mainly as a reference for FSDP2 organization and observability.
  Any imported idea still needs HRM local equivalence, CUDA/bfloat16 gate, and
  same-shape training smoke before adoption.
- Additional GitHub/kernel directions for the 1.2x dense MoE target: PyTorch's
  persistent cache-aware grouped GEMM work supports further Triton scheduler,
  grouped launch, and TMA-style tuning; DeepGEMM is a possible future backend
  only if the rjob image/toolchain matches and BF16 training gradients pass the
  HRM gate; fused gate/up or fused dispatch-combine kernels must include
  training backward validation, not just faster forward timing.
- Do not directly set `HRM_MOE_TRITON_BLOCK_M=256` with the current
  `num_stages=3` Triton grouped GEMM configs. The CUDA/bfloat16 gate job
  `hrm-moe-bm256-eq-06050045` failed at Triton compile time with shared memory
  `Required: 327704, Hardware limit: 232448`. Larger block-M exploration needs
  lower-stage configs or OOR-tolerant autotune first, then the normal
  CUDA/bfloat16 equivalence gate.
- `HRM_MOE_TRITON_BLOCK_M=256` with `num_stages=2` still OORed in
  `hrm-moe-bm256s2-eq-06050051` (`Required: 262160, Hardware limit: 232448`).
  `num_stages=1` passed the CUDA/bfloat16 gate
  (`hrm-moe-bm256s1-eq-06050054`) but the same-shape smoke
  `hrm-moe64x8-bm256s1-06050056` was slower than the current best
  (second-step core about 0.469s vs 0.389s). Do not adopt BM256 stage-1 as a
  default; prefer tuning the BM128 path or narrower grouped-GEMM configs next.
- When a long rjob is reading one worktree, use a separate git worktree for
  MoE speed code experiments. The 32-card run `hrm-moe32g-gt06050041` reads
  `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8`; speed tuning moved to
  `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8-tune` so code edits do
  not perturb the running training job.
- Evaluate MoE checkpoints from the same MoE worktree that produced them; the
  dense HRM-Text checkout may not have the experimental MoE architecture and
  config classes needed to load the checkpoint. The shared monitor script can
  be reused from the main checkout with
  `HRM_MONITOR_REPO_DIR=/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8`
  so rjob evals launch from this worktree and write summaries/docs here.
- When launching SFT from this MoE worktree, pass
  `repo_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8` explicitly.
  The copied `scripts/rjob_hrm_common.sh` still defaults to the dense checkout
  path, and forgetting `repo_dir` can start SFT from code that does not match
  the MoE checkpoint. For UltraData SFT, use
  `scripts/local_ultradata_moe64_sft_after_prepare.sh` so the prepared HRM SFT
  layout is header-validated before launch, `resume_epoch` is pinned to the
  requested stable checkpoint with all 32 carry files, and SFT starts from EMA
  weights with a fresh optimizer via `weights_only_resume_from_ema=true`.
  For the UltraData MoE64 SFT run, do not start from the currently latest
  checkpoint if it is only epoch 3; set `SFT_RESUME_EPOCH=4` and wait for
  `hrm-moe32g-sm16-06050339/fsdp2_epoch_4` to become stable.
- `resume_epoch` is not declared in the Hydra config schema. When the MoE SFT
  watcher passes it through `extra_args`, use the append form
  `+resume_epoch=<N>`; plain `resume_epoch=<N>` makes all training replicas
  fail during config composition before any real SFT work starts.
- For 32-card UltraData MoE SFT, do not reuse the 8-card
  `global_batch_size=32768`. HRM interprets this as total tokens, so 32 cards
  gives only 1024 tokens per rank while the prepared UltraData SFT samples
  require up to 4096 tokens after the autoregressive shift. That can produce an
  rjob `Succeeded` false positive with zero training steps and valid-looking
  checkpoints. Use at least `global_batch_size=131072` for 32 cards, and keep
  the `pretrain.py`/watcher local-batch guard in place.
- If `max_steps` is used to probe a faster UltraData MoE SFT batch size, keep
  it in an explicitly isolated probe job/checkpoint path and stop/delete it
  after the gate. Never carry `max_steps` into the formal SFT launch unless the
  user explicitly asks for a shortened run. The 2026-06-08 formal 32-card r4
  launch used `global_batch_size=262144`, `epochs=5`, and no `max_steps`; this
  halves total optimizer steps versus `131072` while preserving the requested
  five-epoch data pass, but still changes the effective batch-size/optimization
  tradeoff.
- Keep UltraData MoE SFT monitors alive after rjob submission. The watcher must
  inspect rjob state and final checkpoint artifacts until `fsdp2_epoch_5` plus
  all 32 `carry_epoch_5.*.pt` files are stable; a successful submit command is
  not enough. On failure, retry with suffixed job/checkpoint paths such as
  `hrm-moe64-sft-ultra0607-r2` and
  `checkpoints/hrm-moe64x8-ultradata-sft-0607-r2` so partial failed attempts do
  not obscure the next run.
- If an eval rjob must be interrupted to free GPUs for SFT, use
  `rjob stop <job_name>` first and `rjob stop --force <job_name>` if the job
  still appears active. `rjob get` can lag after the pod is deleted; check
  `rjob events <job_name>` for `Deleted pod` and wait for `rjob get` to show
  `Stopped` before assuming the GPU has been released.
- This worktree's `rjob_logs/` may be root-owned because many MoE rjobs wrote
  logs from containers. If a local monitor cannot create its state/log there,
  put the local monitor state and tee log in a writable checkout such as
  `/mnt/shared-storage-user/quxiaoye/HRM-Text/rjob_logs/`, while keeping
  `fanout.output_dir` under the MoE worktree for container-written summaries.
- Large checkpoint directories created by rjob containers may also be
  root-owned. After MoE speed measurements are documented, clear disposable
  speed-test checkpoints with a short CPU rjob that mounts `quxiaoye` and runs
  exact-path `rm -rf`; do not rely on login-node deletion. Preserve active
  long-run checkpoint paths such as `hrm-moe32g-sm16-06050339`.
- For Hugging Face model releases, do not upload native FSDP2 checkpoint
  directories as the public artifact unless explicitly requested. They include
  optimizer/EMA state and dozens of `.distcp` shards. Match the HRM-Text-1B
  release shape by extracting EMA model weights from the FSDP2 checkpoint,
  casting them to bf16, and publishing `model.safetensors` plus config,
  tokenizer, README, license, and optional assets. The 64x8 MoE epoch-4
  checkpoint contains about 21.7 GB of bf16 model weights and about 65 GB of
  optimizer/EMA storage in the native checkpoint.

## Local Validation

CPU-only checks can run locally, for example:

```bash
python scripts/test_prepare_sft_filter.py
```

GPU, distributed, FA3, and FSDP2 behavior should be validated through rjob.
