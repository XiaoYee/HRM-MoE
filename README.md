![](./assets/banner.png)

# HRM-MoE

HRM-MoE is a Mixture-of-Experts fork of
[sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text). It keeps the
original HRM-Text training stack, PrefixLM packing, FlashAttention 3 path,
FSDP2 checkpointing, evaluation tooling, and data workflow, then adds sparse
MoE feed-forward blocks for HRM-style language-model pretraining and SFT.

The main experimental target in this checkout is an XL HRM model with 64
experts and top-k 8 routing. Each active expert uses an intermediate width of
512, so the active MoE FFN width matches the dense XL SwiGLU width of 4096.

## What Changed

- Qwen-style sparse MoE FFN with fp32 router softmax, top-k routing, optional
  top-k probability renormalization, load-balancing auxiliary loss, and expert
  count metrics.
- Multiple expert compute backends:
  - `origin`: reference per-expert ModuleList implementation.
  - `shard`: packed expert shards.
  - `grouped`: padded `torch.bmm` grouped expert compute.
  - `grouped_triton`: unpadded Triton grouped GEMM expert compute.
  - `grouped_cutlass`: CUTLASS grouped GEMM expert compute.
  - `grouped_ep`: expert-parallel all-to-all dispatch across distributed ranks.
- MoE architecture presets under [`config/arch/size`](config/arch/size), led by
  [`XL_moe64x8_grouped_triton.yaml`](config/arch/size/XL_moe64x8_grouped_triton.yaml).
- MoE equivalence tests for local and distributed expert implementations.
- Optional MoE profiling via `HRM_MOE_PROFILE=1`, with forward/backward phase
  timing printed by `pretrain.py`.
- rjob launch wrappers for the PJLab/rjob training environment used by this
  working checkout.

The dense HRM-Text path is still present. Select a MoE size config with Hydra to
enable sparse experts.

## Repository Layout

```text
HRM-MoE/
|-- config/                       # Hydra configs for model, data, and training
|-- config/arch/size/*moe*.yaml   # MoE size presets
|-- conversion/convert_to_hf.py    # FSDP2 checkpoint -> HF-style export
|-- evaluation/                    # Evaluation engines, benchmark wrappers, configs
|-- models/                        # HRM, Transformer blocks, dense layers, MoE layers
|-- models/moe_*                   # MoE kernels and profiling helpers
|-- docker/                        # Tested CUDA/PyTorch/FlashAttention environment
|-- scripts/                       # Data prep, rjob launchers, eval, MoE equivalence tests
|-- dataset_new.py                 # PrefixLM packed dataset loader
|-- multipack_sampler.py           # Distributed multipack batch sampler
|-- pretrain.py                    # FSDP2 pretraining/SFT entrypoint
|-- simple_inference_engine.py     # Native checkpoint inference helper
`-- requirements.txt
```

## Environment

Hopper-class GPUs are the expected target because the training attention path
depends on FlashAttention 3.

The upstream tested image is:

```bash
docker run --gpus all --ipc=host --network=host -it \
  -v "$PWD":/workspace \
  sapientai/hrm-text:latest
```

If you are installing manually, follow the tested CUDA, PyTorch, and
FlashAttention versions in [`docker/Dockerfile`](docker/Dockerfile), then run:

```bash
pip install -r requirements.txt
```

Some MoE backends need extra runtime support:

- `grouped_triton` uses Triton kernels from
  [`models/moe_triton_grouped_gemm.py`](models/moe_triton_grouped_gemm.py).
- `grouped_cutlass` uses the CUTLASS grouped GEMM helper in
  [`models/moe_cutlass_grouped_gemm.py`](models/moe_cutlass_grouped_gemm.py).
- `grouped_ep` requires initialized `torch.distributed` collectives.

## Data

HRM-MoE uses the same sampled tokenized layout as HRM-Text. The training
`data.path` directory must contain:

```text
metadata.json
tokens.npy
epoch_<n>/inst_start.npy
epoch_<n>/inst_len.npy
epoch_<n>/resp_start.npy
epoch_<n>/resp_len.npy
```

The data is normally produced by the companion
[sapientinc/data_io](https://github.com/sapientinc/data_io) pipeline:

```bash
cd <DATA_IO_PATH>
python sample_tokenized.py epochs=4 output_path=/path/to/sampled > show_analytics.md
```

For reusable experiments, keep sampled data on shared storage and pass it with
`data.path=/path/to/sampled`. Avoid relying on `/dev/shm` unless training and
sampling happen in the same container.

## Train

The recommended 64x8 MoE preset is `XL_moe64x8_grouped_triton`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 pretrain.py \
  arch/size@arch=XL_moe64x8_grouped_triton \
  data.path=/path/to/sampled \
  global_batch_size=196608
```

For multi-node runs, launch the same command on each node with the usual
`torchrun` rendezvous arguments:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun \
  --nproc_per_node=8 \
  --nnodes=<NUM_NODES> \
  --node_rank=<NODE_RANK> \
  --master_addr=<MASTER_ADDR> \
  --master_port=<MASTER_PORT> \
  pretrain.py \
  arch/size@arch=XL_moe64x8_grouped_triton \
  data.path=/path/to/sampled
```

MoE-specific training notes:

- `compile_train_batch` remains available, but `pretrain.py` disables compile
  for MoE by default unless `allow_compile_moe=true`.
- `fsdp_wrap_moe_experts=true` wraps packed expert parameters separately when
  possible.
- `HRM_MOE_TRITON_AUTOTUNE=1` enables Triton kernel autotuning.
- `HRM_MOE_TRITON_SM_MARGIN=16` reserves SMs for non-GEMM work in the Triton
  grouped backend.
- `HRM_MOE_PROFILE=1` prints MoE phase timings for router, dispatch, grouped
  GEMMs, activation, combine, aux metrics, and expert-parallel communication.

PJLab/rjob convenience wrappers are kept in [`scripts`](scripts):

```bash
num_gpus=8 arch_size=XL_moe64x8_grouped_triton \
  data_path=/path/to/sampled \
  bash scripts/rjob_hrm_pretrain.sh
```

Use `dry_run=true` with a wrapper before launching a real cluster job.

## SFT

Full-parameter SFT uses the same `pretrain.py` entrypoint with
`--config-name cfg_sft`:

```bash
python scripts/prepare_sft_data.py \
  --train input.jsonl \
  --tokenizer /path/to/tokenizer.json \
  --output /path/to/sft_data \
  --epochs 5

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 pretrain.py \
  --config-name cfg_sft \
  arch/size@arch=XL_moe64x8_grouped_triton \
  data.path=/path/to/sft_data \
  resume_from=/path/to/pretrain_ckpt \
  +checkpoint_path=/path/to/sft_out
```

`--epochs` for data preparation must match the SFT training config. Use
`weights_only_resume_from_ema=true` when fine-tuning from a pretrain EMA while
starting with a fresh optimizer.

## Verify MoE Implementations

Run the CPU equivalence check locally:

```bash
python scripts/test_moe_shard_equivalence.py
```

Run the distributed expert-parallel equivalence check with torchrun:

```bash
torchrun --nproc_per_node=8 scripts/test_moe_ep_distributed_equivalence.py
```

On the rjob cluster:

```bash
bash scripts/rjob_hrm_moe_equiv.sh
```

For CUDA/bfloat16 checks, set the `MOE_EQUIV_*` environment variables used by
the test scripts, for example `MOE_EQUIV_DEVICE=cuda`.

## Evaluate

Evaluation loads the latest checkpoint epoch automatically when `ckpt_epoch` is
not provided:

```bash
python -m evaluation.main ckpt_path=/path/to/checkpoint_dir
```

For selected benchmarks:

```bash
python -m evaluation.main \
  ckpt_path=/path/to/checkpoint_dir \
  run_only='[GSM8k,MATH]' \
  generation_config.batch_size=16
```

The 8-GPU fanout entrypoint shards prompts within each benchmark:

```bash
ckpt_path=/path/to/checkpoint_dir \
num_gpus=8 \
batch_size=16 \
entrypoint=scripts/hrm_eval_fanout_entrypoint.sh \
bash scripts/rjob_hrm_eval.sh
```

## Model Configurations

Architectures live under [`config/arch/net`](config/arch/net):

| Config | Model |
| --- | --- |
| `hrm` | HRM-Text / HRM-MoE backbone |
| `transformer` | Standard Transformer wrapper |
| `trm` | Tiny Recursive Model baseline |
| `trm_match_recurrence` | TRM configured to match HRM recurrence with half parameters |
| `rins` | Recursive Inference Scaling baseline |
| `ut` | Universal Transformer baseline |

MoE size presets live under [`config/arch/size`](config/arch/size):

| Config | Experts | Top-k | Backend |
| --- | ---: | ---: | --- |
| `XL_moe64x8` | 64 | 8 | reference per-expert |
| `XL_moe64x8_shard` | 64 | 8 | packed expert shards |
| `XL_moe64x8_grouped` | 64 | 8 | padded grouped `torch.bmm` |
| `XL_moe64x8_grouped_triton` | 64 | 8 | Triton grouped GEMM |
| `XL_moe64x8_grouped_cutlass` | 64 | 8 | CUTLASS grouped GEMM |
| `XL_moe64x8_grouped_ep` | 64 | 8 | expert-parallel all-to-all |

Dense HRM-Text size presets remain available:

| Config | Layers | Hidden | Heads |
| --- | ---: | ---: | ---: |
| `B` | 12 | 1024 | 8 |
| `L` | 24 | 1280 | 10 |
| `XL` | 32 | 1536 | 12 |
| `XXL` | 72 | 1792 | 14 |
| `XXL_wide` | 32 | 2560 | 20 |

For HRM and RINS, `half_layers: true` splits the configured layer count evenly
between the H and L modules.

## Technical Notes

- [`models/layers.py`](models/layers.py) contains the dense SwiGLU path and the
  sparse MoE implementations.
- [`models/moe_triton_grouped_gemm.py`](models/moe_triton_grouped_gemm.py)
  implements the Triton grouped GEMM backend.
- [`models/moe_cutlass_grouped_gemm.py`](models/moe_cutlass_grouped_gemm.py)
  contains the optional CUTLASS grouped backend wrapper.
- [`models/moe_profile.py`](models/moe_profile.py) records optional CUDA event
  timings for MoE phases.
- [`dataset_new.py`](dataset_new.py) loads PrefixLM packed samples and emits
  FlashAttention sequence metadata.
- [`multipack_sampler.py`](multipack_sampler.py) implements distributed
  multipack batching.
- [`models/flash_attention_prefixlm_v2.py`](models/flash_attention_prefixlm_v2.py)
  implements the two-pass PrefixLM attention path.
- [`models/lm_head.py`](models/lm_head.py) attaches scaled embeddings, the
  output head, losses, token accuracy, and sequence exact accuracy.
- [`pretrain.py`](pretrain.py) handles Hydra config, FSDP2 wrapping, optimizer
  creation, LR schedule, W&B logging, code/config snapshots, and checkpointing.

## Upstream

This repository is derived from HRM-Text:

- Paper: [HRM-Text: Efficient Pretraining Beyond Scaling](https://arxiv.org/abs/2605.20613)
- Upstream code: [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text)
- Upstream model: [sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B)

If you use the original HRM-Text work, please cite:

```bibtex
@misc{wang2026hrmtextefficientpretrainingscaling,
      title={HRM-Text: Efficient Pretraining Beyond Scaling},
      author={Guan Wang and Changling Liu and Chenyu Wang and Cai Zhou and Yuhao Sun and Yifei Wu and Shuai Zhen and Luca Scimeca and Yasin Abbasi Yadkori},
      year={2026},
      eprint={2605.20613},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.20613},
}
```

## License

Apache License 2.0
