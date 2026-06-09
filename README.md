![](./assets/banner.png)

<h1 align="center">HRM-MoE: Efficient Sparse Pretraining with Hierarchical Reasoning</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.20613"><img src="https://img.shields.io/badge/Base-HRM--Text-red?logo=arxiv&logoColor=white" alt="HRM-Text Paper"></a>
  <a href="https://huggingface.co/Xiaoye08/HRM-MoE"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow" alt="HRM-MoE Model"></a>
  <a href="https://github.com/XiaoYee/HRM-MoE"><img src="https://img.shields.io/badge/Code-HRM--MoE-181717?logo=github&logoColor=white" alt="HRM-MoE Code"></a>
</p>

## Model Structure Comparison

HRM-MoE is a sparse Mixture-of-Experts extension of
[sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text). It keeps the
original HRM-Text recipe for hierarchical recurrent modeling, PrefixLM sequence
packing, FlashAttention 3, PyTorch FSDP2 training, checkpointing, evaluation,
and conversion, while replacing the dense FFN path with a routed MoE FFN.

| Config | Layers | Hidden | Heads | FFN / experts | Active FFN width | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| HRM-Text `XL` dense | 32 | 1536 | 12 | dense SwiGLU, intermediate 4096 | 4096 | ~1.18B |
| HRM-MoE [`XL_moe64x8_grouped_triton`](config/arch/size/XL_moe64x8_grouped_triton.yaml) | 32 | 1536 | 12 | 64 routed SwiGLU experts, top-k 8, expert width 512 | `8 x 512 = 4096` | ~5.41B |

The 64x8 preset keeps the per-token active FFN width aligned with the dense XL
HRM-Text FFN while giving the model a larger sparse expert pool. The final
expert path uses fp32 router softmax, normalized top-k routing, auxiliary
load-balancing loss, and grouped Triton GEMMs for expert compute.

## 32-GPU Pretraining Results

The table below compares the 32-GPU dense XL run against the 32-GPU HRM-MoE
64x8 run on the same sampled HRM pretraining data and `global_batch_size=196608`.
Both runs use 4 epochs, and the MoE column reports the completed epoch-4
checkpoint evaluation.

| Benchmark | Metric | Dense XL epoch 4 | HRM-MoE 64x8 epoch 4 |
| --- | --- | ---: | ---: |
| GSM8k | acc | 83.93 | 84.99 (e4) |
| MATH | acc | 54.96 | 60.08 (e4) |
| DROP | em | 79.45 | 80.86 (e4) |
| DROP | f1 | 83.06 | 84.53 (e4) |
| MMLU | acc | 61.38 | 61.18 (e4) |
| ARC | acc | 83.02 | 87.80 (e4) |
| HellaSwag | acc | 61.96 | 73.89 (e4) |
| Winogrande | acc | 71.98 | 73.88 (e4) |
| BoolQ | acc | 87.25 | 88.75 (e4) |
| MMLU-Pro | acc | 32.72 | 37.57 (e4) |
| AIME25 | maj_pass@1 | 13.33 | 16.67 (e4) |
| AIME25 | maj_pass@10 | 36.67 | 36.67 (e4) |
| AIME25 | maj_pass@100 | 53.33 | 56.67 (e4) |


## Launch the MoE Pretraining

### Required Resources

The intended training target is Hopper-class GPUs because the attention path
depends on FlashAttention 3 and the final MoE expert path uses Triton kernels.

The final MoE preset is designed for multi-GPU pretraining:

| Model | Preset | GPUs | Notes |
| --- | --- | ---: | --- |
| HRM-MoE XL 64x8 | `XL_moe64x8_grouped_triton` | 8+ H100/H200 | use shared storage for data and checkpoints |

### 1. Prepare Data

HRM-MoE trains from the same sampled, tokenized data layout as HRM-Text. The
training `data.path` directory must contain:

```text
metadata.json
tokens.npy
epoch_<n>/inst_start.npy
epoch_<n>/inst_len.npy
epoch_<n>/resp_start.npy
epoch_<n>/resp_len.npy
```

Prepare sampled data with the companion
[sapientinc/data_io](https://github.com/sapientinc/data_io) pipeline:

```bash
cd <DATA_IO_PATH>
python sample_tokenized.py epochs=4 output_path=/path/to/sampled > show_analytics.md
```

For reusable experiments, keep sampled data on shared storage and pass it with
`data.path=/path/to/sampled`. Only use `/dev/shm` for short same-container
debugging.

### 2. Start the Environment

The upstream tested Docker image is:

```bash
docker run --gpus all --ipc=host --network=host -it \
  -v "$PWD":/workspace \
  sapientai/hrm-text:latest
```

If you install from source, follow the tested CUDA, PyTorch, and FlashAttention
versions in [`docker/Dockerfile`](docker/Dockerfile), then run:

```bash
pip install -r requirements.txt
```

For multi-node training, mount the same workspace and checkpoint path on every
node. Verify NCCL before starting a long job.

### 3. Launch Pretraining

Single-node example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HRM_MOE_TRITON_AUTOTUNE=1 \
HRM_MOE_TRITON_SM_MARGIN=16 \
torchrun --nproc_per_node=8 pretrain.py \
  arch/size@arch=XL_moe64x8_grouped_triton \
  data.path=/path/to/sampled \
  global_batch_size=196608
```

Multi-node example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HRM_MOE_TRITON_AUTOTUNE=1 \
HRM_MOE_TRITON_SM_MARGIN=16 \
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

Useful MoE training switches:

- `allow_compile_moe=false` is the default; sparse routing currently runs in
  eager mode for stability.
- `fsdp_wrap_moe_experts=true` wraps packed expert parameters separately when
  possible.
- `HRM_MOE_PROFILE=1` prints MoE phase timings for router, dispatch, grouped
  GEMMs, activation, combine, and auxiliary metrics.

On the rjob cluster, the same final preset can be launched through the wrapper:

```bash
num_gpus=8 \
arch_size=XL_moe64x8_grouped_triton \
data_path=/path/to/sampled \
bash scripts/rjob_hrm_pretrain.sh
```

Use `dry_run=true` before submitting a real rjob launch.

### 4. Evaluate

Evaluation loads the latest checkpoint epoch automatically when `ckpt_epoch` is
not provided:

```bash
python -m evaluation.main ckpt_path=/path/to/checkpoint_dir
```

To run a benchmark subset and lower memory use:

```bash
python -m evaluation.main \
  ckpt_path=/path/to/checkpoint_dir \
  run_only='[GSM8k,MATH]' \
  generation_config.batch_size=16
```

For 8-GPU fanout evaluation on rjob:

```bash
ckpt_path=/path/to/checkpoint_dir \
num_gpus=8 \
batch_size=16 \
entrypoint=scripts/hrm_eval_fanout_entrypoint.sh \
bash scripts/rjob_hrm_eval.sh
```

## Fine-Tuning (SFT)

Continue from a pretrain checkpoint on instruction data. Full-parameter SFT uses
the same `pretrain.py` entrypoint with `--config-name cfg_sft`.

Input is a JSONL file with one object per line:

```json
{"instruction": "<full prompt>", "response": "<expected output>", "condition": "direct"}
```

Prepare SFT data:

```bash
python scripts/prepare_sft_data.py \
  --train input.jsonl \
  --tokenizer /path/to/tokenizer.json \
  --output /path/to/sft_data \
  --epochs 5
```

Launch SFT:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HRM_MOE_TRITON_AUTOTUNE=1 \
HRM_MOE_TRITON_SM_MARGIN=16 \
torchrun --nproc_per_node=8 pretrain.py \
  --config-name cfg_sft \
  arch/size@arch=XL_moe64x8_grouped_triton \
  data.path=/path/to/sft_data \
  resume_from=/path/to/pretrain_ckpt \
  +checkpoint_path=/path/to/sft_out
```

`--epochs` for data preparation must match the SFT training config. Add
`weights_only_resume_from_ema=true` when fine-tuning from pretrain EMA weights
with a fresh optimizer.

## Verify the Final MoE Path

Run a local equivalence smoke test:

```bash
python scripts/test_moe_shard_equivalence.py
```

Run the CUDA/rjob MoE gate before trusting a kernel or routing change:

```bash
bash scripts/rjob_hrm_moe_equiv.sh
```

## Repository Layout

```text
HRM-MoE/
|-- config/                       # Hydra configs for model, data, and training
|-- config/arch/size/XL_moe64x8_grouped_triton.yaml
|-- conversion/convert_to_hf.py    # FSDP2 checkpoint -> HF-style export
|-- evaluation/                    # Evaluation engines, benchmark wrappers, configs
|-- models/layers.py               # Attention, dense FFN, and final sparse MoE layer
|-- models/moe_triton_grouped_gemm.py
|-- models/moe_profile.py
|-- docker/                        # Tested CUDA/PyTorch/FlashAttention environment
|-- scripts/                       # Data prep, rjob launchers, eval, validation
|-- dataset_new.py                 # PrefixLM packed dataset loader
|-- multipack_sampler.py           # Distributed multipack batch sampler
|-- pretrain.py                    # FSDP2 pretraining/SFT entrypoint
`-- simple_inference_engine.py     # Native checkpoint inference helper
```

## Technical Notes

- [`models/layers.py`](models/layers.py) contains the final sparse MoE FFN path:
  fp32 router softmax, top-k 8 routing, grouped Triton expert compute, weighted
  combine, and auxiliary load-balancing loss.
- [`models/moe_triton_grouped_gemm.py`](models/moe_triton_grouped_gemm.py)
  implements the grouped Triton expert GEMM path used by the final preset.
- [`models/moe_profile.py`](models/moe_profile.py) records optional CUDA event
  timings when `HRM_MOE_PROFILE=1`.
- [`dataset_new.py`](dataset_new.py) loads PrefixLM packed samples and emits
  FlashAttention sequence metadata.
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
