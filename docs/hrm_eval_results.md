# HRM 预训练实验与评测结果

最后更新：2026-06-04 18:32 HKT。

## 16 卡基线实验

Checkpoint 根目录：
`/mnt/shared-storage-user/quxiaoye/HRM-Text/checkpoints/hrm-pretrain-16g-xl-persistent-0602115537`

训练数据：
`/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097`

训练任务：

| 项目 | 值 |
| --- | --- |
| Job | `hrm-pretrain-16g-xl-persistent-0602115537` |
| 状态 | succeeded |
| GPUs | 16 张 H200，2 replicas x 8 GPUs |
| Config | `cfg_pretrain`, XL |
| 开始时间 | 2026-06-02 11:56:44 HKT |
| 结束时间 | 2026-06-04 08:32:30 HKT |
| 总耗时 | 44h 35m 46s |
| 稳态 epoch 耗时 | 约 11h 40m / epoch |

Checkpoint 完成时间：

| Epoch | Checkpoint 时间 |
| ---: | --- |
| 1 | 2026-06-02 21:32:43 HKT |
| 2 | 2026-06-03 09:13:47 HKT |
| 3 | 2026-06-03 20:53:33 HKT |
| 4 | 2026-06-04 08:32:29 HKT |

## 后续训练尝试

除非特别说明，以下任务使用同一份持久化 4-epoch sampled 数据：
`/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097`.

| 时间 | Job | 资源 | 超参 | 状态 | 记录 |
| --- | --- | --- | --- | --- | --- |
| 2026-06-04 17:48 HKT | `hrm-data-sample-e5-ctx4097-06041755` | CPU data-prep rjob | `epochs=5`, `context_size=4097` | stopped | 为测试 5 epoch 采样启动，随后实验目标改回 4 epoch 后取消。该任务产生了 root-owned 的半成品 `data_sampled_bpe_65k_e5_ctx4097/tokens.npy`；清理任务 `hrm-clean-e5-partial-tokens-06041759` 已成功删除。 |
| 2026-06-04 17:53 HKT | `hrm-pretrain-32g-xl-e4-gbs196k-06041753` | 32 张 H200，4 replicas x 8 GPUs | XL, `epochs=4`, `global_batch_size=196608`, `lr=2.2e-4`, `checkpoint_interval=1`, `WANDB_MODE=online` | failed | 已进入 `World Size 32` 并开始 epoch 1，但 rank 0 在 `wandb.init` 处因 `No API key configured` 失败。后续其他 rank 的 TCPStore/NCCL broken pipe 是 rank 0 退出后的连锁反应，不是首因。该 job 已停止，checkpoint 路径下没有留下 checkpoint 文件。 |
| 2026-06-04 18:00 HKT | `hrm-pre32g-xl-e4-off0604` | 32 张 H200，4 replicas x 8 GPUs | XL, `epochs=4`, `global_batch_size=196608`, `lr=2.2e-4`, `checkpoint_interval=1`, `WANDB_MODE=offline` | running | 使用相同 32 卡可比设置重新提交，但改为 W&B offline，避免依赖 API key。2026-06-04 18:04 HKT 确认 4 个 replica 均为 RUNNING；18:05 HKT 日志确认已进入 `World Size 32` 的 epoch 1。Checkpoint 将写入 `/mnt/shared-storage-user/quxiaoye/HRM-Text/checkpoints/hrm-pre32g-xl-e4-off0604`。 |

操作记录和经验：

- 如果目标是和 16 卡基线比较 wall-clock speedup，同时不改变 optimizer scale
  和 step 数，32 卡对照实验应保持 `global_batch_size=196608`。
- 如果不能确认 W&B credential 已在 rjob 环境中配置，训练任务使用
  `WANDB_MODE=offline`；否则 rank 0 会在 `wandb.init` 处失败，训练还没真正
  开始就退出。
- rjob 名字要短。`hrm-pretrain-32g-xl-e4-gbs196k-offline-06041800` 的
  dry-run 在提交前失败，因为生成的 task label 超过 Kubernetes 63 字符限制。

## 64 选 8 MoE 实验

本实验按用户要求在独立 worktree 中实现，不改动主分支：
`/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8`，分支
`codex/hrm-moe64x8`。

设计目标：

| 项目 | 值 |
| --- | --- |
| 基座 | HRM XL，H/L 各 16 层 |
| MoE 形式 | 参考 Qwen3 MoE 的 sparse FFN：router softmax 后 top-k dispatch |
| Experts | 64 |
| 每 token 选中 experts | 8 |
| Expert FFN intermediate | 512 |
| top-k 权重 | 默认归一化，便于替换 dense FFN 时保持输出尺度 |
| Aux loss | Qwen/Switch 风格 load-balancing loss，系数 0.001 |
| 训练调试策略 | 先用 8 卡 rjob、短 `max_steps` smoke 跑通分布式/FSDP/反向/optimizer |

参数量粗略估计：

| 项目 | 估计 |
| --- | ---: |
| Dense XL 原始总参数 | 约 1.18B |
| MoE XL 64x8 总参数 | 约 5.4B |
| 每 token 激活参数 | 约 1.18B |

实现记录：

- `models/layers.py` 新增 `SparseMoESwiGLU`，使用无 bias router、top-k expert
  dispatch、按 routing weight 加权输出，并记录 load-balancing aux loss。
- `models/transformer.py` 新增 `moe_*` 配置项；`moe_num_experts>0` 时将
  TransformerBlock 的 dense `SwiGLU` 替换为 `SparseMoESwiGLU`。
- `models/lm_head.py` 将 MoE aux loss 加到 CE loss 上，同时保留原始
  `train/loss` 作为 CE/token 指标，额外记录 `moe_aux_loss`、
  `moe_aux_loss_scaled`、`moe_total_loss`、`moe_max_expert_frac`。
- `pretrain.py` 新增 `max_steps` 和 `compile_train_batch`；MoE 配置会自动关闭
  `torch.compile`，因为 Qwen MoE 的动态 expert dispatch 对全图编译不友好。
- 新增配置 `config/arch/size/XL_moe64x8.yaml`，rjob 可用
  `arch_size=XL_moe64x8` 启动。

本地检查：

| 时间 | 检查 | 结果 | 备注 |
| --- | --- | --- | --- |
| 2026-06-04 18:27 HKT | `python -m py_compile ...` | passed | 覆盖 MoE 修改到的 Python 文件。 |
| 2026-06-04 18:27 HKT | 小 MoE 前后向脚本 | 未在登录环境运行 | 登录环境缺 `einops`；按仓库约定，真实训练环境以后续 rjob 结果为准。 |

Smoke 任务记录：

| 时间 | Job | 资源 | 参数 | 状态 | 记录 |
| --- | --- | --- | --- | --- | --- |
| 2026-06-04 18:29 HKT | `hrm-moe64x8-smk06041829` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=32768`, `epochs=1`, `max_steps=2`, `compile_train_batch=false`, `WANDB_MODE=offline` | failed | 未进入模型构建。首因是 Hydra struct 中没有 `max_steps`，普通 override `max_steps=2` 被拒绝；修复方式是在 `cfg_pretrain.yaml` / `cfg_sft.yaml` 中补 `compile_train_batch` 和 `max_steps` 默认项。 |

经验：

- 给 `PretrainConfig` 新增顶层字段时，要同步写入 Hydra YAML 默认配置；否则
  rjob 里用 `key=value` override 会报 `Key ... is not in struct`。也可以用
  `+key=value` 临时追加，但长期可复用参数应进入 YAML。

## 评测设置

| 项目 | 值 |
| --- | --- |
| 资源 | 每个 eval job 8 GPUs，fanout sharding |
| Fanout workers | 8 |
| Batch size | standard 和 MMLU-Pro eval 使用 16 |
| 数据缓存 | `eval_data_hf_parquet` |
| Standard config | `evaluation/config/hrm_benchmarking.yaml` |
| MMLU-Pro config | `evaluation/config/hrm_mmlu_pro_benchmarking.yaml` |
| AIME config | `evaluation/config/hrm_maj_vote_benchmarking.yaml` |

Standard config 包含：
GSM8k, MATH, DROP, MMLU, ARC, HellaSwag, Winogrande, BoolQ.

## 评测任务索引

| Eval set | Epoch | Job | 状态 | Summary |
| --- | ---: | --- | --- | --- |
| Standard | 1 | `hrm-eval-e1-localdata-r4-0603` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260603_025332/summary.json` |
| Standard | 2 | `hrm-eval-e2-localdata-0603` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260603_032940/summary.json` |
| Standard | 3 | `hrm-eval-e3-localdata-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_022538/summary.json` |
| Standard | 4 | `hrm-eval-e4-localdata-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_025213/summary.json` |
| MMLU-Pro | 1 | `hrm-eval-e1-mmlupro-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_031827/summary.json` |
| MMLU-Pro | 2 | `hrm-eval-e2-mmlupro-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_033129/summary.json` |
| MMLU-Pro | 3 | `hrm-eval-e3-mmlupro-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_033142/summary.json` |
| MMLU-Pro | 4 | `hrm-eval-e4-mmlupro-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_033704/summary.json` |
| AIME25 | 1 | `hrm-eval-e1-aime25-0603` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260603_062938/summary.json` |
| AIME25 | 2 | `hrm-eval-e2-aime25-seq-0603` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260603_093009/summary.json` |
| AIME25 | 3 | `hrm-eval-e3-aime25-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_065156/summary.json` |
| AIME25 | 4 | `hrm-eval-e4-aime25-seq-0604` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260604_070045/summary.json` |

## 主指标

数值均为百分比。DROP 同时记录 exact match 和 F1。

| Benchmark | Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8k | acc | 64.82 | 79.53 | 81.43 | 85.06 |
| MATH | acc | 39.90 | 50.98 | 55.16 | 56.12 |
| DROP | em | 61.56 | 74.00 | 77.17 | 78.41 |
| DROP | f1 | 65.18 | 77.61 | 80.76 | 82.16 |
| MMLU | acc | 43.29 | 54.32 | 58.11 | 60.73 |
| ARC | acc | 50.00 | 72.53 | 79.10 | 82.42 |
| HellaSwag | acc | 34.52 | 47.79 | 57.97 | 64.13 |
| Winogrande | acc | 57.38 | 65.04 | 69.77 | 73.40 |
| BoolQ | acc | 74.28 | 84.19 | 85.57 | 86.73 |
| MMLU-Pro | acc | 19.57 | 27.06 | 30.86 | 33.11 |

## Invalid Rate

数值均为百分比。

| Benchmark | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | ---: | ---: | ---: | ---: |
| GSM8k | 2.05 | 1.29 | 1.21 | 1.06 |
| MATH | 9.26 | 7.04 | 5.52 | 5.22 |
| MMLU | 0.18 | 0.25 | 0.14 | 0.23 |
| ARC | 0.00 | 0.00 | 0.00 | 0.00 |
| HellaSwag | 0.00 | 0.00 | 0.00 | 0.00 |
| Winogrande | 0.00 | 0.00 | 0.00 | 0.00 |
| BoolQ | 0.00 | 0.00 | 0.00 | 0.00 |
| MMLU-Pro | 5.05 | 2.48 | 3.87 | 2.84 |

当前 summary 中 DROP 没有 invalid rate。

## AIME25 Majority Voting

数值均为百分比。

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | ---: | ---: | ---: | ---: |
| maj_pass@1 | 3.33 | 16.67 | 13.33 | 23.33 |
| maj_pass@10 | 16.67 | 36.67 | 30.00 | 33.33 |
| maj_pass@100 | 46.67 | 56.67 | 60.00 | 50.00 |
| pass@1 | 0.90 | 3.98 | 3.62 | 5.59 |
| pass@10 | 7.13 | 18.67 | 18.29 | 22.01 |
| pass@100 | 29.17 | 44.16 | 44.16 | 41.58 |

## 标准 Benchmark 样本数

| Benchmark | n |
| --- | ---: |
| GSM8k | 1,319 |
| MATH | 5,000 |
| DROP | 9,536 |
| MMLU | 57 个聚合 subject |
| ARC | 1,172 |
| HellaSwag | 10,042 |
| Winogrande | 1,267 |
| BoolQ | 3,270 |
| MMLU-Pro | 12,032 |
| AIME25 | 30 |

对于 MMLU，顶层 summary 的 `n` 是聚合后的 subject 数；每个 subject 的样本数
可以在原始 benchmark 日志里查。

## 结论和备注

- 标准 benchmark suite 在上述所有主指标上，从 epoch 1 到 epoch 4 单调提升。
- MMLU-Pro 从 epoch 1 的 19.57% 提升到 epoch 4 的 33.11%。
- AIME25 上，epoch 4 的 `maj_pass@1`、`pass@1` 和 `pass@10` 最好；epoch 3
  的 `maj_pass@100` / `pass@100` 最好，因此高采样 majority-vote 指标并不
  随 epoch 单调变化。
- 当前训练数据是 sampled HRM/Data IO instruction-response PrefixLM 数据，不是
  针对评测测试集临时构造的一次性数据。
- 后续所有训练、评测、失败尝试、清理动作和分析结论都用中文追加到本文档。
