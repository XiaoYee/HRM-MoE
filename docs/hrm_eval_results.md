# HRM 预训练实验与评测结果

最后更新：2026-06-06 20:22:30 HKT。

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
  dispatch、按 routing weight 加权输出，并记录 load-balancing aux loss。后续补充
  `SparseMoEExpertCollection` / `SparseMoEExpertShard`，支持 origin 与 shard
  两种等价实现：origin 是逐 expert `SwiGLU`，shard 是每 8 个 expert 堆叠成一个
  有 `forward()` 的 module，便于 FSDP2 单独包 expert 参数。
- `models/transformer.py` 新增 `moe_*` 配置项；`moe_num_experts>0` 时将
  TransformerBlock 的 dense `SwiGLU` 替换为 `SparseMoESwiGLU`。
- `models/lm_head.py` 将 MoE aux loss 加到 CE loss 上，同时保留原始
  `train/loss` 作为 CE/token 指标，额外记录 `moe_aux_loss`、
  `moe_aux_loss_scaled`、`moe_total_loss`、`moe_max_expert_frac`。
- `pretrain.py` 新增 `max_steps` 和 `compile_train_batch`；MoE 配置会自动关闭
  `torch.compile`，因为 Qwen MoE 的动态 expert dispatch 对全图编译不友好。
  后续补充 `fsdp_wrap_moe_experts`，按 pith-train 的 MoE/FSDP 思路先包
  `layer.mlp.experts`，再包完整 `TransformerBlock`；同时增加
  `reduce_metrics_start/done` 等 profile 点，定位 step 后卡住问题。
- 新增配置 `config/arch/size/XL_moe64x8.yaml`，rjob 可用
  `arch_size=XL_moe64x8` 启动。
- 新增配置 `config/arch/size/XL_moe64x8_shard.yaml`，保持 64 选 8 不变，
  但使用 `moe_implementation=shard`、`moe_expert_in_one_shard=8`。
- 新增配置 `config/arch/size/XL_moe64x8_grouped.yaml`，保持 64 选 8 不变，
  但使用 `moe_implementation=grouped`。第一版尝试 ATen `grouped_mm`
  前向很快但反向卡住；最终采用 padded batched GEMM：按 expert 稳定排序
  token/top-k assignment，填充成 `[num_experts, max_tokens_per_expert, hidden]`，
  用两次 `torch.bmm` 完成 gate/up 和 down，再 scatter 回原 token。

本地检查：

| 时间 | 检查 | 结果 | 备注 |
| --- | --- | --- | --- |
| 2026-06-04 18:27 HKT | `python -m py_compile ...` | passed | 覆盖 MoE 修改到的 Python 文件。 |
| 2026-06-04 18:27 HKT | 小 MoE 前后向脚本 | 未在登录环境运行 | 登录环境缺 `einops`；按仓库约定，真实训练环境以后续 rjob 结果为准。 |
| 2026-06-04 19:21 HKT | `python scripts/test_moe_shard_equivalence.py` | passed | 本地补用户态 `einops` 并在测试脚本中 stub FlashAttention；比较 origin/shard 的 forward、aux loss、expert counts、输入梯度、router 梯度和每个 expert 权重梯度，均通过。 |
| 2026-06-04 19:21 HKT | `python -m py_compile ...` | passed | 覆盖 shard/FSDP 修改后的 `models/*.py`、`pretrain.py` 和等价测试脚本。 |
| 2026-06-04 19:35 HKT | `python -m py_compile ...` | passed | dense 对照前复查 MoE 相关 Python 文件语法。 |
| 2026-06-04 19:35 HKT | `python scripts/test_moe_shard_equivalence.py` | passed | 再次确认 origin/shard 的前向、aux loss、expert counts、输入梯度、router 梯度和 expert 梯度等价。 |
| 2026-06-04 19:54 HKT | `python -m py_compile ...` | passed | 覆盖 grouped expert compute 修改后的 `models/layers.py`、`models/transformer.py` 和等价测试脚本。 |
| 2026-06-04 19:54 HKT | `python scripts/test_moe_shard_equivalence.py` | passed | 升级为 origin/shard/grouped 三路等价测试，覆盖 forward、aux loss、expert counts、输入梯度、router 梯度和 expert 权重梯度。 |
| 2026-06-04 20:01 HKT | `python scripts/test_moe_shard_equivalence.py` | passed | grouped 训练路径从 ATen `grouped_mm` 切到 padded `torch.bmm` 后再次确认三路精度等价。 |

Smoke 任务记录：

| 时间 | Job | 资源 | 参数 | 状态 | 记录 |
| --- | --- | --- | --- | --- | --- |
| 2026-06-04 18:29 HKT | `hrm-moe64x8-smk06041829` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=32768`, `epochs=1`, `max_steps=2`, `compile_train_batch=false`, `WANDB_MODE=offline` | failed | 未进入模型构建。首因是 Hydra struct 中没有 `max_steps`，普通 override `max_steps=2` 被拒绝；修复方式是在 `cfg_pretrain.yaml` / `cfg_sft.yaml` 中补 `compile_train_batch` 和 `max_steps` 默认项。 |
| 2026-06-04 18:33 HKT | `hrm-moe64x8-smk2-06041833` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=32768`, `epochs=1`, `max_steps=2`, `log_interval=1`, `compile_train_batch=false`, `WANDB_MODE=offline` | failed | 仍未进入模型构建。首因是 Hydra struct 中没有 `log_interval`，普通 override `log_interval=1` 被拒绝；修复方式是在 `cfg_pretrain.yaml` / `cfg_sft.yaml` 中补 `log_interval` 默认项。 |
| 2026-06-04 18:35 HKT | `hrm-moe64x8-smk3-06041835` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=32768`, `epochs=1`, `max_steps=2`, `log_interval=1`, `compile_train_batch=false`, `WANDB_MODE=offline` | stopped | 已进入 `World Size 8` 的 epoch 1，说明 Hydra、模型构建、FSDP 初始化通过；但每卡 4096 token 的首个 step 超过 5 分钟未完成，先停止，缩小 batch 继续调通链路。 |
| 2026-06-04 18:42 HKT | `hrm-moe64x8-smk4-06041842` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=8192`, `epochs=1`, `max_steps=1`, `log_interval=1`, `compile_train_batch=false`, `WANDB_MODE=offline` | failed | 已进入 epoch 1，但在第一个 step 的 `update_lr` 失败：`total_steps=1` 且 `lr_warmup_steps=1` 时 cosine 分支分母为 0；修复方式是让 warmup 分支覆盖 `step <= warmup_steps`，并让 decay 分母至少为 1。 |
| 2026-06-04 18:46 HKT | `hrm-moe64x8-smk5-06041845` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=8192`, `epochs=1`, `max_steps=1`, `log_interval=1`, `compile_train_batch=false`, `WANDB_MODE=offline` | stopped | 已进入 `World Size 8` 的 epoch 1；每卡 1024 token 的首个 step 超过约 2.5 分钟未完成且日志无新增。为避免盲等，先停止并加入 `profile_train_batch` 打点，用下一轮定位 forward/backward/optimizer 慢点。 |
| 2026-06-04 18:52 HKT | `hrm-moe64x8-prof06041852` | 8 x H200 | `arch_size=XL_moe64x8`, `global_batch_size=8192`, `max_steps=1`, `log_interval=1`, `profile_train_batch=true` | stopped | profile 显示 `forward_done elapsed=5.160s`，但没有 `backward_done`。首因不是 forward 路由，而是 smoke 里 `max_steps=1` 让 `total_steps=1`，HRM bp warmup 第一轮直接使用 `bp_steps=5`，反传图过深；后续 smoke 固定 `bp_steps=2`。 |
| 2026-06-04 19:05 HKT | `hrm-moe64x8-basic-bp2-06041905` | 8 x H200 | origin MoE, `global_batch_size=1024`, `+arch.bp_min_steps=2`, `arch.bp_max_steps=2` | failed | 未进入训练。`arch.bp_min_steps` 不在 Hydra YAML struct 中，普通 override 失败；改用 `+arch.bp_min_steps=2`。 |
| 2026-06-04 19:08 HKT | `hrm-moe64x8-basic-bp2-06041908` | 8 x H200 | origin MoE, `global_batch_size=1024`, `bp_steps=2` | succeeded | 该任务退出成功并保存 checkpoint，但无 `TrainProfile` / metrics，进度条停在 `0/1`；判断为 local batch 过小导致没有有效训练 batch，不能算训练链路跑通。 |
| 2026-06-04 19:11 HKT | `hrm-moe64x8-basic-bp2-gb8192-06041911` | 8 x H200 | origin MoE, `global_batch_size=8192`, `bp_steps=2`, `log_interval=1` | stopped | 已完成实际 step：forward 4.255s、backward 2.680s、optimizer 0.940s、zero-grad 0.004s；但 step 后没有打印 `Reached max_steps`，卡在 metrics reduce 附近。修复：MoE metrics 在所有 rank 固定 key；增加 reduce profile。 |
| 2026-06-04 19:22 HKT | `hrm-moe64x8-shard-bp2-gb8192-06041922` | 8 x H200 | shard MoE, `global_batch_size=8192`, `bp_steps=2` | failed | FSDP2 不能直接 `fully_shard(ModuleList)`，报 `does not support containers that do not implement forward`。修复：引入有 `forward()` 的 `SparseMoEExpertCollection` 包住 expert shards。 |
| 2026-06-04 19:25 HKT | `hrm-moe64x8-shard-bp2-gb8192-06041925` | 8 x H200 | shard MoE, `global_batch_size=8192`, `bp_steps=2`, `log_interval=999` | succeeded | shard + expert FSDP 包装完整跑通并自然退出：forward 3.887s、backward 2.851s、optimizer 0.230s、zero-grad 0.002s，打印 `Reached max_steps=1`。参数量 5,413,797,888。 |
| 2026-06-04 19:27 HKT | `hrm-moe64x8-shard-reduce-bp2-gb8192-06041927` | 8 x H200 | shard MoE, `global_batch_size=8192`, `bp_steps=2`, `log_interval=1` | succeeded | 验证 metrics reduce 修复：forward 3.740s、backward 2.513s、optimizer 0.228s、zero-grad 0.002s、`reduce_metrics_done` 0.006s、`wandb_log_done` 0.002s，完整自然退出。 |
| 2026-06-04 19:55 HKT | `hrm-moe64x8-grouped-bp2-gb8192-06041955` | 8 x H200 | grouped MoE, ATen `grouped_mm`, `global_batch_size=8192`, `bp_steps=2` | stopped | 前向很快：`forward_done elapsed=0.771s`，但超过 2 分钟没有 `backward_done`，卡在 `loss.backward()`。判断为当前容器/PyTorch 的 `grouped_mm` backward/wgrad 路径不适合本训练链路；停止任务，改用 padded `torch.bmm` 训练路径。 |
| 2026-06-04 20:01 HKT | `hrm-moe64x8-grouped-bmm-bp2-gb8192-06042001` | 8 x H200 | grouped MoE, padded `torch.bmm`, `global_batch_size=8192`, `bp_steps=2` | succeeded | grouped-bmm 完整跑通并自然退出：forward 0.892s、backward 0.258s、optimizer 0.110s、zero-grad 0.001s、`reduce_metrics_done` 0.001s、`wandb_log_done` 0.002s。核心训练段 1.261s，参数量 5,413,797,888。 |
| 2026-06-04 20:04 HKT | `hrm-dense-xl-bp2-gb8192-06042004` | 8 x H200 | Dense XL 对照, `global_batch_size=8192`, `bp_steps=2` | succeeded | 为 grouped-bmm 结果补最新 dense 对照：forward 0.473s、backward 0.107s、optimizer 0.087s、zero-grad 0.001s。核心训练段 0.668s。 |

### 2026-06-04 19:34 HKT dense 对照与 MoE shard 性能对比

对照目标：回答“当前加速版 MoE 比 dense 慢多少”，并确认是否还有 infra
加速空间。所有任务都在同一 worktree、同一 8 x H200 节点、同一持久化训练数据、
同一 smoke 口径下运行：`global_batch_size=8192`、`epochs=1`、`max_steps=1`、
`log_interval=1`、`compile_train_batch=false`、`profile_train_batch=true`、
`lr_warmup_steps=1`、`+arch.bp_min_steps=2 arch.bp_max_steps=2`。

| 模型 | Job | 状态 | 参数量 | forward | backward | optimizer | zero-grad | reduce metrics | wandb log | 进度条单步耗时 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense XL | `hrm-dense-xl-bp2-gb8192-06041934` | succeeded | 1,182,793,728 | 0.497s | 0.102s | 0.084s | 0.001s | 0.050s | 0.001s | 4.97s |
| MoE XL 64x8 shard | `hrm-moe64x8-shard-reduce-bp2-gb8192-06041927` | succeeded | 5,413,797,888 | 3.740s | 2.513s | 0.228s | 0.002s | 0.006s | 0.002s | 10.08s |

同口径慢速比例：

| 口径 | Dense | MoE shard | MoE / Dense |
| --- | ---: | ---: | ---: |
| forward | 0.497s | 3.740s | 7.53x |
| backward | 0.102s | 2.513s | 24.64x |
| optimizer | 0.084s | 0.228s | 2.71x |
| 核心训练段，forward + backward + optimizer + zero-grad | 0.684s | 6.483s | 9.48x |
| 加上 metrics reduce 和 wandb log | 0.735s | 6.491s | 8.83x |
| 进度条单步 wall time | 4.97s | 10.08s | 2.03x |

补充口径：尝试用 `max_steps=3` 做短平均时，`epochs=1 max_steps=3`
只产生 1 个有效训练 batch，不适合作为 3-step 结果。改用
`epochs=3 max_steps=3 checkpoint_interval=999` 后，dense 和 MoE shard 都实际
产生 2 个有效训练 batch，第三个 epoch 没有有效 step；因此下面只作为补充
平均值，不替代上面的同配置 single-step headline。

| 模型 | Job | 有效 steps | forward 平均 | backward 平均 | optimizer 平均 | 核心训练段平均 | 加上 metrics/log 平均 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense XL | `hrm-dense-xl-bp2-gb8192-e3s3-06041941` | 2 | 0.614s | 0.076s | 0.063s | 0.754s | 0.756s |
| MoE XL 64x8 shard | `hrm-moe64x8-shard-bp2-gb8192-e3s3-06041943` | 2 | 2.546s | 2.015s | 0.194s | 4.758s | 4.766s |

补充平均慢速比例：

| 口径 | MoE / Dense |
| --- | ---: |
| forward | 4.15x |
| backward | 26.51x |
| optimizer | 3.08x |
| 核心训练段，forward + backward + optimizer + zero-grad | 6.31x |
| 加上 metrics reduce 和 wandb log | 6.30x |

结论：

- 当前 shard/FSDP 加速版 MoE 在核心训练段比 Dense XL 慢约 9.5x；如果看这个
  one-step smoke 的进度条 wall time，则慢约 2.0x。后者被 dataloader、启动后
  首步调度、进度条刷新等固定开销稀释，判断 kernel/infra 瓶颈时应优先看
  `TrainProfile` 的核心训练段。
- 有效 2-step 补充平均下，MoE shard 核心训练段比 Dense XL 慢约 6.3x。这个
  数值比 single-step 小，主要因为第二个有效 batch 明显更短；要得到正式稳态吞吐，
  后续应专门准备可重复多 step 的 smoke 数据切片或使用更长训练窗口。
- shard 版已经明显改善 FSDP/optimizer 开销。origin MoE 的一次有效 step 中
  optimizer 为 0.940s；shard + expert FSDP 后 optimizer 为 0.228s，约 4.1x
  加速。说明“把 64 个 expert 组织成 8 个 expert shard，再让 FSDP 包
  `layer.mlp.experts`”这条路是有效的。
- 大头仍然是 expert compute 本身：当前 `SparseMoEExpertShard.forward` 仍然在
  每个 MoE 层里按 expert 循环，执行 `torch.where`、小 batch `F.linear`、
  routing weight 乘法和 `index_add_`。shard 只是降低 FSDP 对象数量和优化器
  开销，还没有实现真正的 grouped GEMM。

当前 infra 加速空间：

- 第一优先级是 grouped expert compute。把现在每层 64 次 expert 查找/小 GEMM
  的路径，改成按 `selected_experts` 排序或分桶，一次性构造 grouped GEMM 的
  输入，再按原 token/top-k 顺序 scatter 回来。参考 pith-train 的 Qwen3 MoE
  grouped expert 思路，但落地到 HRM 时必须保持 router/top-k 语义完全不变。
- 第二优先级是减少 `torch.where` 和 `index_add_` 的次数。当前每个 expert 都
  扫一遍 `selected_experts`，这会在 64 experts、top-k=8 时放大 Python 调度和
  GPU 小 kernel 开销。更合理的是一次 sort/argsort 生成 expert 分段，再复用分段
  做 gate/up/down 和 scatter。
- 第三优先级是继续保留 expert collection 的 FSDP 包装，但让 collection 内部
  的参数布局服务 grouped GEMM。现有 `moe_expert_in_one_shard=8` 对 optimizer
  已经有收益；后续可以评估每 shard 8/16 experts 的吞吐和显存折中。
- Expert Parallel / all-to-all 是更大的架构改动，单机 8 卡场景下未必是首选；
  只有当 grouped GEMM 后仍被单卡 expert 参数/激活占用卡住，才值得引入 EP。
- 精度守门必须保留：router softmax 使用 fp32；top-k routing weights 归一化后
  再 cast 回 hidden dtype；origin/shard/grouped 任意新实现都要通过 forward、
  aux loss、expert counts、输入梯度、router 梯度和 expert 梯度等价测试，再跑
  8 卡 smoke。MoE 的精度风险主要来自路由顺序、scatter 聚合顺序和 dtype 提前
  cast，不能只用 loss 能下降来判断正确。

### 2026-06-04 20:01 HKT grouped expert compute 达标记录

严格目标：本轮把“接近 dense”量化为同口径核心训练段不超过 Dense XL 的 2x。
原因是 64x8 MoE 即使每 token 激活 FLOPs 约等于 dense FFN，也额外包含 router、
top-k、expert 排序、scatter/gather、padding，以及更大总参数带来的 FSDP/optimizer
开销；因此 1.0x dense 不是现实的第一阶段目标。

实现摘要：

- `models/layers.py` 新增 grouped expert compute。它先把
  `selected_experts` 按 token-major/top-k-major 展平，再按 expert 稳定排序；
  每个 expert 内顺序与 origin 的 `torch.where` 路径一致。
- 为避免当前 ATen `grouped_mm` backward 卡住，训练路径使用 padded
  `torch.bmm`：将 sorted token 填充到
  `[num_experts, max_tokens_per_expert, hidden]`，执行 batched gate/up GEMM 和
  batched down GEMM，然后按原 token index `index_add_` 回写。
- `models/transformer.py` 支持 `moe_implementation=grouped`；新增
  `config/arch/size/XL_moe64x8_grouped.yaml`。
- `scripts/test_moe_shard_equivalence.py` 已升级为 origin/shard/grouped 三路精度
  等价测试。

同口径 single-step 结果：

| 模型 | Job | 状态 | forward | backward | optimizer | zero-grad | 核心训练段 | 加 metrics/log | 单步 wall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense XL | `hrm-dense-xl-bp2-gb8192-06042004` | succeeded | 0.473s | 0.107s | 0.087s | 0.001s | 0.668s | 0.671s | 4.28s |
| MoE 64x8 shard | `hrm-moe64x8-shard-reduce-bp2-gb8192-06041927` | succeeded | 3.740s | 2.513s | 0.228s | 0.002s | 6.483s | 6.491s | 10.08s |
| MoE 64x8 grouped-bmm | `hrm-moe64x8-grouped-bmm-bp2-gb8192-06042001` | succeeded | 0.892s | 0.258s | 0.110s | 0.001s | 1.261s | 1.264s | 5.91s |

慢速比例：

| 对比 | 核心训练段比例 | 结论 |
| --- | ---: | --- |
| shard / dense | 9.71x | shard 只优化了 FSDP/optimizer，expert compute 仍然是逐 expert loop。 |
| grouped-bmm / dense | 1.89x | 达到“<= 2x dense”的严格第一阶段目标。 |
| grouped-bmm / shard | 0.19x | grouped-bmm 相比 shard 核心训练段约 5.1x 加速。 |

下一步加速空间：

- 当前 grouped-bmm 已接近 dense，但仍有约 0.59s 核心段差距。主要剩余开销来自
  router/top-k、sort、padding scatter/gather，以及 padded bmm 的 padding 计算。
- ATen `grouped_mm` forward 更快，但 backward/wgrad 在本容器链路下卡住；不能直接
  用作训练路径。后续如果要冲到 1.2x-1.5x dense，应优先接 XTuner 的 Triton/CUTLASS
  grouped GEMM，并单独验证 wgrad 精度和 FSDP 梯度归约。
- 任何更低层 kernel 替换都必须先通过 origin/shard/grouped forward 和梯度等价测试，
  再跑 8 卡 smoke；MoE 精度仍然是硬门槛。

经验：

- 给 `PretrainConfig` 新增顶层字段，或计划在 rjob 里用普通 `key=value`
  调已有默认字段时，要同步写入 Hydra YAML 默认配置；否则会报
  `Key ... is not in struct`。也可以用 `+key=value` 临时追加，但长期可复用
  参数应进入 YAML。
- `max_steps` 很小的 smoke 会触发 LR schedule 边界条件；`total_steps <=
  lr_warmup_steps` 时不能直接用 `total_steps - lr_warmup_steps` 做 cosine
  分母。
- 64x8 MoE 的 naive eager expert loop 可以跑到初始化阶段，但每卡 4096 token
  的首个 step 很慢。调通链路时先用每卡 1024 token 或更小 batch，再考虑
  grouped GEMM / expert parallel / 更细粒度 FSDP 优化。
- 对慢 step 不能只看进度条；需要在 `train_batch` 内部打点区分 forward、
  backward、optimizer 和 zero-grad，否则无法判断是 MoE dispatch、FSDP
  all-gather/reduce，还是优化器状态初始化导致。
- 对 HRM 这种有 bp warmup 的模型，`max_steps=1` 的 smoke 会让 warmup 进度直接
  到 1；如果不显式固定 `bp_steps`，第一步可能就是完整反传。调试链路时使用
  `+arch.bp_min_steps=2 arch.bp_max_steps=2`。
- `global_batch_size` 过小会产生“成功但没训练”的假阳性；本次
  `global_batch_size=1024` 退出成功但无 profile/metrics。后续 smoke 至少使用已知
  会产出 batch 的 `global_batch_size=8192`，并以 `TrainProfile` 或 metrics 为准。
- FSDP2 不能直接包 `ModuleList`。参考 XTuner 的 `ExpertShard` 和 pith-train 的
  `Qwen3MoeExperts`，expert 参数容器必须是有 `forward()` 的 module；HRM 里用
  `SparseMoEExpertCollection` 解决。
- MoE 精度对齐不能只看 loss 能跑。当前约束是：router softmax 保持 fp32；
  top-k 权重归一化后再 cast 回 hidden dtype；origin/shard 必须通过 forward、
  aux loss、expert counts 和梯度等价测试；FSDP 加速只改变参数组织和 shard 粒度，
  不改变路由语义。
- 参考实现：
  XTuner `mixtral/modeling_mixtral.py` 与 `deepseek_v2/modeling_deepseek.py`
  使用 origin/shard 两种 MoE 实现；pith-train `qwen3_moe.py` 使用 Qwen3 grouped
  expert、top-k router、load-balance loss injector，并在 FSDP 测试中单独包
  `layer.mlp.experts`。

### 2026-06-04 22:54 HKT 1.2x dense 冲刺记录

硬性原则：本轮所有 MoE 加速都先过精度/等价 gate，再看速度；不通过 forward、
aux loss、expert counts、输入梯度、router 梯度和 expert 权重梯度等价的改动，不进入
性能 smoke。允许优化 kernel/tiling/dispatch，但不允许通过降低 top-k、改变 router
fp32 softmax、改变聚合语义或放宽精度来换速度。

对照口径：

- Dense 稳态参考使用 `hrm-dense-xl-bp2-gb8192-e3s3-06041941` 的第 2 个有效
  step：forward 0.135s、backward 0.050s、optimizer 0.038s、zero-grad 0.001s，
  核心训练段约 0.224s。
- 严格目标为 `1.2x dense`，即核心训练段不超过约 0.269s。
- 当前最佳可用 MoE 为 `grouped_triton + autotune + SM_MARGIN=16 + aux bincount`，
  第 2 个有效 step 核心约 0.389s，约为 dense 稳态的 1.74x；尚未达到 1.2x。

精度 gate：

| 时间 | Job | 覆盖内容 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| 22:38 HKT | `hrm-moe-ep-triton-bincount-eq-06042238` | grouped_ep + Triton + bf16，aux loss 从 one-hot 改为 `bincount` | passed | aux loss 改动可用；先过 gate 后才允许测速度。 |
| 22:46 HKT | `hrm-moe-cutlass-eq-06042244` | CUDA/bf16 下 origin/shard/grouped-bmm/grouped-triton/grouped-cutlass/grouped-ep 全路径等价 | passed | CUTLASS kernel 精度可接受，可以进入性能 smoke。 |
| 22:52 HKT | `hrm-moe-sortfast-eq-06042250` | 将 dispatch 排序从 `stable=True` 改为默认 sort 的临时改动 | passed | 数值等价，但性能变差，已回退，不采用。 |

性能 smoke 结果，同一配置均为 `global_batch_size=8192`、`bp_steps=2`、
`max_steps=3`，只取第 2 个有效 step：

| 方案 | Job | forward | backward | optimizer | zero-grad | 核心训练段 | 对 dense 稳态 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton autotune, SM_MARGIN=16, aux `bincount` | `hrm-moe64x8-triton-auto-sm16-bincount-06042241` | 0.258s | 0.086s | 0.044s | 0.001s | 0.389s | 1.74x | 当前最佳；精度已过 gate。 |
| CUTLASS grouped_gemm | `hrm-moe64x8-cutlass-bp2-gb8192-06042246` | 0.638s | 0.496s | 0.044s | 0.001s | 1.179s | 5.26x | 精度过，但训练性能明显慢，不采用。 |
| 默认 sort 临时优化 | `hrm-moe64x8-triton-auto-sm16-sortfast-06042252` | 0.371s | 0.090s | 0.043s | 0.001s | 0.505s | 2.25x | 精度过但速度变差，已回退到 `stable=True`。 |

已有 Triton 参数 sweep：

| 方案 | Job | 第 2 step 核心训练段 | 结论 |
| --- | --- | ---: | --- |
| grouped-triton 默认 | `hrm-moe64x8-triton-06042043` | 约 0.456s | 比 grouped-bmm 快，但不是最佳。 |
| `BLOCK_M=64` | `hrm-moe64x8-triton-bm64-06042208` | 约 0.546s | 更慢，不采用。 |
| `autotune=1` | `hrm-moe64x8-triton-auto-06042212` | 约 0.414s | 明显改善。 |
| `autotune=1, SM_MARGIN=16` | `hrm-moe64x8-triton-auto-sm16-06042215` | 约 0.391s | 最佳区间之一。 |
| `autotune=1, SM_MARGIN=8` | `hrm-moe64x8-triton-auto-sm8-06042219` | 约 0.392s | 与 SM16 基本一致。 |
| `autotune=1, SM_MARGIN=32` | `hrm-moe64x8-triton-auto-sm32-06042217` | 约 0.645s | 明显变慢，不采用。 |

负结果与原因：

- `torch.compile` 强开 MoE 训练不采用。`hrm-moe64x8-triton-compile-force-06042223`
  首步 forward 26.389s、backward 12.209s、optimizer 51.567s，第二步 optimizer
  仍约 8.083s，图断裂和重编译严重污染训练速度。
- grouped_ep 自定义 all-to-all 的精度 gate 已通过，但
  `hrm-moe64x8-ep-triton-a2a-auto-sm16-06042232` 在第二步
  `forward_done elapsed=0.445s` 后没有完成 backward，说明“等价通过”不等于
  “多步训练可用”。EP 后续必须增加多步 backward gate。
- CUTLASS grouped_gemm 在当前镜像中可用且精度过，但 wgrad/backward 明显慢于
  Triton，不作为当前主线。
- 默认 sort 的临时改动精度过但 forward 变慢，已经回退。后续不要把
  `stable=True` 当作必然慢点直接删除，必须以同口径 smoke 为准。

当前判断：

- 在不牺牲精度的前提下，当前最佳是 Triton grouped GEMM + autotune +
  `SM_MARGIN=8/16`。它把 MoE 从 shard 版的多倍慢拉到约 1.7x dense 稳态，但还没到
  1.2x。
- 下一步要继续接近 1.2x，不能再靠宏观开关猜。需要对 MoE forward 内部做细粒度
  profile，拆开 router/top-k、sort、index_select、两次 grouped GEMM、activation、
  index_add，以及 Triton backward 的 grad-input/wgrad 时间；然后只对被证明占比高的
  环节做 kernel 或 dispatch 优化。

### 2026-06-05 00:00 HKT GitHub 调研与 MoE profiling

外部实现调研：

- MegaBlocks 的 README 把 Hopper 代 GPU 上的 grouped GEMM 作为当前推荐路径，并强调
  dMoE/block-sparse reformulation 可以避免 token dropping 仍保持硬件效率。
  这支持我们继续沿 grouped GEMM / block sparse 方向优化，而不是回到逐 expert loop。
  参考：https://github.com/databricks/megablocks
- Tutel 定位为 optimized MoE library，支持 top-k gate、expert 参数跳过普通 allreduce、
  单机 8 GPU 和多机分布式 MoE。它对当前 HRM 的启发主要是：通信/参数归约语义必须显式
  隔离，不能让 expert 参数走普通 dense FSDP/allreduce 语义。
  参考：https://github.com/microsoft/Tutel
- DeepEP 重点是 Expert Parallel 通信，文档强调 IB/RDMA、虚拟 lane、adaptive routing、
  以及通信-计算 overlap/SM-free RDMA path 等。它适合后续重新做 EP，但本轮 profile
  显示当前单机 grouped_triton 的主要问题还在本地 MoE forward，不是 EP 通信。
  参考：https://github.com/deepseek-ai/DeepEP
- TorchTitan 代表 PyTorch 原生 FSDP2/多维并行/observability 路线，适合借鉴结构化
  profiling 和 FSDP2 参数组织；但本仓库当前 MoE `torch.compile` 仍然不稳定，不能照搬
  compile 默认开启。
  参考：https://github.com/pytorch/torchtitan

新增 instrumentation：

- 新增 `models/moe_profile.py`，只有 `HRM_MOE_PROFILE=1` 时才插 CUDA event 并同步；
  默认训练不受影响。
- `pretrain.py` 在 `forward_done` 和 `backward_done` 后分别打印
  `[MoEProfile] forward/backward ...`，便于和 `[TrainProfile]` 对齐。
- `models/layers.py` 拆分 router、dispatch、gate_up_gemm、activation、
  down_gemm、combine、aux_metrics；EP 路径拆分 dispatch/all_to_all/local_experts/combine。
- Triton/CUTLASS grouped GEMM 的自定义 autograd backward 拆分
  `grad_input_gemm` 和 `grad_weight_gemm`。

验证：

| 时间 | Job/命令 | 结果 | 结论 |
| --- | --- | --- | --- |
| 23:49 HKT | `python scripts/test_moe_shard_equivalence.py` | passed | profiling 默认关闭时不改变本地等价。 |
| 23:51 HKT | `hrm-moe-profile-gate-06042349` | passed | profiling/导入重构后 CUDA/bf16 等价仍通过。 |
| 23:57 HKT | `hrm-moe-tokenidx-eq-06042356` | passed | token index 临时优化数值等价，但需性能 smoke 决定是否采用。 |

MoE profile 结果，当前最佳 Triton 配置，`HRM_MOE_PROFILE=1`，第 2 个有效 step：

| Job | forward | MoE forward breakdown | backward | MoE backward breakdown | 结论 |
| --- | ---: | --- | ---: | --- | --- |
| `hrm-moe64x8-triton-sm16-moeprof-06042345` | 0.407s | total 0.156s；router 0.012s、dispatch 0.032s、gate_up 0.046s、activation 0.006s、down 0.034s、combine 0.019s、aux 0.006s | 0.085s | 未拆 backward | forward 不是单个大 kernel，而是 dispatch+combine+两次 GEMM 的组合开销。 |
| `hrm-moe64x8-triton-sm16-bwprof-06042351` | 0.348s | total 0.155s；router 0.012s、dispatch 0.032s、gate_up 0.045s、activation 0.006s、down 0.034s、combine 0.019s、aux 0.006s | 0.119s | total 0.037s；grad_input 0.020s、wgrad 0.017s | backward grouped GEMM 不是最大问题；继续优化应优先看 forward dispatch/combine 和 forward grouped GEMM。 |

负优化：

| 时间 | Job | 改动 | 结果 | 处理 |
| --- | --- | --- | --- | --- |
| 23:57 HKT | `hrm-moe64x8-triton-sm16-tokenidx-06042357` | 用 `sort_idx // top_k` 替代 `arange(...).repeat_interleave(top_k)` 再 `index_select` | 第 2 step forward 0.277s、backward 0.098s、optimizer 0.048s、核心约 0.424s，慢于当前最佳 0.389s | 已回退；这类微优化必须以同口径 smoke 为准。 |

后续优先级：

- 保留 `HRM_MOE_PROFILE=1` 作为诊断开关，默认关闭。
- 当前最值得继续尝试的是减少 forward dispatch/combine 开销，或参考 MegaBlocks 的
  block-sparse/dropless 思路，把 expert-sorted dispatch、GEMM 和 combine 做更强融合。
- EP/DeepEP 路线仍有价值，但要先修复当前 grouped_ep 第二步 backward stall，并增加
  多步训练 gate；不能只凭 all-to-all 等价脚本通过就采用。

### 2026-06-05 00:35 HKT MoE 精度失败回滚与 GitHub 继续调研

本轮目标仍是严格冲到 `1.2x dense`，但精度 gate 高于速度。上一次临时尝试把
routing weight 从 down projection 之后前移到 down projection 之前，意图是减少
combine 阶段的乘法/访存，并与部分 fused MoE 实现的思路相近。

结果：

| 项目 | 内容 |
| --- | --- |
| 本地 CPU/fp32 等价 | passed |
| CUDA/bfloat16 gate | failed |
| Job | `hrm-moe-preweight-eq-06050027` |
| 失败点 | `origin.gate.weight.grad` vs `grouped_triton.gate.weight.grad` |
| 最大绝对误差 | 0.1640625，超过 `atol=5e-2` |
| 最大相对误差 | 117.5，超过 `rtol=3e-2` |
| 处理 | 已回滚，不进入性能 smoke，不采用 |

结论：这个改动虽然在实数代数上等价，但在 CUDA/bfloat16 训练里改变了舍入位置，
会影响 router 梯度。MoE 的精度对 bf16 舍入、加权位置和聚合顺序非常敏感，
后续不能用“代数等价”替代 CUDA/bfloat16 gate。

回滚后验证：

| 检查 | 结果 |
| --- | --- |
| `PYTHONPYCACHEPREFIX=/tmp/hrm_pycache_check python -m py_compile ...` | passed |
| `python scripts/test_moe_shard_equivalence.py` | passed |

GitHub/外部实现继续调研：

- PyTorch/Triton 的 persistent cache-aware grouped GEMM 方向与当前 HRM Triton
  backend 一致：使用每 SM 一个 persistent program、grouped launch ordering
  提升 L2 reuse、Hopper TMA 载入 expert weight；官方报告 grouped launch 带来
  L2 hit 提升和 kernel 加速。这说明下一步优先应继续做 Triton scheduler/tiling
  和 TMA/descriptor 方向，而不是改变路由语义。参考：
  https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- `bassrehab/triton-kernels` 的 MoE dispatch 文档把可落地方向拆得很清楚：
  router matmul 保持 cuBLAS，softmax/top-k、permute/unpermute 和 expert FFN 用
  Triton；它强调 fused gate+up projection 能消掉 gate/up 中间 buffer，且
  FP32 accumulation 对 router、SiLU 和 weighted combine 很关键。这个方向可能继续
  压缩 forward，但训练版必须配套 backward 等价，不能只做 forward kernel。
  参考：https://github.com/bassrehab/triton-kernels/blob/main/docs/moe_dispatch.md
- MegaBlocks 仍是 block-sparse / dropless MoE 的重要参考；其 README 明确
  `megablocks[gg]` grouped GEMM 是 Hopper 推荐路径。对 HRM 的启发是保留
  dropless/top-k 语义，尽量把 expert-sorted dispatch 与 GEMM 调度结合得更紧。
  参考：https://github.com/databricks/megablocks
- DeepGEMM 已包含 BF16 GEMM、fused MoE / Mega MoE、JIT 编译和 MoE backward
  weight-gradient kernel，是后续可评估的新 backend 候选。但它偏向更完整的 CUDA
  kernel 栈，接入前要先确认当前 rjob 镜像的 CUDA/PyTorch/NVCC 版本是否满足，并
  必须单独跑 origin/grouped_triton/DeepGEMM 三路 CUDA/bfloat16 gate。
  参考：https://github.com/deepseek-ai/DeepGEMM
- DeepEP/Tutel 主要价值仍在 EP 通信/dispatch-combine/overlap。当前单机 grouped
  Triton profile 显示 forward 本地 dispatch+GEMM 仍是主要空间；EP 之前已经出现
  第二步 backward stall，所以 EP 路线需要先补多步训练 gate。
  参考：https://github.com/deepseek-ai/DeepEP 和 https://github.com/microsoft/Tutel

下一轮只排一个 8 卡实验的建议：

1. 先不改精度语义，继续使用当前 `grouped_triton + autotune + SM_MARGIN=16`。
2. 只做 Triton scheduler/tiling sweep 或可回退 backend，候选包括 `BLOCK_M=256`、
   扩展 `BLOCK_N/BLOCK_K/GROUP_M` autotune 集、以及 DeepGEMM 环境探测。
3. 每个候选必须按固定顺序走：本地等价 -> CUDA/bfloat16 gate -> 同口径
   `max_steps=3` smoke -> 中文记录。
4. 当前集群上 `hrm-pre32g-xl-e4-off0604` 仍在 RUNNING；为避免资源互相影响，本轮
   暂不新增 8 卡 MoE 性能任务，等资源窗口合适时只提交一个候选。

### 2026-06-05 00:41 HKT 32 卡 MoE 长训启动

按用户要求先启动 32 卡 MoE 训练，再继续做效率优化。本次使用当前已经通过精度 gate
且速度最快的安全路径：`grouped_triton + autotune + SM_MARGIN=16`。没有引入
新的未验证优化。

| 项目 | 值 |
| --- | --- |
| Job | `hrm-moe32g-gt06050041` |
| 状态 | 2026-06-05 00:41 HKT 已提交，4 个 replica 均为 `STARTING` |
| 资源 | 32 张 H200，4 replicas x 8 GPUs |
| Worktree | `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8` |
| Branch | `codex/hrm-moe64x8` |
| Commit | `8b4248a` |
| Config | `cfg_pretrain`, `arch_size=XL_moe64x8_grouped_triton` |
| 数据 | `/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097` |
| Checkpoint | `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8/checkpoints/hrm-moe32g-gt06050041` |
| Epochs | 4 |
| Global batch | 196,608 tokens |
| MoE | 64 experts, top-k=8, expert intermediate=512 |
| Triton env | `HRM_MOE_TRITON_AUTOTUNE=1`, `HRM_MOE_TRITON_SM_MARGIN=16` |
| 其他参数 | `checkpoint_interval=1`, `log_interval=5`, `compile_train_batch=false`, `fsdp_wrap_moe_experts=true` |
| W&B | `WANDB_MODE=offline`，避免 32 卡任务因 API key 缺失在 `wandb.init` 失败 |
| auto_restart | `false` |

后续动作：

1. 确认 4 个 replica 进入 `RUNNING`，并检查日志是否进入 `World Size 32`。
2. 如果训练启动失败，先定位首因并记录；不要把后续 rank 的 NCCL/TCPStore 退出当首因。
3. 训练挂起后继续做 MoE 效率优化，但新优化仍必须先过本地等价和 CUDA/bfloat16 gate。

00:46 HKT 更新：

- `rjob list` 显示 4 个 replica 全部 `RUNNING`。
- rank0 日志确认 `torchrun --nnodes=4 --nproc_per_node=8`，并进入
  `[Rank *, World Size 32]: Epoch 1`。
- W&B 使用 offline：`wandb/offline-run-20260604_164256-ni3cinqr`。
- 进度条已从 `0/304789` 推进到约 `190/304789`，约 `1.2-1.4 it/s`，说明不是
  只完成容器启动，而是已经实际执行训练 step。
- 当前没有看到 `Traceback`、`RuntimeError`、OOM、NCCL failure 或 W&B credential
  失败。

同一时间启动剩余 8 卡效率 gate：

| 项目 | 值 |
| --- | --- |
| Job | `hrm-moe-bm256-eq-06050045` |
| 目的 | 测试 Triton grouped GEMM `BLOCK_M=256` 是否保持 CUDA/bfloat16 梯度等价 |
| 资源 | 8 张 H200，单节点 |
| 状态 | 2026-06-05 00:46 HKT 为 `RUNNING` |
| 精度约束 | origin/shard/grouped/grouped_triton/grouped_cutlass/grouped_ep 前向、aux、expert counts、输入梯度、router 梯度、expert 梯度等价 |
| 处理规则 | 只有通过 gate 后才允许排性能 smoke；失败则不测速、不采用 |

00:48 HKT 更新：

- `hrm-moe-bm256-eq-06050045` 失败，不进入性能 smoke。
- 失败首因不是数值 mismatch，而是 Triton 编译资源不足：
  `OutOfResources: shared memory, Required: 327704, Hardware limit: 232448`。
- 结论：当前 `num_stages=3` 的 grouped GEMM kernel 不能直接把 `BLOCK_M` 拉到 256。
  如果继续探索大 `BLOCK_M`，需要降低 `num_stages` 或增加可跳过 OOR config 的
  autotune 逻辑，并重新从 CUDA/bfloat16 gate 开始。
- 为避免影响正在运行的 `hrm-moe32g-gt06050041`，后续效率代码改动应在单独
  worktree 中完成，再用独立 rjob 验证。

### 2026-06-05 01:03 HKT 剩余 8 卡 MoE Triton 调参

为了不影响正在运行的 32 卡 MoE 长训，效率代码实验转到独立 worktree：
`/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8-tune`，分支
`codex/hrm-moe64x8-tune`。

代码改动：

- 新增 `HRM_MOE_TRITON_NUM_STAGES`，默认仍为 3；只有显式设置时才改变
  Triton grouped GEMM config。
- `scripts/rjob_hrm_common.sh` 透传 `moe_triton_num_stages`，方便 gate 和 smoke
  使用同一参数。
- 本地验证通过：`py_compile`、`bash -n`、`python scripts/test_moe_shard_equivalence.py`。
- tune 分支 commit：`4ee2b33 Add Triton num stages tuning knob`。

调参结果：

| 时间 | Job | 参数 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| 00:50 HKT | `hrm-moe-bm256s2-eq-06050051` | `BLOCK_M=256`, `num_stages=2`, autotune, SM16 | failed | 仍然 OOR：shared memory required 262160 > hardware limit 232448，不测速。 |
| 00:54 HKT | `hrm-moe-bm256s1-eq-06050054` | `BLOCK_M=256`, `num_stages=1`, autotune, SM16 | passed | CUDA/bfloat16 等价通过，可以测速。 |
| 00:56 HKT | `hrm-moe64x8-bm256s1-06050056` | 同上，8 卡 `max_steps=3` smoke | succeeded | 第 2 个有效 step：forward 0.294s、backward 0.129s、optimizer 0.045s、zero-grad 0.001s，核心约 0.469s。慢于当前最佳 0.389s，不采用。 |
| 01:03 HKT | `hrm-moe-bm128s2-eq-06050103` | 默认 `BLOCK_M=128`, `num_stages=2`, autotune, SM16 | running | 正在跑 CUDA/bfloat16 gate；通过后才允许性能 smoke。 |

当前判断：

- 32 卡 MoE 长训 `hrm-moe32g-gt06050041` 仍然 `RUNNING`，rank0 进度持续推进，
  01:02 HKT 已推进到约 `1625/304789`。
- `BLOCK_M=256` 不是当前的好方向：`num_stages=3/2` OOR，`num_stages=1` 虽然
  精度过但稳态核心段慢于当前最佳。
- 下一步优先看默认 `BLOCK_M=128` 的 stage/config 调整，以及更细的
  `BLOCK_N/BLOCK_K/GROUP_M` autotune；仍然保持“先 gate，后 smoke”的顺序。

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

## MoE speed checkpoint 清理：`hrm-moe-clean0605`

2026-06-05 11:42 HKT，确认 64x8 MoE speed test 已测完后，清理
`/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8/checkpoints` 下旧的
bring-up / speed / dense 对照 checkpoint。登录节点直接 `rm -rf` 因 rjob
容器 root-owned 文件报 `Permission denied`，改用短 CPU rjob
`hrm-moe-clean0605` 在容器内删除。

已删除的大目录包括：
`hrm-moe64x8-triton-06042043`,
`hrm-moe64x8-cutlass-06042057`,
`hrm-moe64x8-bmm-e4s3-06042046`,
`hrm-moe64x8-basic-bp2-06041908`,
`hrm-moe64x8-shard-bp2-gb8192-e3s3-06041943`,
`hrm-moe64x8-triton-06042038`,
`hrm-moe64x8-triton-auto-*`,
`hrm-moe64x8-triton-compile*`,
`hrm-moe64x8-triton-bm64-06042208`,
`hrm-moe64x8-triton-fsdp2-06042159`,
`hrm-moe64x8-cutlass-bp2-gb8192-06042246`,
`hrm-moe64x8-triton-sm16-*`,
`hrm-dense-xl-bp2-gb8192-s3-06041939`,
`hrm-dense-xl-bp2-gb8192-e3s3-06041941`。

保留当前长跑 checkpoint
`hrm-moe32g-sm16-06050339`；清理后 MoE worktree 的 `checkpoints/` 从约
`2.4T` 降到约 `2.3M`，整个 `quxiaoye` 挂载从约 `97%` 使用率降到约
`49%`，可用空间约 `7.8T`。

## 初版 32 卡 MoE 代码快照

2026-06-06 15:56 HKT，用户要求在后续切到加速版 MoE 前备份当前正在运行的
初版 MoE 代码，方便后续比较效果是否对得上。

| 项目 | 值 |
| --- | --- |
| 训练 Job | `hrm-moe32g-sm16-06050339` |
| 当前训练进度 | `137490/304789`，总进度约 45.1%，epoch 2 内约 80.0% |
| 当前速度 | 约 `1.05-1.08 s/it` |
| 源 worktree | `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8` |
| 源分支 | `codex/hrm-moe64x8` |
| 备份 commit | `90b81a430a847fdb7239e04c5ce788d08da648fb` |
| 备份 tag | `backup/moe-initial-hrm-moe32g-sm16-20260606-1556` |
| 备份 branch | `backup/moe-initial-hrm-moe32g-sm16-20260606-1556` |
| Archive | `code_backups/hrm-moe32g-sm16-initial-90b81a4-20260606-1556.tar.gz` |
| Archive sha256 | `9ad02c968e592bd9f7fbfa6414032a8edf2a0bebaa985e066d268459f02fabd3` |

该快照固定的是正在长跑的初版 `grouped_triton` MoE 代码，不包含 checkpoint、
数据、W&B 和 rjob 日志。后续如果启动最快的 MoE 加速版，应把质量结果优先和
这个快照对应的 `hrm-moe32g-sm16-06050339` 各 epoch 评测结果对齐比较。

<!-- HRM_EVAL_MONITOR:hrm-moe32g-sm16-06050339:start -->
## 32 卡在线评测监控：`hrm-moe32g-sm16-06050339`

最后刷新：2026-06-06 20:22:30 HKT。

| 项目 | 值 |
| --- | --- |
| Job | `hrm-moe32g-sm16-06050339` |
| 状态 | Running |
| Checkpoint | `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8/checkpoints/hrm-moe32g-sm16-06050339` |
| 评测数据 | `/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet` |
| 提交策略 | 每个 epoch 完成并稳定后提交 Standard / MMLU-Pro / AIME25，各 8 GPUs fanout |

评测任务索引：

| Eval set | Epoch | Job | 状态 | Summary |
| --- | --- | --- | --- | --- |
| Standard | 1 | `hrmmoe32-0605-e1-std` | succeeded | `rjob_logs/hrmmoe32-0605-e1-std_bench/summary.json` |
| MMLU-Pro | 1 | `hrmmoe32-0605-e1-mmlu` | succeeded | `rjob_logs/hrmmoe32-0605-e1-mmlu_bench/summary.json` |
| AIME25 | 1 | `hrmmoe32-0605-e1-aime` | succeeded | `rjob_logs/hrmmoe32-0605-e1-aime_bench/summary.json` |
| Standard | 2 | `hrmmoe32-0605-e2-std` | submitted | - |
| MMLU-Pro | 2 | `hrmmoe32-0605-e2-mmlu` | submitted | - |
| AIME25 | 2 | `hrmmoe32-0605-e2-aime` | submitted | - |
| Standard | 3 | `hrmmoe32-0605-e3-std` | waiting_checkpoint | - |
| MMLU-Pro | 3 | `hrmmoe32-0605-e3-mmlu` | waiting_checkpoint | - |
| AIME25 | 3 | `hrmmoe32-0605-e3-aime` | waiting_checkpoint | - |
| Standard | 4 | `hrmmoe32-0605-e4-std` | waiting_checkpoint | - |
| MMLU-Pro | 4 | `hrmmoe32-0605-e4-mmlu` | waiting_checkpoint | - |
| AIME25 | 4 | `hrmmoe32-0605-e4-aime` | waiting_checkpoint | - |

主指标（百分比）：

| Benchmark | Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | --- | --- | --- | --- | --- |
| GSM8k | acc | 71.27 | - | - | - |
| MATH | acc | 45.26 | - | - | - |
| DROP | em | 67.85 | - | - | - |
| DROP | f1 | 71.62 | - | - | - |
| MMLU | acc | 49.64 | - | - | - |
| ARC | acc | 63.63 | - | - | - |
| HellaSwag | acc | 38.20 | - | - | - |
| Winogrande | acc | 61.56 | - | - | - |
| BoolQ | acc | 80.92 | - | - | - |
| MMLU-Pro | acc | 20.79 | - | - | - |

Invalid Rate（百分比）：

| Benchmark | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | --- | --- | --- | --- |
| GSM8k | 1.36 | - | - | - |
| MATH | 7.06 | - | - | - |
| MMLU | 0.25 | - | - | - |
| ARC | 0.26 | - | - | - |
| HellaSwag | 0.00 | - | - | - |
| Winogrande | 0.00 | - | - | - |
| BoolQ | 0.00 | - | - | - |
| MMLU-Pro | 2.61 | - | - | - |

AIME25 Majority Voting（百分比）：

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | --- | --- | --- | --- |
| maj_pass@1 | 0.00 | - | - | - |
| maj_pass@10 | 23.33 | - | - | - |
| maj_pass@100 | 43.33 | - | - | - |
| pass@1 | 0.86 | - | - | - |
| pass@10 | 7.68 | - | - | - |
| pass@100 | 34.90 | - | - | - |

最近运行记录：
- 2026-06-05 22:13:57 HKT：Eval job hrmmoe32-0605-e1-mmlu is Running.
- 2026-06-05 23:24:49 HKT：Eval job hrmmoe32-0605-e1-aime is Running.
- 2026-06-05 23:24:50 HKT：Eval job hrmmoe32-0605-e1-mmlu is Succeeded.
- 2026-06-05 23:24:51 HKT：Loaded summary for hrmmoe32-0605-e1-mmlu: rjob_logs/hrmmoe32-0605-e1-mmlu_bench/summary.json.
- 2026-06-05 23:29:58 HKT：Eval job hrmmoe32-0605-e1-std is Succeeded.
- 2026-06-05 23:29:58 HKT：Loaded summary for hrmmoe32-0605-e1-std: rjob_logs/hrmmoe32-0605-e1-std_bench/summary.json.
- 2026-06-06 01:51:45 HKT：Eval job hrmmoe32-0605-e1-aime is Succeeded.
- 2026-06-06 01:51:47 HKT：Loaded summary for hrmmoe32-0605-e1-aime: rjob_logs/hrmmoe32-0605-e1-aime_bench/summary.json.
- 2026-06-06 20:22:27 HKT：Epoch 2 checkpoint is stable; submitting all eval sets.
- 2026-06-06 20:22:28 HKT：Submitted hrmmoe32-0605-e2-std for epoch 2 Standard.
- 2026-06-06 20:22:29 HKT：Submitted hrmmoe32-0605-e2-mmlu for epoch 2 MMLU-Pro.
- 2026-06-06 20:22:30 HKT：Submitted hrmmoe32-0605-e2-aime for epoch 2 AIME25.
<!-- HRM_EVAL_MONITOR:hrm-moe32g-sm16-06050339:end -->
