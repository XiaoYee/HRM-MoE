# HRM Evaluation Results

Last updated: 2026-06-03 HKT.

Checkpoint root:
`/mnt/shared-storage-user/quxiaoye/HRM-Text/checkpoints/hrm-pretrain-16g-xl-persistent-0602115537`

Evaluation setup:

- Data: local parquet cache under `eval_data_hf_parquet`
- Resources: 8 GPUs, 8 fanout workers
- Batch size: 16
- Config: `evaluation/config/hrm_benchmarking.yaml`

## Runs

| Epoch | Eval job | Status | Summary |
| --- | --- | --- | --- |
| 1 | `hrm-eval-e1-localdata-r4-0603` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260603_025332/summary.json` |
| 2 | `hrm-eval-e2-localdata-0603` | succeeded | `rjob_logs/hrm-eval-fanout_0_0_bench_20260603_032940/summary.json` |

## Metrics

| Benchmark | Metric | Epoch 1 | Epoch 2 |
| --- | --- | ---: | ---: |
| GSM8k | acc | 0.648218 | 0.795299 |
| GSM8k | invalid | 0.020470 | 0.012889 |
| MATH | acc | 0.399000 | 0.509800 |
| MATH | invalid | 0.092600 | 0.070400 |
| DROP | em | 0.615562 | 0.740038 |
| DROP | f1 | 0.651795 | 0.776124 |
| MMLU | acc | 0.432901 | 0.543184 |
| MMLU | invalid | 0.001759 | 0.002469 |
| ARC | acc | 0.500000 | 0.725256 |
| ARC | invalid | 0.000000 | 0.000000 |
| HellaSwag | acc | 0.345250 | 0.477893 |
| HellaSwag | invalid | 0.000000 | 0.000000 |
| Winogrande | acc | 0.573796 | 0.650355 |
| Winogrande | invalid | 0.000000 | 0.000000 |
| BoolQ | acc | 0.742813 | 0.841896 |
| BoolQ | invalid | 0.000000 | 0.000000 |

## Sample Counts

| Benchmark | n |
| --- | ---: |
| GSM8k | 1319 |
| MATH | 5000 |
| DROP | 9536 |
| MMLU | 57 |
| ARC | 1172 |
| HellaSwag | 10042 |
| Winogrande | 1267 |
| BoolQ | 3270 |

Note: for MMLU, the top-level summary `n` is the aggregated subject count in
the benchmark summary; per-subject sample counts are available in the raw
metrics files.
