# HRM Pretrain Evaluation Results

Last updated: 2026-06-04 15:01 HKT.

## Experiment

Checkpoint root:
`/mnt/shared-storage-user/quxiaoye/HRM-Text/checkpoints/hrm-pretrain-16g-xl-persistent-0602115537`

Training data:
`/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097`

Training run:

| Item | Value |
| --- | --- |
| Job | `hrm-pretrain-16g-xl-persistent-0602115537` |
| Status | succeeded |
| GPUs | 16 H200 GPUs, 2 replicas x 8 GPUs |
| Config | `cfg_pretrain`, XL |
| Training start | 2026-06-02 11:56:44 HKT |
| Training end | 2026-06-04 08:32:30 HKT |
| Wall time | 44h 35m 46s |
| Stable epoch time | about 11h 40m per epoch |

Checkpoint completion times:

| Epoch | Checkpoint time |
| ---: | --- |
| 1 | 2026-06-02 21:32:43 HKT |
| 2 | 2026-06-03 09:13:47 HKT |
| 3 | 2026-06-03 20:53:33 HKT |
| 4 | 2026-06-04 08:32:29 HKT |

## Evaluation Setup

| Item | Value |
| --- | --- |
| Resources | 8 GPUs per eval job, fanout sharding |
| Fanout workers | 8 |
| Batch size | 16 for standard and MMLU-Pro evals |
| Data cache | `eval_data_hf_parquet` |
| Standard config | `evaluation/config/hrm_benchmarking.yaml` |
| MMLU-Pro config | `evaluation/config/hrm_mmlu_pro_benchmarking.yaml` |
| AIME config | `evaluation/config/hrm_maj_vote_benchmarking.yaml` |

Standard config benchmarks:
GSM8k, MATH, DROP, MMLU, ARC, HellaSwag, Winogrande, BoolQ.

## Run Index

| Eval set | Epoch | Job | Status | Summary |
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
| AIME25 | 3 | `hrm-eval-e3-aime25-seq-0604` | running | pending |
| AIME25 | 4 | `hrm-eval-e4-aime25-seq-0604` | running | pending |

## Main Metrics

Values are shown as percentages. DROP reports exact match and F1.

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

## Invalid Rates

Values are shown as percentages.

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

DROP does not report an invalid rate in the current summary.

## AIME25 Majority Voting

Values are shown as percentages. Epoch 3 and 4 jobs are running as of the last
update and should be filled when their summaries appear.

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | ---: | ---: | ---: | ---: |
| maj_pass@1 | 3.33 | 16.67 | pending | pending |
| maj_pass@10 | 16.67 | 36.67 | pending | pending |
| maj_pass@100 | 46.67 | 56.67 | pending | pending |
| pass@1 | 0.90 | 3.98 | pending | pending |
| pass@10 | 7.13 | 18.67 | pending | pending |
| pass@100 | 29.17 | 44.16 | pending | pending |

## Standard Benchmark Sample Counts

| Benchmark | n |
| --- | ---: |
| GSM8k | 1,319 |
| MATH | 5,000 |
| DROP | 9,536 |
| MMLU | 57 aggregated subjects |
| ARC | 1,172 |
| HellaSwag | 10,042 |
| Winogrande | 1,267 |
| BoolQ | 3,270 |
| MMLU-Pro | 12,032 |
| AIME25 | 30 |

For MMLU, the top-level summary `n` is the aggregated subject count; per-subject
sample counts are available in the raw benchmark logs.

## Notes

- The standard benchmark suite improves monotonically from epoch 1 to epoch 4 on
  all primary metrics listed above.
- MMLU-Pro improves from 19.57 percent at epoch 1 to 33.11 percent at epoch 4.
- AIME25 improved substantially from epoch 1 to epoch 2; epoch 3 and 4 are still
  running at the time of this update.
- The training data is the sampled HRM/Data IO instruction-response PrefixLM
  dataset, not a one-off dataset built from the evaluation test sets.
