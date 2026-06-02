from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from evaluation.main import BenchmarkConfig, EvaluationConfig
from utils.functions import load_model_class


def _load_config(argv: list[str]) -> tuple[EvaluationConfig, dict[str, Any], str]:
    cli_conf = OmegaConf.from_dotlist(argv)
    fanout_conf = cli_conf.pop("fanout", {})
    config_path = cli_conf.pop("config", "evaluation/config/hrm_benchmarking.yaml")
    base_conf = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base_conf, cli_conf)
    cfg = EvaluationConfig(**OmegaConf.to_container(merged, resolve=True))  # type: ignore[arg-type]
    if OmegaConf.is_config(fanout_conf):
        fanout_dict = OmegaConf.to_container(fanout_conf, resolve=True)
    else:
        fanout_dict = dict(fanout_conf)
    return cfg, fanout_dict, str(config_path)  # type: ignore[arg-type]


def _selected_benchmarks(cfg: EvaluationConfig) -> list[BenchmarkConfig]:
    if cfg.run_only is None:
        return cfg.benchmarks
    requested = set(cfg.run_only)
    return [b_cfg for b_cfg in cfg.benchmarks if b_cfg.name in requested]


def _build_benchmark(b_cfg: BenchmarkConfig):
    bench_cls = load_model_class(f"benchmarks@{b_cfg.name}", prefix="evaluation.")
    return bench_cls(**(b_cfg.__pydantic_extra__ or {}))


def _generation_config(cfg: EvaluationConfig, b_cfg: BenchmarkConfig, benchmark) -> dict[str, Any]:
    return dict(cfg.generation_config | benchmark.generation_overrides | b_cfg.generation_config)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def _tail(path: Path, lines: int = 80) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "\n".join(content[-lines:])


def _run_worker(argv: list[str], cfg: EvaluationConfig, fanout_conf: dict[str, Any]) -> None:
    benchmark_name = str(fanout_conf["benchmark"])
    shard_index = int(fanout_conf["shard_index"])
    num_shards = int(fanout_conf["num_shards"])
    output_dir = Path(str(fanout_conf["output_dir"]))

    b_cfg = next((item for item in cfg.benchmarks if item.name == benchmark_name), None)
    if b_cfg is None:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    benchmark = _build_benchmark(b_cfg)
    gen_cfg = _generation_config(cfg, b_cfg, benchmark)
    prompt_template = gen_cfg.pop("prompt_template", "{prompt}")
    prompts = [prompt_template.format(prompt=s) for s in benchmark.prompts]
    indices = list(range(shard_index, len(prompts), num_shards))
    shard_prompts = [prompts[i] for i in indices]

    engine_cls = load_model_class(f"engines@{cfg.engine}", prefix="evaluation.")
    engine = engine_cls(**(cfg.__pydantic_extra__ or {}))

    print(
        f"Worker benchmark={benchmark_name} shard={shard_index}/{num_shards} "
        f"prompts={len(shard_prompts)}/{len(prompts)}"
    )
    generations = engine.generate(shard_prompts, **gen_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(benchmark_name)}.shard_{shard_index:02d}.json"
    with output_path.open("w") as f:
        json.dump(
            {
                "benchmark": benchmark_name,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "total_prompts": len(prompts),
                "indices": indices,
                "generations": generations,
            },
            f,
        )
    print(f"Wrote shard output: {output_path}")


def _aggregate_benchmark(output_dir: Path, b_cfg: BenchmarkConfig, num_shards: int) -> dict[str, Any]:
    benchmark = _build_benchmark(b_cfg)
    generations: list[str | None] = [None] * len(benchmark.prompts)

    for shard_index in range(num_shards):
        output_path = output_dir / f"{_safe_name(b_cfg.name)}.shard_{shard_index:02d}.json"
        with output_path.open("r") as f:
            payload = json.load(f)

        if payload["benchmark"] != b_cfg.name:
            raise ValueError(f"Shard {output_path} belongs to {payload['benchmark']}, not {b_cfg.name}")
        if payload["total_prompts"] != len(generations):
            raise ValueError(f"Shard {output_path} has inconsistent prompt count")

        for index, generation in zip(payload["indices"], payload["generations"]):
            if generations[index] is not None:
                raise ValueError(f"Duplicate generation for {b_cfg.name} prompt {index}")
            generations[index] = generation

    missing = [idx for idx, value in enumerate(generations) if value is None]
    if missing:
        raise ValueError(f"Missing {len(missing)} generations for {b_cfg.name}; first missing index {missing[0]}")

    metrics = benchmark.compute_metrics([str(value) for value in generations])
    result_path = output_dir / f"{_safe_name(b_cfg.name)}.metrics.json"
    with result_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def _run_parent(argv: list[str], cfg: EvaluationConfig, fanout_conf: dict[str, Any]) -> None:
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA devices available for sharded evaluation")

    requested_workers = int(fanout_conf.get("num_workers") or gpu_count)
    num_workers = max(1, min(requested_workers, gpu_count))
    output_dir = Path(str(fanout_conf.get("output_dir") or "rjob_logs/hrm_eval_shards"))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_argv = [item for item in argv if not item.startswith("fanout.")]
    all_results: dict[str, dict[str, Any]] = {}

    print(f"Sharded evaluation output dir: {output_dir}")
    print(f"CUDA device count: {gpu_count}; workers per benchmark: {num_workers}")

    for b_cfg in _selected_benchmarks(cfg):
        benchmark = _build_benchmark(b_cfg)
        prompt_count = len(benchmark.prompts)
        shard_count = min(num_workers, prompt_count)
        print("\n" + "=" * 50)
        print(f"Running {b_cfg.name}: prompts={prompt_count}, shards={shard_count}")
        print("=" * 50)

        processes: list[tuple[str, int, Path, subprocess.Popen]] = []
        for shard_index in range(shard_count):
            worker_log = output_dir / f"{_safe_name(b_cfg.name)}.shard_{shard_index:02d}.log"
            worker_argv = base_argv + [
                "fanout.role=worker",
                f"fanout.benchmark={b_cfg.name}",
                f"fanout.shard_index={shard_index}",
                f"fanout.num_shards={shard_count}",
                f"fanout.output_dir={output_dir}",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(shard_index % gpu_count)
            env["PYTHONUNBUFFERED"] = "1"
            log_handle = worker_log.open("w")
            process = subprocess.Popen(
                [sys.executable, "-m", "evaluation.sharded_main", *worker_argv],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            log_handle.close()
            processes.append((b_cfg.name, shard_index, worker_log, process))

        failed = False
        for benchmark_name, shard_index, worker_log, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed = True
                print(f"Shard failed: {benchmark_name} shard {shard_index}, log={worker_log}", file=sys.stderr)
                print(_tail(worker_log), file=sys.stderr)

        if failed:
            raise RuntimeError(f"One or more shards failed for {b_cfg.name}")

        metrics = _aggregate_benchmark(output_dir, b_cfg, shard_count)
        all_results[b_cfg.name] = metrics

        print(f"\n--- {b_cfg.name} ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<25}: {value:.4f}")
            else:
                print(f"{key:.<25}: {value}")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)

    print("\n" + "#" * 50 + "\nEVALUATION SUMMARY\n" + "#" * 50)
    for benchmark_name, metrics in all_results.items():
        print(f"\n--- {benchmark_name} ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<25}: {value:.4f}")
            else:
                print(f"{key:.<25}: {value}")
    print(f"\nSaved summary: {summary_path}")


def main() -> None:
    argv = sys.argv[1:]
    cfg, fanout_conf, _ = _load_config(argv)
    role = str(fanout_conf.get("role") or "parent")
    if role == "worker":
        _run_worker(argv, cfg, fanout_conf)
    elif role == "parent":
        _run_parent(argv, cfg, fanout_conf)
    else:
        raise ValueError(f"Unknown fanout.role: {role}")


if __name__ == "__main__":
    main()
