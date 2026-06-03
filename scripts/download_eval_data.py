#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests


EVAL_DATA_FILES: tuple[tuple[str, str], ...] = (
    ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "algebra/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "counting_and_probability/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "geometry/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "intermediate_algebra/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "number_theory/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "prealgebra/test-00000-of-00001.parquet"),
    ("EleutherAI/hendrycks_math", "precalculus/test-00000-of-00001.parquet"),
    ("EleutherAI/drop", "drop_validation.parquet"),
    ("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet"),
    ("cais/mmlu", "all/dev-00000-of-00001.parquet"),
    ("cais/mmlu", "all/test-00000-of-00001.parquet"),
    ("allenai/ai2_arc", "ARC-Challenge/validation-00000-of-00001.parquet"),
    ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
    ("Rowan/hellaswag", "data/train-00000-of-00001.parquet"),
    ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
    ("allenai/winogrande", "winogrande_debiased/train-00000-of-00001.parquet"),
    ("allenai/winogrande", "winogrande_debiased/validation-00000-of-00001.parquet"),
    ("google/boolq", "data/train-00000-of-00001.parquet"),
    ("google/boolq", "data/validation-00000-of-00001.parquet"),
    ("math-ai/aime25", "test.jsonl"),
)


def _validate_file(path: Path, relative_path: str) -> int:
    if relative_path.endswith(".jsonl"):
        rows = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                json.loads(line)
                rows += 1
        return rows

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return -1
    return pq.ParquetFile(path).metadata.num_rows


def _download_url(endpoint: str, repo_id: str, relative_path: str) -> str:
    return (
        f"{endpoint.rstrip('/')}/datasets/"
        f"{quote(repo_id, safe='/')}/resolve/main/{quote(relative_path, safe='/')}"
    )


def _download_one(
    output: Path,
    endpoint: str,
    repo_id: str,
    relative_path: str,
    timeout: int,
    chunk_size: int,
    force: bool,
) -> tuple[str, Path, int, bool]:
    destination = output / repo_id / relative_path
    label = f"{repo_id}/{relative_path}"

    if destination.is_file() and destination.stat().st_size > 0 and not force:
        return label, destination, _validate_file(destination, relative_path), True

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    url = _download_url(endpoint, repo_id, relative_path)

    with requests.get(url, stream=True, timeout=(30, timeout)) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    rows = _validate_file(tmp_path, relative_path)
    tmp_path.replace(destination)
    return label, destination, rows, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HRM eval benchmark parquet files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet"),
    )
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _download_one,
                args.output,
                args.endpoint,
                repo_id,
                relative_path,
                args.timeout,
                args.chunk_size,
                args.force,
            )
            for repo_id, relative_path in EVAL_DATA_FILES
        ]
        for future in as_completed(futures):
            try:
                label, destination, rows, reused = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))
                print(f"FAILED {exc}", file=sys.stderr)
                continue

            row_text = "unknown rows" if rows < 0 else f"{rows} rows"
            status = "ok cached" if reused else "ok downloaded"
            print(f"{status}: {label} -> {destination} ({row_text})")

    if failures:
        print(f"\n{len(failures)} download(s) failed.", file=sys.stderr)
        return 1

    print(f"\nAll eval data files are ready under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
