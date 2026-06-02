#!/usr/bin/env python3
"""Compare two data_io tokenized output directories.

Use this on a small sample after running both the official Rust tokenizer and
the Python fallback. It checks task presence, tokenizer_info.json, per-task
metadata, and all token/index arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_FILES = ("tokens.npy", "inst_start.npy", "inst_len.npy", "resp_start.npy", "resp_len.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path, help="Reference tokenized output, usually Rust")
    parser.add_argument("actual", type=Path, help="Output to verify, usually Python fallback")
    parser.add_argument("--chunk-size", type=int, default=8_000_000)
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Only compare arrays and tokenizer_info.json; ignore per-task metadata.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r") as handle:
        return json.load(handle)


def task_dirs(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "tokens.npy").exists()
    }


def compare_array(expected_path: Path, actual_path: Path, chunk_size: int) -> None:
    expected = np.load(expected_path, mmap_mode="r")
    actual = np.load(actual_path, mmap_mode="r")

    if expected.dtype != actual.dtype:
        raise AssertionError(f"dtype mismatch for {expected_path.name}: {expected.dtype} != {actual.dtype}")
    if expected.shape != actual.shape:
        raise AssertionError(f"shape mismatch for {expected_path.name}: {expected.shape} != {actual.shape}")

    total = int(expected.shape[0])
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        if not np.array_equal(expected[start:end], actual[start:end]):
            raise AssertionError(f"value mismatch for {expected_path.name} at slice [{start}:{end}]")


def main() -> None:
    args = parse_args()

    expected_tasks = task_dirs(args.expected)
    actual_tasks = task_dirs(args.actual)
    if expected_tasks != actual_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        extra = sorted(actual_tasks - expected_tasks)
        raise AssertionError(f"task set mismatch: missing={missing[:10]} extra={extra[:10]}")

    if load_json(args.expected / "tokenizer_info.json") != load_json(args.actual / "tokenizer_info.json"):
        raise AssertionError("tokenizer_info.json mismatch")

    for task in sorted(expected_tasks):
        expected_dir = args.expected / task
        actual_dir = args.actual / task

        if not args.skip_metadata and load_json(expected_dir / "metadata.json") != load_json(actual_dir / "metadata.json"):
            raise AssertionError(f"metadata.json mismatch for {task}")

        for name in ARRAY_FILES:
            compare_array(expected_dir / name, actual_dir / name, args.chunk_size)

    print(f"tokenized_outputs_match expected={args.expected} actual={args.actual} tasks={len(expected_tasks)}")


if __name__ == "__main__":
    main()
