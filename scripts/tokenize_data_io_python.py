#!/usr/bin/env python3
"""Tokenize data_io cleaned files without the Rust tokenizer binary.

The output layout matches data_io/tokenizer:
  <output>/<safe_name>/{tokens,inst_start,inst_len,resp_start,resp_len}.npy
  <output>/<safe_name>/metadata.json
  <output>/tokenizer_info.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    import orjson
except Exception:  # pragma: no cover - plain json fallback is fine
    orjson = None


TOKEN_FLUSH_VALUES = 1_000_000
INDEX_FLUSH_VALUES = 1_000_000
NPY_COPY_VALUES = 8_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--boq", default="<|im_start|>")
    parser.add_argument("--eoq", default="<|im_end|>")
    parser.add_argument("--eoa", default="<|box_end|>")
    parser.add_argument(
        "--conditions",
        default="direct=<|object_ref_start|>,cot=<|object_ref_end|>,noisy=<|quad_start|>,synth=<|quad_end|>",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def load_json_line(line: bytes) -> dict:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def scan_inputs(input_dirs: list[Path]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for input_dir in input_dirs:
        for path in input_dir.rglob("*"):
            if path.suffix not in {".jsonl", ".parquet"}:
                continue
            safe_name = str(path.relative_to(input_dir)).replace("/", "__").replace("\\", "__")
            files.append((path, safe_name))
    return sorted(files, key=lambda item: item[0].stat().st_size, reverse=True)


def should_process(input_path: Path, output_subdir: Path) -> bool:
    meta_path = output_subdir / "metadata.json"
    if not meta_path.exists():
        return True
    try:
        cached = json.loads(meta_path.read_text())
        stat = input_path.stat()
    except Exception:
        return True
    return cached.get("source_size") != stat.st_size or cached.get("source_mtime") != int(stat.st_mtime)


def iter_jsonl(path: Path) -> Iterable[tuple[str, str, str]]:
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = load_json_line(line)
            yield row["condition"], row["instruction"], row["response"]


def iter_parquet(path: Path, batch_size: int) -> Iterable[tuple[str, str, str]]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["condition", "instruction", "response"],
    ):
        data = batch.to_pydict()
        yield from zip(data["condition"], data["instruction"], data["response"])


def token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id_value = tokenizer.token_to_id(token)
    if token_id_value is None:
        raise ValueError(f"special token missing from tokenizer: {token!r}")
    return int(token_id_value)


def flush_values(handle, values: list[int], dtype: np.dtype) -> None:
    if values:
        np.asarray(values, dtype=dtype).tofile(handle)
        values.clear()


def raw_to_npy(raw_path: Path, npy_path: Path, dtype: np.dtype, count: int) -> None:
    output = np.lib.format.open_memmap(npy_path, mode="w+", dtype=dtype, shape=(count,))
    copied = 0
    with raw_path.open("rb") as handle:
        while copied < count:
            chunk = np.fromfile(handle, dtype=dtype, count=min(NPY_COPY_VALUES, count - copied))
            if chunk.size == 0:
                break
            output[copied : copied + chunk.size] = chunk
            copied += int(chunk.size)
    output.flush()
    del output
    if copied != count:
        raise IOError(f"short raw read while writing {npy_path}: copied={copied} expected={count}")
    raw_path.unlink()


def process_file(
    input_path: Path,
    output_subdir: Path,
    tokenizer_path: str,
    condition_ids: dict[str, int],
    boq_id: int,
    eoq_id: int,
    eoa_id: int,
    batch_size: int,
) -> None:
    tokenizer = Tokenizer.from_file(tokenizer_path)

    output_subdir.mkdir(parents=True, exist_ok=True)
    tmp_paths = {
        "tokens": output_subdir / ".tokens.u32.tmp",
        "inst_start": output_subdir / ".inst_start.u64.tmp",
        "inst_len": output_subdir / ".inst_len.u64.tmp",
        "resp_start": output_subdir / ".resp_start.u64.tmp",
        "resp_len": output_subdir / ".resp_len.u64.tmp",
    }
    for tmp_path in tmp_paths.values():
        tmp_path.unlink(missing_ok=True)

    token_buffer: list[int] = []
    inst_start_buffer: list[int] = []
    inst_len_buffer: list[int] = []
    resp_start_buffer: list[int] = []
    resp_len_buffer: list[int] = []
    token_count = 0
    row_count = 0

    def append_token(value: int, token_handle) -> None:
        nonlocal token_count
        token_buffer.append(value)
        token_count += 1
        if len(token_buffer) >= TOKEN_FLUSH_VALUES:
            flush_values(token_handle, token_buffer, np.uint32)

    def append_tokens(values: list[int], token_handle) -> None:
        nonlocal token_count
        token_buffer.extend(values)
        token_count += len(values)
        if len(token_buffer) >= TOKEN_FLUSH_VALUES:
            flush_values(token_handle, token_buffer, np.uint32)

    def flush_index_buffers(handles: dict[str, object]) -> None:
        flush_values(handles["inst_start"], inst_start_buffer, np.uint64)
        flush_values(handles["inst_len"], inst_len_buffer, np.uint64)
        flush_values(handles["resp_start"], resp_start_buffer, np.uint64)
        flush_values(handles["resp_len"], resp_len_buffer, np.uint64)

    with (
        tmp_paths["tokens"].open("wb") as token_handle,
        tmp_paths["inst_start"].open("wb") as inst_start_handle,
        tmp_paths["inst_len"].open("wb") as inst_len_handle,
        tmp_paths["resp_start"].open("wb") as resp_start_handle,
        tmp_paths["resp_len"].open("wb") as resp_len_handle,
    ):
        index_handles = {
            "inst_start": inst_start_handle,
            "inst_len": inst_len_handle,
            "resp_start": resp_start_handle,
            "resp_len": resp_len_handle,
        }
        rows = iter_parquet(input_path, batch_size) if input_path.suffix == ".parquet" else iter_jsonl(input_path)
        for condition, instruction, response in rows:
            inst_ids = tokenizer.encode(instruction, add_special_tokens=False).ids
            resp_ids = tokenizer.encode(response, add_special_tokens=False).ids

            i_start = token_count
            append_token(boq_id, token_handle)
            for cond in condition.split(","):
                append_token(condition_ids[cond], token_handle)
            append_tokens(inst_ids, token_handle)
            append_token(eoq_id, token_handle)
            inst_start_buffer.append(i_start)
            inst_len_buffer.append(token_count - i_start)

            r_start = token_count
            append_tokens(resp_ids, token_handle)
            append_token(eoa_id, token_handle)
            resp_start_buffer.append(r_start)
            resp_len_buffer.append(token_count - r_start)

            row_count += 1
            if len(inst_start_buffer) >= INDEX_FLUSH_VALUES:
                flush_index_buffers(index_handles)

        flush_values(token_handle, token_buffer, np.uint32)
        flush_index_buffers(index_handles)

    raw_to_npy(tmp_paths["tokens"], output_subdir / "tokens.npy", np.uint32, token_count)
    raw_to_npy(tmp_paths["inst_start"], output_subdir / "inst_start.npy", np.uint64, row_count)
    raw_to_npy(tmp_paths["inst_len"], output_subdir / "inst_len.npy", np.uint64, row_count)
    raw_to_npy(tmp_paths["resp_start"], output_subdir / "resp_start.npy", np.uint64, row_count)
    raw_to_npy(tmp_paths["resp_len"], output_subdir / "resp_len.npy", np.uint64, row_count)

    stat = input_path.stat()
    (output_subdir / "metadata.json").write_text(
        json.dumps({"source_mtime": int(stat.st_mtime), "source_size": stat.st_size})
    )


def main() -> None:
    args = parse_args()
    tokenizer_path = str((Path(args.tokenizer_path) / "tokenizer.json") if Path(args.tokenizer_path).is_dir() else args.tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    condition_mapping = dict(pair.split("=", 1) for pair in args.conditions.split(","))
    condition_ids = {key: token_id(tokenizer, value) for key, value in condition_mapping.items()}
    boq_id = token_id(tokenizer, args.boq)
    eoq_id = token_id(tokenizer, args.eoq)
    eoa_id = token_id(tokenizer, args.eoa)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tokenizer_info.json").write_text(
        json.dumps(
            {
                "tokenizer_path": args.tokenizer_path,
                "boq": args.boq,
                "eoq": args.eoq,
                "eoa": args.eoa,
                "condition_mapping": condition_mapping,
                "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            }
        )
    )

    files = scan_inputs(args.dirs)
    expected_outputs = {args.output_dir / safe_name for _, safe_name in files}
    for path in args.output_dir.iterdir() if args.output_dir.exists() else []:
        if path.is_dir() and path not in expected_outputs:
            print(f"orphan_output_present {path}")

    pending = [
        (input_path, args.output_dir / safe_name)
        for input_path, safe_name in files
        if should_process(input_path, args.output_dir / safe_name)
    ]
    print(f"files_total={len(files)} files_pending={len(pending)} workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_file,
                input_path,
                output_subdir,
                tokenizer_path,
                condition_ids,
                boq_id,
                eoq_id,
                eoa_id,
                args.batch_size,
            )
            for input_path, output_subdir in pending
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing files"):
            future.result()


if __name__ == "__main__":
    main()
