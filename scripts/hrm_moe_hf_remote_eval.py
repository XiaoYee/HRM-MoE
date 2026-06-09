import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.benchmarks import GSM8k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Xiaoye08/HRM-MoE")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["usage", "gsm8k", "aggregate"], default="usage")
    parser.add_argument("--condition", default="synth,cot")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def condition_tokens(condition: str, mapping: dict[str, str]) -> str:
    return "".join(mapping[name] for name in condition.split(","))


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(tokenizer, model, prompt: str, condition: str, max_new_tokens: int) -> str:
    cfg = model.config
    prefix = condition_tokens(condition, cfg.condition_mapping)
    text = f"{tokenizer.bos_token}{prefix}{prompt}<|im_end|>"
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False).to(model.device)
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    generated_ids = output_ids[inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def run_usage(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model_id, args.device)
    prompt = "Explain why the sky is blue."
    generation = generate_one(tokenizer, model, prompt, args.condition, args.max_new_tokens)
    record = {
        "model_id": args.model_id,
        "prompt": prompt,
        "condition": args.condition,
        "generation": generation,
    }
    (output_dir / "usage.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def run_gsm8k(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model_id, args.device)
    benchmark = GSM8k(split="test")
    rows = []
    shard_indices = [i for i in range(len(benchmark.prompts)) if i % args.shard_count == args.shard_index]
    for idx in tqdm(shard_indices, desc=f"gsm8k shard {args.shard_index}/{args.shard_count}"):
        generation = generate_one(tokenizer, model, benchmark.prompts[idx].strip(), args.condition, args.max_new_tokens)
        pred = benchmark._extract_answer(generation)
        truth = benchmark.ground_truths[idx]
        rows.append({
            "idx": idx,
            "truth": truth,
            "prediction": pred,
            "correct": pred == truth,
            "invalid": pred is None,
            "generation": generation,
        })

    out_path = output_dir / f"gsm8k_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.jsonl"
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_rows(rows)
    summary["shard_index"] = args.shard_index
    summary["shard_count"] = args.shard_count
    (output_dir / f"gsm8k_shard_{args.shard_index:02d}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    invalid = sum(1 for row in rows if row["invalid"])
    return {
        "n": total,
        "correct": correct,
        "invalid_count": invalid,
        "acc": correct / max(1, total),
        "invalid": invalid / max(1, total),
    }


def run_aggregate(args) -> None:
    output_dir = Path(args.output_dir)
    rows = []
    for path in sorted(output_dir.glob("gsm8k_shard_*_of_*.jsonl")):
        with path.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda row: row["idx"])
    if len(rows) != 1319:
        raise SystemExit(f"expected full GSM8k test set n=1319, got n={len(rows)}")
    summary = summarize_rows(rows)
    summary["benchmark"] = "GSM8k"
    summary["split"] = "test"
    summary["readme_acc"] = 0.8499
    summary["delta_vs_readme"] = summary["acc"] - summary["readme_acc"]
    (output_dir / "gsm8k_full_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.mode == "usage":
        run_usage(args)
    elif args.mode == "gsm8k":
        run_gsm8k(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
