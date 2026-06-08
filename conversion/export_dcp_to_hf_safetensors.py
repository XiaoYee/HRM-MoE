import argparse
import json
import math
import pickle
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import save_file
from transformers import AutoTokenizer


SKIP_PREFIXES = (
    "model.H_level.core.rotary_emb.",
    "model.L_level.core.rotary_emb.",
)
DROP_KEYS = {"model.zH_init"}


def remap_key(key: str) -> str | None:
    if key in DROP_KEYS or key.startswith(SKIP_PREFIXES):
        return None
    key = key.replace("model.H_level.core.layers.", "model.H_module.layers.")
    key = key.replace("model.L_level.core.layers.", "model.L_module.layers.")
    key = key.replace("model.zL_init", "model.z_L_init")
    if key == "embed_tokens.embedding_weight":
        return "model.embed_tokens.weight"
    return key


def parse_dtype(value: str) -> torch.dtype:
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown torch dtype: {value}") from exc


def compute_l_bp_cycles(cfg: dict) -> list[int]:
    h_cycles, l_cycles = int(cfg["H_cycles"]), int(cfg["L_cycles"])
    bp_steps = int(cfg.get("bp_max_steps", cfg.get("max_bp_steps", h_cycles + 1)))
    h_bp_steps = min(h_cycles, max(0, bp_steps - 1))
    l_bp_steps = min(h_cycles * l_cycles, max(0, bp_steps - h_bp_steps))
    threshold = h_cycles * l_cycles - l_bp_steps
    return [max(0, min(l_cycles, (i + 1) * l_cycles - threshold)) for i in range(h_cycles)]


def initializer_range(cfg: dict) -> float:
    hidden_size = int(cfg["hidden_size"])
    init_type = cfg.get("init_type", "fixed_normal")
    init_std = cfg.get("init_std")
    if init_type == "lecun_normal":
        return 1.0 / math.sqrt(hidden_size)
    if init_std is not None:
        return float(init_std)
    if init_type == "megatron":
        return 1.0 / math.sqrt(hidden_size)
    return 0.02


def build_config(train_cfg: dict, metadata: dict, tokenizer) -> dict:
    arch = train_cfg["arch"]
    init_std = initializer_range(arch)
    per_stack_layers = int(arch["n_layers"]) // 2 if arch.get("half_layers") else int(arch["n_layers"])
    token_info = metadata["tokenizer_info"]
    cfg = {
        "model_type": "hrm_text_moe",
        "architectures": ["HrmTextMoEForCausalLM"],
        "vocab_size": int(metadata["vocab_size"]),
        "hidden_size": int(arch["hidden_size"]),
        "intermediate_size": int(arch.get("moe_intermediate_size", 0)),
        "num_hidden_layers": per_stack_layers,
        "num_attention_heads": int(arch["num_heads"]),
        "num_key_value_heads": int(arch["num_heads"]),
        "head_dim": int(arch["hidden_size"]) // int(arch["num_heads"]),
        "H_cycles": int(arch["H_cycles"]),
        "L_cycles": int(arch["L_cycles"]),
        "L_bp_cycles": compute_l_bp_cycles(arch),
        "max_position_embeddings": int(metadata["max_seq_len"]),
        "rms_norm_eps": float(arch.get("norm_eps", 1e-6)),
        "rope_theta": float(arch.get("rope_theta", 10000.0)),
        "tie_word_embeddings": False,
        "initializer_range": init_std,
        "embedding_scale": 1.0 / init_std,
        "prefix_lm": True,
        "moe_num_experts": int(arch["moe_num_experts"]),
        "moe_top_k": int(arch["moe_top_k"]),
        "moe_intermediate_size": int(arch["moe_intermediate_size"]),
        "moe_implementation": arch.get("moe_implementation", "grouped_triton"),
        "moe_norm_topk_prob": bool(arch.get("moe_norm_topk_prob", True)),
        "moe_router_aux_loss_coef": float(arch.get("moe_router_aux_loss_coef", 0.0)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or 0,
        "condition_mapping": token_info.get("condition_mapping", {}),
    }
    for key, token_name in (("bos_token_id", "boq"), ("eos_token_id", "eoa")):
        token = token_info.get(token_name)
        if token is not None:
            cfg[key] = int(tokenizer.convert_tokens_to_ids(token))
    return cfg


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_metadata(checkpoint_id: Path):
    with (checkpoint_id / ".metadata").open("rb") as f:
        return pickle.load(f)


def tokenizer_path(metadata: dict, checkpoint_root: Path, override: Path | None) -> Path:
    if override is not None:
        path = override
    else:
        raw_path = Path(metadata["tokenizer_info"]["tokenizer_path"])
        path = raw_path if raw_path.is_absolute() else checkpoint_root / raw_path
    return path.parent if path.name == "tokenizer.json" else path


def set_tokenizer_special_tokens(tokenizer, token_info: dict):
    if "boq" in token_info:
        tokenizer.bos_token = token_info["boq"]
    if "eoa" in token_info:
        tokenizer.eos_token = token_info["eoa"]
    return tokenizer


def export_weights(args):
    checkpoint_id = args.ckpt_path / f"fsdp2_epoch_{args.ckpt_epoch}"
    dcp_metadata = load_metadata(checkpoint_id)
    state_meta = dcp_metadata.state_dict_metadata

    model_inner_keys = sorted(k.removeprefix("model.") for k in state_meta if k.startswith("model."))
    ema_inner_keys = {
        k.removeprefix("optim.state.").removesuffix(".param_ema")
        for k in state_meta
        if k.startswith("optim.state.") and k.endswith(".param_ema")
    }

    load_tensors: dict[str, torch.Tensor] = {}
    key_for_inner: dict[str, str] = {}
    for inner_key in model_inner_keys:
        if args.use_ema and inner_key in ema_inner_keys:
            full_key = f"optim.state.{inner_key}.param_ema"
        else:
            full_key = f"model.{inner_key}"
        meta = state_meta[full_key]
        load_tensors[full_key] = torch.empty(tuple(meta.size), dtype=meta.properties.dtype, device="cpu")
        key_for_inner[inner_key] = full_key

    print(f"[export] loading {len(load_tensors)} tensors from {checkpoint_id}")
    dcp.load(load_tensors, checkpoint_id=str(checkpoint_id), no_dist=True)

    hf_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for inner_key in model_inner_keys:
        out_key = remap_key(inner_key)
        tensor = load_tensors.pop(key_for_inner[inner_key])
        if out_key is None:
            skipped.append(inner_key)
            continue
        if tensor.dtype != args.save_dtype:
            tensor = tensor.to(args.save_dtype)
        hf_state[out_key] = tensor.contiguous()

    print(f"[export] mapped {len(hf_state)} tensors; skipped {len(skipped)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        hf_state,
        args.out_dir / "model.safetensors",
        metadata={
            "format": "pt",
            "source_checkpoint": str(args.ckpt_path),
            "checkpoint_epoch": str(args.ckpt_epoch),
            "weights": "ema" if args.use_ema else "model",
            "dtype": str(args.save_dtype).removeprefix("torch."),
        },
    )
    print(f"[export] wrote {args.out_dir / 'model.safetensors'}")


def export_metadata(args):
    train_cfg = load_yaml(args.ckpt_path / "all_config.yaml")
    metadata = load_yaml(args.ckpt_path / "train_metadata.yaml")
    tok_path = tokenizer_path(metadata, args.ckpt_path, args.tokenizer_path)
    print(f"[export] using tokenizer at {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    tokenizer = set_tokenizer_special_tokens(tokenizer, metadata["tokenizer_info"])
    tokenizer.save_pretrained(args.out_dir)
    (args.out_dir / "config.json").write_text(
        json.dumps(build_config(train_cfg, metadata, tokenizer), indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--ckpt-epoch", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--save-dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    export_metadata(args)
    export_weights(args)


if __name__ == "__main__":
    main()
