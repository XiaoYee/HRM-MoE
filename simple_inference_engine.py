from typing import Any, Iterator, Generator, Optional
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import json
import os
import yaml

import torch
from torch import Tensor, nn
import torch.utils._pytree as pytree
import numpy as np
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from transformers import AutoTokenizer, PreTrainedTokenizer

from pretrain import Carry, PretrainConfig, V1DatasetMeta, load_model_class, AdamATan2


@dataclass
class InferenceCheckpoint:
    model: nn.Module
    carry: Carry
    tokenizer: PreTrainedTokenizer
    tokenizer_info: dict[str, Any]

    def tokenize_prompt(self, condition: str, prompt: str) -> np.ndarray:
        condition_tokens = "".join(self.tokenizer_info["condition_mapping"][c] for c in condition.split(","))
        return self.tokenizer(f'{self.tokenizer_info["boq"]}{condition_tokens}{prompt}{self.tokenizer_info["eoq"]}',
                              return_tensors="np", return_attention_mask=False, add_special_tokens=False)["input_ids"][0]  # pyright: ignore[reportIndexIssue]
    
    def decode_generation(self, tokens: np.ndarray, eos_id: int) -> str:
        if tokens.size > 0 and tokens[-1] == eos_id:
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens)  # pyright: ignore[reportReturnType]


def _resolve_tokenizer_path(tokenizer_path: str) -> str:
    path = Path(tokenizer_path)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.name == "tokenizer.json":
            return str(resolved.parent)
        if resolved.exists():
            return str(resolved)

    return tokenizer_path


def _compile_for_eval(**kwargs):
    def decorator(fn):
        enabled = os.environ.get("HRM_EVAL_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}
        if enabled:
            return torch.compile(**kwargs)(fn)
        return fn
    return decorator


def inference_load_checkpoint(ckpt_path: str, ckpt_epoch: Optional[int], ckpt_use_ema: bool):
    # Load Checkpoint
    # Load config
    with open(os.path.join(ckpt_path, "all_config.yaml"), "r") as f:
        model_cfg = PretrainConfig(**yaml.safe_load(f))
    with open(os.path.join(ckpt_path, "train_metadata.yaml"), "r") as f:
        train_metadata = V1DatasetMeta(**yaml.safe_load(f))

    # Create model
    model_cls = load_model_class(model_cfg.arch.name)
    head_cls = load_model_class(model_cfg.arch.head)
    with torch.device("cuda"):
        combined_cfg = model_cfg.arch.model_dump() | train_metadata.model_dump() | model_cfg.data.model_dump()

        model: nn.Module = model_cls(combined_cfg)
        # Attach loss head
        model = head_cls(model, combined_cfg)
        # Optimizer (ONLY for loading states)
        optim = AdamATan2(model.parameters(),
                        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
                        betas=(model_cfg.beta1, model_cfg.beta2),
                        weight_decay=model_cfg.weight_decay,
                        ema=model_cfg.ema)
    
    # Detect checkpoint epoch if not specified
    if ckpt_epoch is None:
        ckpt_files = glob(os.path.join(ckpt_path, "fsdp2_epoch_*"))
        if len(ckpt_files) == 0:
            raise ValueError(f"No checkpoint files found in {ckpt_path}")

        ckpt_epoch = max(int(Path(f).stem.split("_")[-1]) for f in ckpt_files)
        print(f"Detected latest checkpoint epoch: {ckpt_epoch}")

    # Load checkpoint
    dcp.load({"model": model.state_dict(), "optim": get_optimizer_state_dict(model, optim)},  # pyright: ignore[reportPrivateImportUsage]
        checkpoint_id=os.path.join(ckpt_path, f"fsdp2_epoch_{ckpt_epoch}"),
        no_dist=True  # <--- Critical for single rank loading
    )
    carry = torch.load(os.path.join(ckpt_path, f"carry_epoch_{ckpt_epoch}.0.pt"), map_location="cuda")

    # Use EMA weights
    if ckpt_use_ema:
        optim.swap_ema()
    # Cast to fwd dtype & eval mode
    model = model.to(getattr(torch, model_cfg.fwd_bwd_dtype)).eval()

    # Load tokenizer
    tokenizer_path = _resolve_tokenizer_path(train_metadata.tokenizer_info["tokenizer_path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    return InferenceCheckpoint(
        model=model,
        carry=carry,
        tokenizer=tokenizer,
        tokenizer_info=train_metadata.tokenizer_info
    )


def _resolve_hf_checkpoint_dir(model_id_or_path: str, revision: Optional[str], local_files_only: bool) -> Path:
    path = Path(model_id_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=model_id_or_path,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    ))


def _hf_to_native_key(key: str) -> str:
    key = key.replace("model.H_module.layers.", "model.H_level.core.layers.")
    key = key.replace("model.L_module.layers.", "model.L_level.core.layers.")
    key = key.replace("model.z_L_init", "model.zL_init")
    if key == "model.embed_tokens.weight":
        return "embed_tokens.embedding_weight"
    return key


def _native_config_from_hf_config(hf_config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(hf_config["hidden_size"])
    initializer_range = float(hf_config.get("initializer_range", hidden_size ** -0.5))

    return {
        "vocab_size": int(hf_config["vocab_size"]),
        "max_seq_len": int(hf_config["max_position_embeddings"]),
        "n_layers": int(hf_config["num_hidden_layers"]),
        "half_layers": False,
        "hidden_size": hidden_size,
        "num_heads": int(hf_config["num_attention_heads"]),
        "expansion": 4.0,
        "norm_type": "pre",
        "norm_eps": float(hf_config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(hf_config.get("rope_theta", 10000.0)),
        "pos_emb_type": "rope",
        "init_type": "lecun_normal",
        "init_std": initializer_range,
        "attn_type": "prefixlm",
        "H_cycles": int(hf_config["H_cycles"]),
        "L_cycles": int(hf_config["L_cycles"]),
        "H_override": {},
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": 2,
        "bp_max_steps": 5,
        "moe_num_experts": int(hf_config.get("moe_num_experts", 0)),
        "moe_top_k": int(hf_config.get("moe_top_k", 1)),
        "moe_intermediate_size": (
            int(hf_config["moe_intermediate_size"])
            if hf_config.get("moe_intermediate_size") is not None
            else None
        ),
        "moe_norm_topk_prob": bool(hf_config.get("moe_norm_topk_prob", True)),
        "moe_router_aux_loss_coef": float(hf_config.get("moe_router_aux_loss_coef", 0.0)),
        "moe_implementation": hf_config.get("moe_implementation", "grouped_triton"),
        "moe_expert_in_one_shard": int(hf_config.get("moe_expert_in_one_shard", 1)),
    }


def _tokenizer_info_from_hf_config(hf_config: dict[str, Any], tokenizer: PreTrainedTokenizer) -> dict[str, Any]:
    return {
        "boq": tokenizer.bos_token or "<|im_start|>",
        "eoq": hf_config.get("eoq_token", "<|im_end|>"),
        "eoa": tokenizer.eos_token or "<|box_end|>",
        "condition_mapping": hf_config.get("condition_mapping", {
            "direct": "<|object_ref_start|>",
            "cot": "<|object_ref_end|>",
            "noisy": "<|quad_start|>",
            "synth": "<|quad_end|>",
        }),
    }


def inference_load_hf_checkpoint(
    model_id_or_path: str = "Xiaoye08/HRM-MoE",
    *,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> InferenceCheckpoint:
    """Load the single-file Hugging Face HRM-MoE release for native inference."""
    from safetensors.torch import load_file

    ckpt_dir = _resolve_hf_checkpoint_dir(model_id_or_path, revision, local_files_only)
    hf_config = json.loads((ckpt_dir / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True, local_files_only=True)
    tokenizer_info = _tokenizer_info_from_hf_config(hf_config, tokenizer)
    native_cfg = _native_config_from_hf_config(hf_config)

    model_cls = load_model_class("baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel")
    head_cls = load_model_class("lm_head@LMHead")
    with torch.device("cpu"):
        model: nn.Module = model_cls(native_cfg)
        model = head_cls(model, native_cfg)

    hf_state = load_file(ckpt_dir / "model.safetensors", device="cpu")
    native_state = {_hf_to_native_key(key): value for key, value in hf_state.items()}
    incompatible = model.load_state_dict(native_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Invalid HRM-MoE HF checkpoint keys: {incompatible}")

    model = model.to(device=device, dtype=dtype).eval()
    return InferenceCheckpoint(
        model=model,
        carry=None,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
    )


class HRMMoEForCausalLM:
    """Small generate-style wrapper around the native HRM-MoE inference engine."""

    def __init__(self, ckpt: InferenceCheckpoint) -> None:
        self.ckpt = ckpt
        self.tokenizer = ckpt.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str = "Xiaoye08/HRM-MoE",
        *,
        dtype: torch.dtype = torch.bfloat16,
        torch_dtype: Optional[torch.dtype] = None,
        device: str | torch.device = "cpu",
        revision: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "HRMMoEForCausalLM":
        if torch_dtype is not None:
            dtype = torch_dtype
        ckpt = inference_load_hf_checkpoint(
            model_id_or_path,
            revision=revision,
            local_files_only=local_files_only,
            device=device,
            dtype=dtype,
        )
        return cls(ckpt)

    @property
    def model(self) -> nn.Module:
        return self.ckpt.model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def cuda(self, device: Optional[int | torch.device] = None) -> "HRMMoEForCausalLM":
        self.ckpt.model = self.model.cuda(device)
        return self

    def to(self, *args, **kwargs) -> "HRMMoEForCausalLM":
        self.ckpt.model = self.model.to(*args, **kwargs)
        return self

    def eval(self) -> "HRMMoEForCausalLM":
        self.model.eval()
        return self

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[str],
        *,
        condition: str = "synth,cot",
        max_new_tokens: int = 256,
        max_length: int = 4096,
        do_sample: bool = False,
        temperature: float = 1.0,
        batch_size: int = 1,
    ) -> str | list[str]:
        if self.device.type != "cuda":
            raise RuntimeError("HRM-MoE native generation requires CUDA. Call .cuda() before generate().")

        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)
        temp = temperature if do_sample else 0.0
        outputs = [""] * len(prompts)
        iterator = ((idx, (condition, text)) for idx, text in enumerate(prompts))
        for idx, text in inference_generate(
            self.ckpt,
            iterator,
            max_tokens=max_length,
            max_generation=max_new_tokens,
            batch_size=batch_size,
            temp=temp,
        ):
            outputs[idx] = text
        return outputs[0] if single else outputs


@_compile_for_eval(fullgraph=True)
def _sample_gumbel(logits: Tensor, temp: Tensor):
    scaled_logits = logits.to(torch.float32) / temp
    return (scaled_logits - torch.log(-torch.log(torch.rand_like(scaled_logits).clamp_min(torch.finfo(scaled_logits.dtype).tiny)))).argmax(-1)


def _sample(logits: Tensor, temp: float) -> Tensor:
    if temp < 1e-5:
        return logits.argmax(-1)
    return _sample_gumbel(logits, torch.tensor(temp, dtype=torch.float32))


@_compile_for_eval(fullgraph=True)
def _prefill(model: nn.Module, carry: Carry, inputs: Tensor, cache: Any) -> Tensor:
    return model(carry=carry, batch={"inputs": inputs.unsqueeze(0), "position_ids": torch.arange(inputs.shape[0], device=inputs.device), "cache": cache, "cache_lengths": 0})[-1][..., -1, :]


@_compile_for_eval(dynamic=False, fullgraph=True)
def _batched_decode(model: nn.Module, carry: Carry, inputs: Tensor, cache: Any, cache_lengths: Tensor) -> Tensor:
    return model(carry=carry, batch={"inputs": inputs.unsqueeze(-1), "position_ids": cache_lengths.unsqueeze(-1), "cache": cache, "cache_lengths": cache_lengths})[-1][..., -1, :]


@torch.inference_mode()
def inference_generate(ckpt: InferenceCheckpoint, iterator: Iterator[tuple[int, tuple[str, str]]], max_tokens: int, max_generation: int, batch_size: int, temp: float = 0.0) -> Generator[tuple[int, str], None, None]:
    def fetch_next():
        for pid, p_tuple in iterator:
            tok = ckpt.tokenize_prompt(*p_tuple)
            # Check length: if it exceeds or equals max_tokens, it will overflow buffers
            if tok.size >= max_tokens:
                yield pid, ""  # Instantly yield empty string to the caller
            else:
                return pid, tok
        return -1, None

    # Stop condition
    stop_token: int = ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoa"])  # pyright: ignore[reportAssignmentType]

    # Create GPU tensors: KV-cache
    gpu_cache = ckpt.model.create_cache(max_batch_size=batch_size, max_seq_len=max_tokens, dtype=torch.bfloat16, device="cuda")  # FIXME: hardcoded dtype # pyright: ignore[reportCallIssue]
    gpu_cache_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    gpu_last_tokens = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")

    generated = np.zeros((batch_size, max_tokens), dtype=np.int64)
    generated_starts = np.zeros(batch_size, dtype=np.int64)
    generated_lengths = np.zeros(batch_size, dtype=np.int64)
    stopped = np.ones(batch_size, dtype=bool)

    # Output ID tracking
    generation_ids = [-1] * batch_size
    # Prefetch tokenized
    tokenized_prompt_id, tokenized_prompt = yield from fetch_next()

    while True:
        # PHASE 1: PREFILL & YIELD (Optimized for CPU-GPU Overlap)
        for i in stopped.nonzero()[0]:
            # Launch GPU prefill kernel
            launched_prefill = False
            if tokenized_prompt is not None:
                length = tokenized_prompt.size  # pyright: ignore[reportOptionalMemberAccess]
                inputs = torch.from_numpy(tokenized_prompt).cuda()  # <--- NOTE CPU to GPU (async)

                torch._dynamo.mark_dynamic(inputs, 0, min=1, max=max_tokens)
                gpu_last_tokens[i] = _sample(_prefill(ckpt.model, ckpt.carry, inputs, pytree.tree_map(lambda x: x[i: i+1], gpu_cache)), temp)[0]
                gpu_cache_lengths[i] = length
                launched_prefill = True

            # ---- De-tokenize & yield (Overlap with prefill)
            if generation_ids[i] != -1:
                yield generation_ids[i], ckpt.decode_generation(generated[i, generated_starts[i]: generated_lengths[i]], stop_token)
                generation_ids[i] = -1

            # ---- Prefetch tokenized (Overlap with prefill)
            if launched_prefill:
                generation_ids[i] = tokenized_prompt_id
                tokenized_prompt_id, tokenized_prompt = yield from fetch_next()

                generated_starts[i] = max(length, max_tokens - max_generation)  # pyright: ignore[reportPossiblyUnboundVariable]
                generated_lengths[i] = generated_starts[i] + 1

                last_tokens = gpu_last_tokens[i].item()  # <--- NOTE BLOCKING SYNC
                generated[i, generated_starts[i]] = last_tokens
                stopped[i] = (last_tokens == stop_token) or (generated_lengths[i] >= max_tokens)

        # PHASE 2: DECODE
        if not stopped.all():
            # Decode one token
            gpu_last_tokens = _sample(_batched_decode(ckpt.model, ckpt.carry, gpu_last_tokens, gpu_cache, gpu_cache_lengths), temp)
            gpu_cache_lengths.add_(1).clamp_max_(max_tokens - 1)  # Saturating add: prevent buffer overflow

            # Put to generated (Overlap with decode)
            active_mask = ~stopped
            generated_lengths[active_mask] += 1

            last_tokens = gpu_last_tokens.cpu().numpy()  # <--- NOTE BLOCKING SYNC
            generated[active_mask, generated_lengths[active_mask] - 1] = last_tokens[active_mask]
            stopped |= (last_tokens == stop_token) | (generated_lengths >= max_tokens)
        else:
            # Exit condition: if everything is stopped AND no more to prefill
            if tokenized_prompt is None:
                break
    
    # Flush: yield any remaining completed generations that were left in the pipeline
    for i in range(batch_size):
        if generation_ids[i] != -1:
            yield generation_ids[i], ckpt.decode_generation(generated[i, generated_starts[i]: generated_lengths[i]], stop_token)
