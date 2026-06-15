#!/usr/bin/env python3
"""Collect M1 SmoothQuant activation absmax stats for Cosmos3 TaylorSeer.

The stats are alpha-independent. Run one rank per GPU, then merge rank files:

  CUDA_VISIBLE_DEVICES=0 python tools/m1_smoothquant_calibrate.py --rank 0 --world-size 8
  python tools/m1_smoothquant_calibrate.py --merge --world-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO_ROOT / "assets/cosmos3_t2v_prompts_seed100_upsampled/prompts"
DEFAULT_OUT_DIR = Path("/root/cosmos/m1_smoothquant_stats")

GROUP_SPECS: dict[str, tuple[str, ...]] = {
    "input_layernorm_moe_gen": (
        "self_attn.add_q_proj",
        "self_attn.add_k_proj",
        "self_attn.add_v_proj",
    ),
    "post_attention_layernorm_moe_gen": (
        "mlp_moe_gen.gate_proj",
        "mlp_moe_gen.up_proj",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--steps-per-prompt", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--merge", action="store_true", help="Merge rank-local stats and exit.")

    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output-type", choices=("latent", "pt", "np", "pil"), default="latent")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--enable-safety-check", action="store_true")
    parser.add_argument("--enable-resolution-template", action="store_true")
    parser.add_argument("--enable-duration-template", action="store_true")
    parser.add_argument("--disable-progress-bar", dest="disable_progress_bar", action="store_true", default=True)
    parser.add_argument("--show-progress-bar", dest="disable_progress_bar", action="store_false")

    args = parser.parse_args()
    if args.world_size < 1:
        parser.error("--world-size must be at least 1")
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be in range [0, world-size)")
    if args.num_prompts < 1:
        parser.error("--num-prompts must be at least 1")
    if args.steps_per_prompt < 1:
        parser.error("--steps-per-prompt must be at least 1")
    if args.height < 1 or args.width < 1 or args.num_frames < 1:
        parser.error("--height, --width, and --num-frames must be positive")
    return args


def rank_path(out_dir: Path, rank: int) -> Path:
    return out_dir / f"act_absmax_rank{rank}.pt"


def final_path(out_dir: Path) -> Path:
    return out_dir / "act_absmax.pt"


def prompt_paths(prompt_dir: Path, num_prompts: int) -> list[Path]:
    paths = sorted(prompt_dir.glob("prompt_*.json"))
    if len(paths) < num_prompts:
        raise FileNotFoundError(f"found {len(paths)} prompt jsons under {prompt_dir}, need {num_prompts}")
    return paths[:num_prompts]


def shard_prompts(paths: list[Path], rank: int, world_size: int) -> list[tuple[int, Path]]:
    return [(index, path) for index, path in enumerate(paths) if index % world_size == rank]


def module_by_fqn(root: Any, fqn: str) -> Any:
    if hasattr(root, "get_submodule"):
        return root.get_submodule(fqn)
    module = root
    for part in fqn.split("."):
        if part.isdigit() and isinstance(module, (list, tuple)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def group_key_for_linear(linear_fqn: str) -> str | None:
    marker = ".self_attn."
    if marker in linear_fqn:
        layer_prefix, leaf = linear_fqn.split(marker, 1)
        if f"self_attn.{leaf}" in GROUP_SPECS["input_layernorm_moe_gen"]:
            return f"{layer_prefix}.input_layernorm_moe_gen"
        return None

    marker = ".mlp_moe_gen."
    if marker in linear_fqn:
        layer_prefix, leaf = linear_fqn.split(marker, 1)
        if leaf in {"gate_proj", "up_proj"}:
            return f"{layer_prefix}.post_attention_layernorm_moe_gen"
    return None


def source_linears_for_group(group_key: str) -> list[str]:
    for norm_suffix, consumer_suffixes in GROUP_SPECS.items():
        suffix = f".{norm_suffix}"
        if group_key.endswith(suffix):
            layer_prefix = group_key[: -len(suffix)]
            return [f"{layer_prefix}.{consumer_suffix}" for consumer_suffix in consumer_suffixes]
    raise KeyError(group_key)


def expected_group_keys(transformer: Any) -> list[str]:
    layers = getattr(transformer, "layers", [])
    return [f"layers.{index}.{norm_suffix}" for index in range(len(layers)) for norm_suffix in GROUP_SPECS]


def collectable_linear_names(transformer: Any) -> list[str]:
    import torch.nn as nn

    names: list[str] = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and group_key_for_linear(name) is not None:
            names.append(name)
    return names


def register_absmax_hooks(transformer: Any) -> tuple[list[Any], dict[str, Any], dict[str, list[str]]]:
    import torch
    import torch.nn as nn

    running: dict[str, torch.Tensor] = {}
    source_linears: dict[str, list[str]] = {key: source_linears_for_group(key) for key in expected_group_keys(transformer)}
    handles = []

    def make_hook(group_key: str, linear_name: str, in_features: int):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            with torch.no_grad():
                flat = x.detach().reshape(-1, in_features).to(dtype=torch.float32)
                current = flat.abs().amax(dim=0)
                prior = running.get(group_key)
                if prior is None:
                    running[group_key] = current
                else:
                    if prior.device != current.device:
                        current = current.to(prior.device)
                    running[group_key] = torch.maximum(prior, current)

        hook._m1_smoothquant_linear_name = linear_name  # type: ignore[attr-defined]
        return hook

    for linear_name in collectable_linear_names(transformer):
        group_key = group_key_for_linear(linear_name)
        if group_key is None:
            continue
        module = module_by_fqn(transformer, linear_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{linear_name} is not nn.Linear")
        handles.append(module.register_forward_pre_hook(make_hook(group_key, linear_name, module.in_features)))

    return handles, running, source_linears


def stats_to_cpu(running: dict[str, Any], group_keys: list[str]) -> dict[str, Any]:
    import torch

    result: dict[str, torch.Tensor] = {}
    missing = [key for key in group_keys if key not in running]
    if missing:
        raise RuntimeError(f"no activation stats collected for {len(missing)} groups, first missing: {missing[0]}")
    for key in group_keys:
        result[key] = running[key].detach().to(device="cpu", dtype=torch.float32)
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def transformer_config_digest(transformer: Any) -> tuple[str | None, str | None]:
    config = getattr(transformer, "config", None)
    if config is None:
        return None, None
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), text[:2000]


def select_device(args: argparse.Namespace) -> str:
    import torch

    device = args.device
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return device
    if ":" in device:
        torch.cuda.set_device(device)
        return device
    count = torch.cuda.device_count()
    if count < 1:
        return device
    local_index = args.rank % count
    selected = f"cuda:{local_index}"
    torch.cuda.set_device(selected)
    return selected


def load_pipeline(args: argparse.Namespace):
    from diffusers import Cosmos3OmniTaylorSeerPipeline, Cosmos3OmniTaylorSeerTransformer

    from diffusers_m1fp8_t2v_benchmark import configure_pipeline, dtype_from_name

    dtype = dtype_from_name(args.dtype)
    transformer_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.device.startswith("cuda"):
        transformer_kwargs["device_map"] = args.device
    transformer = Cosmos3OmniTaylorSeerTransformer.from_pretrained(
        args.model,
        subfolder="transformer",
        **transformer_kwargs,
    )
    pipeline_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "enable_safety_checker": args.enable_safety_check,
    }
    if args.device.startswith("cuda"):
        pipeline_kwargs["device_map"] = "cuda"
    pipe = Cosmos3OmniTaylorSeerPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        **pipeline_kwargs,
    )
    configure_pipeline(pipe, args)
    # Do not call enable_taylorseer(): calibration should observe every denoise step
    # as a real full transformer pass, with no cache/predicted steps in the stats.
    return pipe


def run_rank(args: argparse.Namespace) -> Path:
    import torch

    from diffusers_m1fp8_t2v_benchmark import assert_local_diffusers, ensure_local_diffusers_source, load_prompt

    local_diffusers_dir = ensure_local_diffusers_source()
    import diffusers

    diffusers_source = assert_local_diffusers(diffusers.__file__, local_diffusers_dir)
    args.device = select_device(args)
    selected_prompts = prompt_paths(args.prompt_dir, args.num_prompts)
    rank_prompts = shard_prompts(selected_prompts, args.rank, args.world_size)
    if not rank_prompts:
        raise RuntimeError(f"rank {args.rank} received no prompts")

    print(
        json.dumps(
            {
                "event": "m1_smoothquant_calibrate_rank_start",
                "rank": args.rank,
                "world_size": args.world_size,
                "prompt_count": len(rank_prompts),
                "num_prompts_total": args.num_prompts,
                "steps_per_prompt": args.steps_per_prompt,
                "device": args.device,
                "diffusers_source": diffusers_source,
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    pipe = load_pipeline(args)
    handles, running, source_linears = register_absmax_hooks(pipe.transformer)
    group_keys = expected_group_keys(pipe.transformer)
    config_hash, config_excerpt = transformer_config_digest(pipe.transformer)

    prompt_records = []
    started = time.perf_counter()
    try:
        for global_index, prompt_path in rank_prompts:
            prompt = load_prompt(prompt_path, "")
            observed_steps: list[dict[str, int]] = []

            def on_step_end(_pipe, step_index: int, timestep, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                step_value = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
                observed_steps.append({"step_index": int(step_index), "timestep": step_value})
                return callback_kwargs

            generator = torch.Generator(device=args.device).manual_seed(args.seed + global_index) if args.device.startswith("cuda") else torch.Generator().manual_seed(args.seed + global_index)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    image=None,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    fps=args.fps,
                    num_inference_steps=args.steps_per_prompt,
                    guidance_scale=args.guidance_scale,
                    enable_sound=False,
                    generator=generator,
                    output_type=args.output_type,
                    add_resolution_template=args.enable_resolution_template,
                    add_duration_template=args.enable_duration_template,
                    enable_safety_check=args.enable_safety_check,
                    callback_on_step_end=on_step_end,
                )
            del result
            prompt_records.append(
                {
                    "global_prompt_index": global_index,
                    "path": str(prompt_path),
                    "observed_step_count": len(observed_steps),
                    "observed_timesteps": [entry["timestep"] for entry in observed_steps],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "m1_smoothquant_prompt_done",
                        "rank": args.rank,
                        "global_prompt_index": global_index,
                        "observed_step_count": len(observed_steps),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = rank_path(args.out_dir, args.rank)
    payload = {
        "act_absmax": stats_to_cpu(running, group_keys),
        "metadata": {
            "format": "m1_smoothquant_act_absmax_v1",
            "rank": args.rank,
            "world_size": args.world_size,
            "num_prompts_total": args.num_prompts,
            "rank_prompt_count": len(rank_prompts),
            "steps_per_prompt": args.steps_per_prompt,
            "step_coverage_strategy": "scheduler trajectory compressed by setting num_inference_steps=steps_per_prompt; default 10 covers early/mid/late denoise timesteps without TaylorSeer cache",
            "taylorseer_cache_enabled": False,
            "taylorseer_enabled": False,
            "model": args.model,
            "model_config_sha256": config_hash,
            "model_config_excerpt": config_excerpt,
            "source_linears": source_linears,
            "prompt_records": prompt_records,
            "elapsed_seconds": time.perf_counter() - started,
            "torch_version": str(torch.__version__),
            "torchao_version": package_version("torchao"),
            "diffusers_source": diffusers_source,
            "device": args.device,
            "dtype": args.dtype,
            "output_type": args.output_type,
        },
    }
    torch.save(payload, out_path)
    print(json.dumps({"event": "m1_smoothquant_rank_saved", "path": str(out_path)}, sort_keys=True), flush=True)
    return out_path


def merge_rank_stats(args: argparse.Namespace) -> Path:
    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, torch.Tensor] = {}
    rank_metadata = []
    expected_keys: set[str] | None = None

    for rank in range(args.world_size):
        path = rank_path(args.out_dir, rank)
        if not path.is_file():
            raise FileNotFoundError(f"missing rank stats: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        act_absmax = payload.get("act_absmax")
        if not isinstance(act_absmax, dict):
            raise TypeError(f"{path} does not contain an act_absmax dict")
        keys = set(act_absmax)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError(f"rank {rank} keys differ from earlier rank files")
        for key, value in act_absmax.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{path}:{key} is not a tensor")
            value = value.detach().to(device="cpu", dtype=torch.float32)
            if key not in merged:
                merged[key] = value
            else:
                if merged[key].shape != value.shape:
                    raise RuntimeError(f"shape mismatch for {key}: {merged[key].shape} vs {value.shape}")
                merged[key] = torch.maximum(merged[key], value)
        rank_metadata.append(payload.get("metadata", {}))

    out_path = final_path(args.out_dir)
    payload = {
        "act_absmax": merged,
        "metadata": {
            "format": "m1_smoothquant_act_absmax_v1",
            "merged": True,
            "world_size": args.world_size,
            "rank_files": [str(rank_path(args.out_dir, rank)) for rank in range(args.world_size)],
            "rank_metadata": rank_metadata,
            "group_count": len(merged),
            "merge_rule": "per-key per-channel torch.maximum across rank-local absmax tensors",
            "created_at_unix": time.time(),
        },
    }
    torch.save(payload, out_path)
    print(json.dumps({"event": "m1_smoothquant_merged", "path": str(out_path), "group_count": len(merged)}, sort_keys=True), flush=True)
    return out_path


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_rank_stats(args)
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
