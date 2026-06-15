#!/usr/bin/env python3
"""Per-layer gen-attention sensitivity analysis for Cosmos3 M3 Phase C.

This script runs one prompt_068 denoising pass with TaylorSeer cache + M1
SmoothQuant/FP8 + dense M2 SPARGE attention, captures real generation-path q/k/v
at selected full-attention steps, and compares PyTorch dense SDPA against
SpargeAttn topk block-sparse attention on the same tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_BENCH_PATH = REPO_ROOT / "tools" / "diffusers_m1m2sparse_t2v_benchmark.py"
DEFAULT_PROMPT_JSON = REPO_ROOT / "assets" / "cosmos3_t2v_prompts_seed100_upsampled" / "prompts" / "prompt_068.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "m3_perlayer" / "sensitivity_068.json"


def load_sparse_benchmark_module():
    spec = importlib.util.spec_from_file_location("m1m2sparse_bench", SPARSE_BENCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import sparse benchmark helpers from {SPARSE_BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_steps(text: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError("capture steps must be zero-based non-negative integers")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("at least one capture step is required")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--prompt-id", default="068")
    parser.add_argument("--prompt-json", type=Path, default=DEFAULT_PROMPT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--capture-steps", default="2,18,34", help="0-based denoise step indices to capture")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output-type", choices=("latent", "pt", "np", "pil"), default="latent")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--sparse-topk", type=float, default=0.5)
    parser.add_argument("--taylorseer-interval", type=int, default=2)
    parser.add_argument("--taylorseer-max-order", type=int, default=1)
    parser.add_argument("--taylorseer-first-enhance", type=int, default=1)
    parser.add_argument("--taylorseer-last-enhance", type=int, default=5)
    parser.add_argument("--taylorseer-cache-max-gib", type=float, default=64.0)
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument(
        "--smoothquant-stats-path",
        type=Path,
        default=REPO_ROOT / "m1_smoothquant_stats" / "act_absmax.pt",
    )
    args = parser.parse_args()
    args.capture_step_indices = parse_steps(args.capture_steps)
    if not math.isfinite(args.sparse_topk) or not 0.0 < args.sparse_topk <= 1.0:
        parser.error("--sparse-topk must be finite and in (0, 1]")
    return args


def make_bench_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        prompt="",
        negative_prompt=args.negative_prompt,
        prompt_json=args.prompt_json,
        negative_prompt_json=None,
        prompt_supplied=True,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        output_type=args.output_type,
        enable_safety_check=False,
        enable_resolution_template=False,
        enable_duration_template=False,
        disable_progress_bar=True,
        m1_fp8=True,
        m1_smoothquant=True,
        smoothquant_alpha=args.smoothquant_alpha,
        smoothquant_stats_path=args.smoothquant_stats_path,
        m1_compile=False,
        m1_compile_forward=False,
        attention_backend="sparge",
        sparse_topk=args.sparse_topk,
        sparse_tau=0.98,
        sparse_theta=0.6,
        taylorseer_interval=args.taylorseer_interval,
        taylorseer_fresh_threshold=None,
        taylorseer_force_scheduler=False,
        taylorseer_max_order=args.taylorseer_max_order,
        taylorseer_first_enhance=args.taylorseer_first_enhance,
        taylorseer_last_enhance=args.taylorseer_last_enhance,
        taylorseer_force_final_full=True,
        taylorseer_layer_indices=None,
        taylorseer_cache_und=True,
        taylorseer_stagger_layers=False,
        taylorseer_cache_max_gib=args.taylorseer_cache_max_gib,
        taylorseer_branches="both",
        taylorseer_delta_change_threshold=None,
        taylorseer_prediction_target="layer_delta",
        taylorseer_slope_scale=1.0,
    )


_DENSE_SPARGE_OP = None


def install_dense_m2_sparge_backend() -> dict[str, Any]:
    import torch

    global _DENSE_SPARGE_OP
    if _DENSE_SPARGE_OP is None:

        @torch.library.custom_op(
            "m3sens::sparge_fp8_dense_sm90",
            mutates_args=(),
            device_types="cuda",
            schema="(Tensor query, Tensor key, Tensor value, float? scale=None) -> Tensor",
        )
        def sparge_fp8_dense_sm90(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            scale: float | None = None,
        ) -> torch.Tensor:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90

            key, value = expand_kv_for_gqa(query, key, value)
            return sageattn_qk_int8_pv_fp8_cuda_sm90(
                query,
                key,
                value,
                tensor_layout="NHD",
                is_causal=False,
                qk_quant_gran="per_thread",
                sm_scale=scale,
                pv_accum_dtype="fp32+fp32",
                smooth_k=True,
                return_lse=False,
            )

        @sparge_fp8_dense_sm90.register_fake
        def _(query, key, value, scale=None):
            return query.new_empty(query.shape)

        _DENSE_SPARGE_OP = sparge_fp8_dense_sm90

    from diffusers.models import attention_dispatch
    from diffusers.models.attention_dispatch import AttentionBackendName

    registry = attention_dispatch._AttentionBackendRegistry

    def dense_sparge_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        _parallel_config=None,
    ):
        if attn_mask is not None:
            raise ValueError("attn_mask is not supported for m3 sensitivity dense sparge attention")
        if dropout_p != 0.0:
            raise ValueError("dropout is not supported for m3 sensitivity dense sparge attention")
        if return_lse:
            raise ValueError("return_lse is not supported for m3 sensitivity dense sparge attention")
        if _parallel_config is not None:
            raise ValueError("context-parallel is not supported for m3 sensitivity dense sparge attention")
        if is_causal:
            return attention_dispatch._native_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
                enable_gqa=enable_gqa,
                _parallel_config=None,
            )
        return _DENSE_SPARGE_OP(query, key, value, scale)

    registry._backends[AttentionBackendName.SPARGE] = dense_sparge_attention
    registry._constraints[AttentionBackendName.SPARGE] = []
    registry.set_active_backend(AttentionBackendName.SPARGE)
    return {
        "event": "m3_sensitivity_dense_m2_backend_installed",
        "backend": "sparge",
        "custom_op": "m3sens::sparge_fp8_dense_sm90",
        "gen_path": "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "causal_path": "native_attention",
    }


def expand_kv_for_gqa(query, key, value):
    num_heads_q = query.shape[2]
    num_heads_kv = key.shape[2]
    if num_heads_q == num_heads_kv:
        return key, value
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(f"num_heads_q ({num_heads_q}) must be a multiple of num_heads_kv ({num_heads_kv})")
    groups = num_heads_q // num_heads_kv
    return key.repeat_interleave(groups, dim=2), value.repeat_interleave(groups, dim=2)


def dense_sdpa_nhd(query, key, value, scale):
    import torch.nn.functional as F

    key, value = expand_kv_for_gqa(query, key, value)
    out = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def sparse_topk_nhd(query, key, value, scale, topk: float):
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

    key, value = expand_kv_for_gqa(query, key, value)
    old_kernel = os.environ.get("SPARGE_SM90_BLOCKSPARSE_KERNEL")
    old_topk = os.environ.get("SPARGE_TOPK_OVERRIDE")
    os.environ["SPARGE_SM90_BLOCKSPARSE_KERNEL"] = "no_pv"
    os.environ["SPARGE_TOPK_OVERRIDE"] = str(topk)
    try:
        return spas_sage2_attn_meansim_topk_cuda(
            query,
            key,
            value,
            topk=float(topk),
            is_causal=False,
            scale=scale,
            tensor_layout="NHD",
            output_dtype=query.dtype,
        ).contiguous()
    finally:
        if old_kernel is None:
            os.environ.pop("SPARGE_SM90_BLOCKSPARSE_KERNEL", None)
        else:
            os.environ["SPARGE_SM90_BLOCKSPARSE_KERNEL"] = old_kernel
        if old_topk is None:
            os.environ.pop("SPARGE_TOPK_OVERRIDE", None)
        else:
            os.environ["SPARGE_TOPK_OVERRIDE"] = old_topk


def compare_attention_outputs(query, key, value, scale, topk: float) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        dense = dense_sdpa_nhd(query, key, value, scale)
        sparse = sparse_topk_nhd(query, key, value, scale, topk)
        dense_f = dense.float()
        sparse_f = sparse.float()
        diff_f = sparse_f - dense_f
        rel_l1_heads = diff_f.abs().sum(dim=(0, 1, 3)) / dense_f.abs().sum(dim=(0, 1, 3)).clamp_min(1e-20)
        dense_heads = dense_f.permute(2, 0, 1, 3).reshape(dense.shape[2], -1)
        sparse_heads = sparse_f.permute(2, 0, 1, 3).reshape(sparse.shape[2], -1)
        cosine_heads = F.cosine_similarity(sparse_heads, dense_heads, dim=1, eps=1e-8)
        result = {
            "rel_l1": float(rel_l1_heads.mean().item()),
            "cosine": float(cosine_heads.mean().item()),
            "rel_l1_heads_min": float(rel_l1_heads.min().item()),
            "rel_l1_heads_max": float(rel_l1_heads.max().item()),
            "cosine_heads_min": float(cosine_heads.min().item()),
            "cosine_heads_max": float(cosine_heads.max().item()),
            "dense_l1": float(dense_f.abs().sum().item()),
            "diff_l1": float(diff_f.abs().sum().item()),
        }
        del dense, sparse, dense_f, sparse_f, diff_f, dense_heads, sparse_heads, rel_l1_heads, cosine_heads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def shape_record(tensor) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


def summarize_records(records: list[dict[str, Any]], args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    layer_indices = sorted({int(record["layer_idx"]) for record in records})
    per_layer: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        layer_records = [record for record in records if int(record["layer_idx"]) == layer_idx]
        per_step: list[dict[str, Any]] = []
        for step_index in args.capture_step_indices:
            step_records = [record for record in layer_records if int(record["step_index"]) == step_index]
            if not step_records:
                continue
            per_step.append(
                {
                    "step_index": int(step_index),
                    "step_number_1based": int(step_index + 1),
                    "timestep": step_records[0].get("timestep"),
                    "rel_l1": statistics.fmean(float(item["rel_l1"]) for item in step_records),
                    "cosine": statistics.fmean(float(item["cosine"]) for item in step_records),
                    "branches": [
                        {
                            key: item[key]
                            for key in (
                                "branch",
                                "rel_l1",
                                "cosine",
                                "rel_l1_heads_min",
                                "rel_l1_heads_max",
                                "cosine_heads_min",
                                "cosine_heads_max",
                            )
                        }
                        for item in sorted(step_records, key=lambda item: str(item.get("branch")))
                    ],
                }
            )
        per_layer.append(
            {
                "layer_idx": int(layer_idx),
                "rel_l1": statistics.fmean(float(item["rel_l1"]) for item in layer_records),
                "cosine": statistics.fmean(float(item["cosine"]) for item in layer_records),
                "sample_count": len(layer_records),
                "per_step": per_step,
            }
        )
    ranking = [
        {"rank": rank, "layer_idx": item["layer_idx"], "rel_l1": item["rel_l1"], "cosine": item["cosine"]}
        for rank, item in enumerate(sorted(per_layer, key=lambda item: (-item["rel_l1"], item["cosine"])), start=1)
    ]
    return {
        "metadata": {
            "prompt_id": args.prompt_id,
            "prompt_json": str(args.prompt_json),
            "num_inference_steps": int(args.num_inference_steps),
            "capture_step_indices_0based": [int(item) for item in args.capture_step_indices],
            "capture_step_numbers_1based": [int(item + 1) for item in args.capture_step_indices],
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "fps": float(args.fps),
            "guidance_scale": float(args.guidance_scale),
            "seed": int(args.seed),
            "dtype": args.dtype,
            "output_type": args.output_type,
            "taylorseer_config": config,
            "dense_generation_backend": "m3sens::sparge_fp8_dense_sm90",
            "metric_dense_reference": "torch.nn.functional.scaled_dot_product_attention",
            "metric_sparse_reference": "spas_sage2_attn_meansim_topk_cuda",
            "branches_averaged": sorted({str(record.get("branch")) for record in records}),
        },
        "topk_setting": {
            "mode": "crude_topk",
            "topk": float(args.sparse_topk),
            "drop_fraction": float(1.0 - args.sparse_topk),
            "sm90_blocksparse_kernel": "no_pv",
            "env": {
                "SPARGE_SM90_BLOCKSPARSE_KERNEL": "no_pv",
                "SPARGE_TOPK_OVERRIDE": str(args.sparse_topk),
            },
        },
        "per_layer": per_layer,
        "ranking": ranking,
        "raw_record_count": len(records),
        "raw_records": records,
    }


def main() -> None:
    args = parse_args()
    bench = load_sparse_benchmark_module()
    package_dir = bench.ensure_local_diffusers_source()

    import torch
    from diffusers import Cosmos3OmniTaylorSeerPipeline, Cosmos3OmniTaylorSeerTransformer

    bench_args = make_bench_namespace(args)
    dtype = bench.dtype_from_name(args.dtype)
    config = bench.build_taylorseer_config(bench_args)
    prompt = bench.load_prompt(args.prompt_json, "")
    backend_record = install_dense_m2_sparge_backend()
    print(json.dumps(backend_record, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "event": "m3_sensitivity_start",
                "prompt_id": args.prompt_id,
                "prompt_json": str(args.prompt_json),
                "num_inference_steps": args.num_inference_steps,
                "capture_step_indices_0based": list(args.capture_step_indices),
                "capture_step_numbers_1based": [item + 1 for item in args.capture_step_indices],
                "sparse_topk": args.sparse_topk,
                "diffusers_source": str(package_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    load_t0 = time.perf_counter()
    transformer = Cosmos3OmniTaylorSeerTransformer.from_pretrained(
        args.model,
        subfolder="transformer",
        **bench.transformer_load_kwargs(bench_args, dtype),
    )
    pipe = Cosmos3OmniTaylorSeerPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        **bench.base_pipeline_kwargs(bench_args, dtype),
    )
    bench.configure_pipeline(pipe, bench_args)
    pipe.enable_taylorseer(**bench.taylorseer_call_kwargs(config))

    stats = bench.load_m1_smoothquant_stats(args.smoothquant_stats_path)
    m1_smoothquant_record = bench.apply_m1_smoothquant(pipe.transformer, stats, args.smoothquant_alpha)
    m1_fp8_record = bench.apply_m1_fp8(pipe.transformer)
    bench.cuda_sync(args.device)
    print(
        json.dumps(
            {
                "event": "m3_sensitivity_loaded",
                "seconds": time.perf_counter() - load_t0,
                "m1_smoothquant": {
                    "alpha": m1_smoothquant_record["alpha"],
                    "group_count": m1_smoothquant_record["group_count"],
                },
                "m1_fp8": {
                    "hit_count": m1_fp8_record["hit_count"],
                    "expected_hit_count": m1_fp8_record["expected_hit_count"],
                },
                **bench.cuda_stats(args.device),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    records: list[dict[str, Any]] = []
    capture_steps = set(args.capture_step_indices)
    state = SimpleNamespace(layer_idx=None)

    import diffusers.models.transformers.transformer_cosmos3_taylorseer as ts_mod

    original_dispatch = ts_mod.dispatch_attention_fn

    def capturing_dispatch(query, key, value, *dispatch_args, **dispatch_kwargs):
        out = original_dispatch(query, key, value, *dispatch_args, **dispatch_kwargs)
        is_causal = bool(dispatch_kwargs.get("is_causal", False))
        layer_idx = state.layer_idx
        if (not is_causal) and layer_idx is not None:
            context = getattr(pipe.transformer, "_taylorseer_context", None)
            step_index = None if context is None else int(context.step_index)
            if step_index in capture_steps:
                branch = "unknown" if context is None else str(context.branch)
                timestep = None if context is None else int(context.timestep)
                metric = compare_attention_outputs(
                    query.detach(),
                    key.detach(),
                    value.detach(),
                    dispatch_kwargs.get("scale"),
                    args.sparse_topk,
                )
                record = {
                    "layer_idx": int(layer_idx),
                    "step_index": int(step_index),
                    "step_number_1based": int(step_index + 1),
                    "timestep": timestep,
                    "branch": branch,
                    "query": shape_record(query),
                    "key": shape_record(key),
                    "value": shape_record(value),
                    "scale": dispatch_kwargs.get("scale"),
                    **metric,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "event": "m3_sensitivity_record",
                            "layer_idx": record["layer_idx"],
                            "step_index": record["step_index"],
                            "branch": record["branch"],
                            "rel_l1": record["rel_l1"],
                            "cosine": record["cosine"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return out

    ts_mod.dispatch_attention_fn = capturing_dispatch
    originals: list[tuple[Any, Any]] = []
    for layer_idx, layer in enumerate(pipe.transformer.layers):
        original = layer.forward_with_gen_delta

        def wrapped_forward_with_gen_delta(und_seq, gen_seq, rotary_emb, _original=original, _layer_idx=layer_idx):
            previous = state.layer_idx
            state.layer_idx = _layer_idx
            try:
                return _original(und_seq, gen_seq, rotary_emb)
            finally:
                state.layer_idx = previous

        originals.append((layer, original))
        layer.forward_with_gen_delta = wrapped_forward_with_gen_delta

    run_t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                image=None,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                enable_sound=False,
                generator=bench.make_generator(bench_args),
                output_type=args.output_type,
                add_resolution_template=False,
                add_duration_template=False,
                enable_safety_check=False,
            )
        bench.cuda_sync(args.device)
        del result
    finally:
        ts_mod.dispatch_attention_fn = original_dispatch
        for layer, original in originals:
            layer.forward_with_gen_delta = original

    run_seconds = time.perf_counter() - run_t0
    expected_min_records = len(pipe.transformer.layers) * len(args.capture_step_indices)
    if len(records) < expected_min_records:
        raise RuntimeError(f"captured only {len(records)} records, expected at least {expected_min_records}")

    summary = summarize_records(records, args, config)
    summary["metadata"]["run_seconds"] = run_seconds
    summary["metadata"]["cuda"] = bench.cuda_stats(args.device)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "m3_sensitivity_saved",
                "path": str(args.output_json),
                "records": len(records),
                "run_seconds": run_seconds,
                "top5": summary["ranking"][:5],
                "bottom5": summary["ranking"][-5:],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
