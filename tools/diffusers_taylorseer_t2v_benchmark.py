#!/usr/bin/env python3
"""Cosmos3 Diffusers TaylorSeer text-to-video benchmark and visual QA runner.

Defaults match the public Nano T2V Diffusers benchmark profile: single-GPU
H200-class 720p generation with 189 frames at 24 FPS.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


QA_PROMPTS: list[dict[str, str]] = [
    {
        "label": "warehouse_robot",
        "prompt": "A mobile robot navigates a warehouse aisle and stops at a shelf.",
    },
    {
        "label": "fast_motion",
        "prompt": "A woman, wearing a vibrant orange sports outfit, is running briskly down a scenic trail. The trail winds through a lush forest, with sunlight filtering through the canopy of trees, casting dappled shadows on the ground. Her hair, styled in a neat ponytail, bounces slightly with each stride. In the background, a babbling brook adds a soothing sound to the peaceful surroundings.",
    },
    {
        "label": "fine_texture",
        "prompt": "A young boy, dressed in a blue t-shirt and jeans, is gently petting a golden retriever dog with soft, fluffy fur. The dog, named Max, is lying comfortably on a green grass lawn, surrounded by a variety of colorful flowers and trees. The boy's smile reflects his joy and love for Max, who appears to be enjoying the attention with its tail wagging happily. The sun is shining brightly in the sky, casting a warm glow over the scene, creating a peaceful and heartwarming atmosphere.",
    },
    {
        "label": "multi_object_relation",
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    {
        "label": "camera_motion_landscape",
        "prompt": "A girl, dressed in casual summer attire, stands on a green, lush island surrounded by a vast, deep blue sea. The island is teeming with vibrant flora and fauna, creating a picturesque natural setting. The camera captures the scene from a low angle, focusing on the girl as she looks out into the horizon. The shot then slowly pulls away, revealing the entire island and the sea in the background. This sequence is executed in a cinematic style, aiming for a realistic and immersive experience.",
    },
    {
        "label": "symbolic_structure",
        "prompt": "A time-lapse sequence captures the transformation of the iconic Eiffel Tower from daylight into the evening. The tower, standing tall and majestic in its original golden hue, gradually transitions into a silhouette against the twilight sky. As the sun sets, the city lights begin to flicker on, casting a warm glow over the Parisian landscape. The tower's intricate iron lattice structure becomes more defined, its shadow lengthening across the Champ de Mars. The background includes the Seine River and the Parisian rooftops, adding depth and context to the scene. As darkness falls, the Eiffel Tower is illuminated by its own lights, turning into a beacon of Paris, shimmering against the starry backdrop.",
    },
]
QA_PROMPTS_BY_LABEL = {entry["label"]: entry for entry in QA_PROMPTS}

QA_CONFIGS_FULL: list[dict[str, Any]] = [
    {"label": "i5_all", "interval": 5, "max_order": 1, "layers": None, "first_enhance": 1, "last_enhance": 1},
    {"label": "i4_all", "interval": 4, "max_order": 1, "layers": None, "first_enhance": 1, "last_enhance": 1},
    {"label": "i3_all", "interval": 3, "max_order": 1, "layers": None, "first_enhance": 1, "last_enhance": 1},
    {"label": "i5_l12_35", "interval": 5, "max_order": 1, "layers": "12-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i4_l12_35", "interval": 4, "max_order": 1, "layers": "12-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i3_l12_35", "interval": 3, "max_order": 1, "layers": "12-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i5_l18_35", "interval": 5, "max_order": 1, "layers": "18-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i4_l18_35", "interval": 4, "max_order": 1, "layers": "18-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i3_l18_35", "interval": 3, "max_order": 1, "layers": "18-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i5_l24_35", "interval": 5, "max_order": 1, "layers": "24-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i4_l24_35", "interval": 4, "max_order": 1, "layers": "24-35", "first_enhance": 1, "last_enhance": 1},
    {"label": "i3_l24_35", "interval": 3, "max_order": 1, "layers": "24-35", "first_enhance": 1, "last_enhance": 1},
]
QA_CONFIGS_SMOKE = [QA_CONFIGS_FULL[0]]
QA_CONFIG_PRIORITY = {entry["label"]: index for index, entry in enumerate(QA_CONFIGS_FULL)}

MANUAL_REVIEW_PENDING: dict[str, Any] = {
    "status": "pending",
    "semantic_drift_vs_baseline": None,
    "quality_drop_vs_baseline": None,
    "blur_or_noise_vs_baseline": None,
    "temporal_instability_vs_baseline": None,
    "notes": "",
}


def _flag_supplied(*names: str) -> bool:
    for item in sys.argv[1:]:
        for name in names:
            if item == name or item.startswith(f"{name}="):
                return True
    return False


def parse_cache_max_gib(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float or 'none'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive or 'none'")
    return parsed


def parse_layer_spec(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None

    layers: list[int] = []
    seen: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid empty layer selector in {spec!r}")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"invalid layer range {part!r}")
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid descending layer range {part!r}")
            values = range(start, end + 1)
        else:
            if not part.isdigit():
                raise ValueError(f"invalid layer index {part!r}")
            values = (int(part),)
        for value in values:
            if value not in seen:
                layers.append(value)
                seen.add(value)
    return tuple(layers)


def layer_label(layer_spec: str | None) -> str:
    if layer_spec is None or not layer_spec.strip():
        return "all"
    compact = layer_spec.replace(" ", "").replace(",", "_").replace("-", "_")
    return f"l{compact}"


def config_label(
    interval: int,
    layer_spec: str | None,
    *,
    max_order: int | None = None,
    first_enhance: int | None = None,
    last_enhance: int | None = None,
    force_final_full: bool | None = None,
    fresh_threshold: int | None = None,
    force_scheduler: bool | None = None,
    branches: str | None = None,
    delta_change_threshold: float | None = None,
    prediction_target: str | None = None,
    cache_und: bool | None = None,
    stagger_layers: bool | None = None,
    slope_scale: float | None = None,
) -> str:
    label = f"i{interval}_{layer_label(layer_spec)}"
    if max_order is not None:
        label = f"i{interval}_o{max_order}_{layer_label(layer_spec)}"
    if first_enhance is not None or last_enhance is not None:
        label += f"_w{first_enhance if first_enhance is not None else 0}_c{last_enhance if last_enhance is not None else 0}"
    if fresh_threshold is not None:
        label += f"_ft{fresh_threshold}"
    if force_scheduler:
        label += "_fs"
    if branches is not None and branches != "both":
        label += f"_b{branches}"
    if delta_change_threshold is not None:
        label += f"_thr{delta_change_threshold:g}"
    if prediction_target is not None and prediction_target != "layer_delta":
        label += f"_p{prediction_target.replace('_delta', '')}"
    if cache_und is False:
        label += "_exactund"
    if stagger_layers:
        label += "_stagger"
    if slope_scale is not None and slope_scale != 1.0:
        label += f"_s{slope_scale:g}"
    if force_final_full is False:
        label += "_nofinal"
    return label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument(
        "--prompt",
        default="A mobile robot navigates a warehouse aisle and stops at a shelf.",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--prompt-json", type=Path, help="JSON file to serialize as the prompt, matching the cookbook.")
    parser.add_argument("--negative-prompt-json", type=Path)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=189,
        help="Video frame count; default matches the public Cosmos3 T2V benchmark profile.",
    )
    parser.add_argument("--benchmark-profile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output-type", choices=("pil", "np", "pt", "latent"), default="pil")
    parser.add_argument("--save-video", type=Path, help="Optional path; export time is measured separately.")
    parser.add_argument("--enable-safety-check", action="store_true", help="Disabled by default for benchmark timing.")
    parser.add_argument("--enable-resolution-template", action="store_true")
    parser.add_argument("--enable-duration-template", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--profile-steps", action="store_true", help="Synchronize and log every denoising step.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Untimed full pipeline runs after model load.")
    parser.add_argument("--runs", type=int, default=1, help="Timed full pipeline runs to report after warmup.")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--summary-json", type=Path)

    parser.add_argument("--taylorseer-interval", type=int, default=5)
    parser.add_argument("--taylorseer-fresh-threshold", type=int)
    parser.add_argument("--taylorseer-force-scheduler", action="store_true")
    parser.add_argument("--taylorseer-max-order", type=int, default=1)
    parser.add_argument("--taylorseer-first-enhance", type=int, default=1)
    parser.add_argument("--taylorseer-last-enhance", type=int, default=1)
    force_final = parser.add_mutually_exclusive_group()
    force_final.add_argument("--taylorseer-force-final-full", dest="taylorseer_force_final_full", action="store_true")
    force_final.add_argument("--no-taylorseer-force-final-full", dest="taylorseer_force_final_full", action="store_false")
    parser.set_defaults(taylorseer_force_final_full=True)
    parser.add_argument(
        "--taylorseer-layer-indices",
        default=None,
        help="Comma-separated N or A-B inclusive ranges, e.g. 0,3,12-35. Omit for all layers.",
    )
    cache_und = parser.add_mutually_exclusive_group()
    cache_und.add_argument("--taylorseer-cache-und", dest="taylorseer_cache_und", action="store_true")
    cache_und.add_argument("--no-taylorseer-cache-und", dest="taylorseer_cache_und", action="store_false")
    parser.set_defaults(taylorseer_cache_und=True)
    parser.add_argument("--taylorseer-stagger-layers", action="store_true")
    parser.add_argument("--taylorseer-cache-max-gib", type=parse_cache_max_gib, default=64.0)
    parser.add_argument("--taylorseer-branches", choices=("both", "cond", "uncond"), default="both")
    parser.add_argument("--taylorseer-delta-change-threshold", type=float)
    parser.add_argument("--taylorseer-prediction-target", choices=("layer_delta", "attention_delta", "mlp_delta", "gen_component_delta", "und_cache"), default="layer_delta")
    parser.add_argument("--taylorseer-slope-scale", type=float, default=1.0)

    parser.add_argument("--qa-mode", choices=("single", "pair", "matrix-worker", "aggregate-only"), default="single")
    parser.add_argument("--qa-output-dir", type=Path)
    parser.add_argument("--qa-prompt-set", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--qa-prompt-label")
    parser.add_argument("--qa-frame-count", type=int, default=12)
    parser.add_argument("--qa-save-contact-sheets", dest="qa_save_contact_sheets", action="store_true", default=False)
    parser.add_argument("--qa-matrix-kind", choices=("smoke", "full"), default="full")
    parser.add_argument("--qa-worker-id", type=int)
    parser.add_argument("--qa-num-workers", type=int, default=16)
    parser.add_argument("--qa-review-json", type=Path)

    args = parser.parse_args()
    args.prompt_supplied = _flag_supplied("--prompt", "--prompt-json")

    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.taylorseer_interval < 1:
        parser.error("--taylorseer-interval must be at least 1")
    if args.taylorseer_fresh_threshold is not None and args.taylorseer_fresh_threshold < 1:
        parser.error("--taylorseer-fresh-threshold must be at least 1")
    if args.taylorseer_max_order not in {0, 1}:
        parser.error("--taylorseer-max-order must be 0 or 1")
    if args.taylorseer_delta_change_threshold is not None and args.taylorseer_delta_change_threshold <= 0:
        parser.error("--taylorseer-delta-change-threshold must be positive")
    if not math.isfinite(args.taylorseer_slope_scale) or args.taylorseer_slope_scale < 0:
        parser.error("--taylorseer-slope-scale must be finite and non-negative")
    if args.taylorseer_first_enhance < 0:
        parser.error("--taylorseer-first-enhance must be non-negative")
    if args.taylorseer_last_enhance < 0:
        parser.error("--taylorseer-last-enhance must be non-negative")
    if args.qa_frame_count < 1:
        parser.error("--qa-frame-count must be at least 1")
    if args.qa_prompt_label is not None and args.qa_prompt_label not in QA_PROMPTS_BY_LABEL:
        parser.error(f"unknown --qa-prompt-label {args.qa_prompt_label!r}")
    if args.qa_mode in {"pair", "matrix-worker", "aggregate-only"} and args.qa_output_dir is None:
        parser.error(f"--qa-output-dir is required for --qa-mode {args.qa_mode}")
    if args.qa_mode in {"pair", "matrix-worker"} and args.output_type != "pil":
        raise ValueError("visual QA modes require --output-type pil")
    if args.qa_mode == "matrix-worker":
        if args.qa_worker_id is None:
            parser.error("--qa-worker-id is required for --qa-mode matrix-worker")
        if args.qa_num_workers < 1:
            parser.error("--qa-num-workers must be at least 1")
        if not 0 <= args.qa_worker_id < args.qa_num_workers:
            parser.error("--qa-worker-id must be in range [0, qa-num-workers)")
    try:
        args.parsed_taylorseer_layer_indices = parse_layer_spec(args.taylorseer_layer_indices)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def load_prompt(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    with path.open("r", encoding="utf-8") as handle:
        return json.dumps(json.load(handle))


def ensure_local_diffusers_source() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    package_dir = src_dir / "diffusers"
    if not package_dir.is_dir():
        raise RuntimeError(f"Local Diffusers source not found: {package_dir}")
    src_text = str(src_dir)
    if sys.path[0] != src_text:
        sys.path.insert(0, src_text)
    return package_dir.resolve()


def assert_local_diffusers(module_file: str | None, package_dir: Path) -> str:
    if module_file is None:
        raise RuntimeError("Imported diffusers has no __file__; local source check failed")
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(package_dir)
    except ValueError as exc:
        raise RuntimeError(f"Imported external diffusers from {resolved}, expected under {package_dir}") from exc
    return str(resolved)


def dtype_from_name(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def cuda_sync(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_stats(device: str) -> dict[str, float | str | bool | None]:
    import torch

    stats: dict[str, float | str | bool | None] = {"cuda_available": torch.cuda.is_available()}
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return stats

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    stats.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_total_gib": total_bytes / 1024**3,
            "memory_free_gib": free_bytes / 1024**3,
            "memory_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    )
    return stats


def clear_cuda(device: str) -> None:
    import torch

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_steps(step_end_offsets: list[float], total_seconds: float) -> dict[str, Any]:
    if not step_end_offsets:
        return {}

    deltas = [step_end_offsets[0], *[b - a for a, b in zip(step_end_offsets, step_end_offsets[1:])]]
    steady = deltas[1:] if len(deltas) > 1 else deltas
    return {
        "steps_observed": len(step_end_offsets),
        "first_step_end_seconds": step_end_offsets[0],
        "last_step_end_seconds": step_end_offsets[-1],
        "post_denoise_tail_seconds": total_seconds - step_end_offsets[-1],
        "step_delta_first_seconds": deltas[0],
        "step_delta_avg_excluding_first_seconds": statistics.fmean(steady),
        "step_delta_median_excluding_first_seconds": statistics.median(steady),
        "step_delta_min_excluding_first_seconds": min(steady),
        "step_delta_max_excluding_first_seconds": max(steady),
    }


def maybe_h200_reference(args: argparse.Namespace, num_frames: int, measured_seconds: float) -> dict[str, Any] | None:
    is_nano_720p = (
        args.model == "nvidia/Cosmos3-Nano"
        and args.height == 720
        and args.width == 1280
        and args.num_inference_steps == 35
        and math.isclose(args.guidance_scale, 6.0)
        and args.output_type == "pil"
    )
    if not is_nano_720p:
        return None

    benchmark_seconds = 239.60
    scaled_seconds = benchmark_seconds * (num_frames / 189.0)
    return {
        "reference": "inference_benchmarks.md Nano T2V Diffusers H200 141GB HBM3 720p/1",
        "benchmark_frames": 189,
        "benchmark_seconds": benchmark_seconds,
        "frame_scaled_seconds": scaled_seconds,
        "frame_scaled_is_observed_benchmark": num_frames == 189,
        "measured_over_frame_scaled": measured_seconds / scaled_seconds if scaled_seconds > 0 else None,
    }


def make_generator(args: argparse.Namespace):
    import torch

    if args.device.startswith("cuda"):
        return torch.Generator(device=args.device).manual_seed(args.seed)
    return torch.Generator().manual_seed(args.seed)


def video_shape_of(result, output_type: str) -> Any:
    if output_type in {"latent", "pt", "np"}:
        return list(result.video.shape)
    return [len(result.video), getattr(result.video[0], "height", None), getattr(result.video[0], "width", None)]


def taylorseer_call_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "interval": config["interval"],
        "fresh_threshold": config["fresh_threshold"],
        "force_scheduler": config["force_scheduler"],
        "max_order": config["max_order"],
        "first_enhance": config["first_enhance"],
        "last_enhance": config["last_enhance"],
        "force_final_full": config["force_final_full"],
        "layer_indices": tuple(config["layer_indices"]) if config["layer_indices"] is not None else None,
        "cache_und": config["cache_und"],
        "stagger_layers": config["stagger_layers"],
        "slope_scale": config["slope_scale"],
        "cache_max_gib": config["cache_max_gib"],
        "branches": config["branches"],
        "delta_change_threshold": config["delta_change_threshold"],
        "prediction_target": config["prediction_target"],
    }


def build_taylorseer_config(
    args: argparse.Namespace,
    *,
    interval: int | None = None,
    layer_spec_text: str | None = None,
    label: str | None = None,
    first_enhance: int | None = None,
    last_enhance: int | None = None,
    max_order: int | None = None,
    fresh_threshold: int | None = None,
    force_scheduler: bool | None = None,
    branches: str | None = None,
    delta_change_threshold: float | None = None,
    prediction_target: str | None = None,
    cache_und: bool | None = None,
    stagger_layers: bool | None = None,
    slope_scale: float | None = None,
) -> dict[str, Any]:
    actual_interval = args.taylorseer_interval if interval is None else interval
    actual_layer_spec = args.taylorseer_layer_indices if layer_spec_text is None else layer_spec_text
    actual_first_enhance = args.taylorseer_first_enhance if first_enhance is None else first_enhance
    actual_last_enhance = args.taylorseer_last_enhance if last_enhance is None else last_enhance
    actual_max_order = args.taylorseer_max_order if max_order is None else max_order
    actual_fresh_threshold = args.taylorseer_fresh_threshold if fresh_threshold is None else fresh_threshold
    actual_force_scheduler = args.taylorseer_force_scheduler if force_scheduler is None else force_scheduler
    actual_branches = args.taylorseer_branches if branches is None else branches
    actual_delta_change_threshold = (
        args.taylorseer_delta_change_threshold if delta_change_threshold is None else delta_change_threshold
    )
    actual_prediction_target = args.taylorseer_prediction_target if prediction_target is None else prediction_target
    actual_cache_und = args.taylorseer_cache_und if cache_und is None else cache_und
    actual_stagger_layers = args.taylorseer_stagger_layers if stagger_layers is None else stagger_layers
    actual_slope_scale = args.taylorseer_slope_scale if slope_scale is None else slope_scale
    layer_indices = parse_layer_spec(actual_layer_spec)
    actual_label = label
    if actual_label is not None and actual_cache_und is False and "_exactund" not in actual_label:
        actual_label += "_exactund"
    if actual_label is not None and actual_stagger_layers and "_stagger" not in actual_label:
        actual_label += "_stagger"
    if actual_label is not None and actual_slope_scale != 1.0 and f"_s{actual_slope_scale:g}" not in actual_label:
        actual_label += f"_s{actual_slope_scale:g}"
    if actual_label is not None and actual_fresh_threshold is not None and f"_ft{actual_fresh_threshold}" not in actual_label:
        actual_label += f"_ft{actual_fresh_threshold}"
    if actual_label is not None and actual_force_scheduler and "_fs" not in actual_label:
        actual_label += "_fs"
    return {
        "enabled": True,
        "label": actual_label
        or config_label(
            actual_interval,
            actual_layer_spec,
            max_order=actual_max_order,
            first_enhance=actual_first_enhance,
            last_enhance=actual_last_enhance,
            force_final_full=args.taylorseer_force_final_full,
            fresh_threshold=actual_fresh_threshold,
            force_scheduler=actual_force_scheduler,
            branches=actual_branches,
            delta_change_threshold=actual_delta_change_threshold,
            prediction_target=actual_prediction_target,
            cache_und=actual_cache_und,
            stagger_layers=actual_stagger_layers,
            slope_scale=actual_slope_scale,
        ),
        "interval": actual_interval,
        "fresh_threshold": actual_fresh_threshold,
        "force_scheduler": actual_force_scheduler,
        "max_order": actual_max_order,
        "first_enhance": actual_first_enhance,
        "last_enhance": actual_last_enhance,
        "force_final_full": args.taylorseer_force_final_full,
        "branches": actual_branches,
        "delta_change_threshold": actual_delta_change_threshold,
        "prediction_target": actual_prediction_target,
        "cache_und": actual_cache_und,
        "stagger_layers": actual_stagger_layers,
        "slope_scale": actual_slope_scale,
        "layer_spec": actual_layer_spec,
        "layer_indices": list(layer_indices) if layer_indices is not None else None,
        "cache_max_gib": args.taylorseer_cache_max_gib,
    }


def select_prompts(args: argparse.Namespace, *, matrix_kind: str | None = None) -> list[dict[str, str]]:
    if args.prompt_supplied:
        return [{"label": "custom_prompt", "prompt": load_prompt(args.prompt_json, args.prompt)}]
    if args.qa_prompt_label is not None:
        return [QA_PROMPTS_BY_LABEL[args.qa_prompt_label]]
    if matrix_kind is not None:
        return [QA_PROMPTS[0]] if matrix_kind == "smoke" else list(QA_PROMPTS)
    if args.qa_prompt_set == "smoke":
        return [QA_PROMPTS[0]]
    return list(QA_PROMPTS)


def base_pipeline_kwargs(args: argparse.Namespace, dtype: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "enable_safety_checker": args.enable_safety_check,
    }
    if args.device.startswith("cuda"):
        kwargs["device_map"] = args.device
    return kwargs


def transformer_load_kwargs(args: argparse.Namespace, dtype: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.device.startswith("cuda"):
        kwargs["device_map"] = args.device
    return kwargs


def configure_pipeline(pipe, args: argparse.Namespace):
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    return pipe


def load_baseline_pipeline(args: argparse.Namespace, dtype: Any):
    from diffusers import Cosmos3OmniPipeline

    t0 = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(args.model, **base_pipeline_kwargs(args, dtype))
    configure_pipeline(pipe, args)
    cuda_sync(args.device)
    seconds = time.perf_counter() - t0
    print(json.dumps({"event": "loaded", "pipeline": "baseline", "seconds": seconds, **cuda_stats(args.device)}), flush=True)
    return pipe, seconds


def load_taylorseer_pipeline(args: argparse.Namespace, dtype: Any, config: dict[str, Any]):
    from diffusers import Cosmos3OmniTaylorSeerPipeline, Cosmos3OmniTaylorSeerTransformer

    t0 = time.perf_counter()
    transformer = Cosmos3OmniTaylorSeerTransformer.from_pretrained(
        args.model,
        subfolder="transformer",
        **transformer_load_kwargs(args, dtype),
    )
    pipe = Cosmos3OmniTaylorSeerPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        **base_pipeline_kwargs(args, dtype),
    )
    configure_pipeline(pipe, args)
    pipe.enable_taylorseer(**taylorseer_call_kwargs(config))
    cuda_sync(args.device)
    seconds = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "event": "loaded",
                "pipeline": "taylorseer",
                "seconds": seconds,
                "taylorseer_config": config,
                **cuda_stats(args.device),
            }
        ),
        flush=True,
    )
    return pipe, seconds


def run_pipeline_call(
    pipe,
    args: argparse.Namespace,
    *,
    prompt: str,
    negative_prompt: str,
    output_type: str,
    pipeline_kind: str,
    run_kind: str,
    run_index: int,
    profile_steps: bool,
):
    import torch

    step_end_offsets: list[float] = []
    call_start = 0.0

    def on_step_end(_pipe, step_index: int, timestep, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        cuda_sync(args.device)
        offset = time.perf_counter() - call_start
        step_end_offsets.append(offset)
        if args.log_every > 0 and ((step_index + 1) % args.log_every == 0 or step_index == 0):
            print(
                json.dumps(
                    {
                        "event": "step_end",
                        "pipeline": pipeline_kind,
                        "run_kind": run_kind,
                        "run_index": run_index,
                        "step": step_index + 1,
                        "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
                        "elapsed_seconds": offset,
                    }
                ),
                flush=True,
            )
        return callback_kwargs

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cuda_sync(args.device)
    call_start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=None,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        enable_sound=False,
        generator=make_generator(args),
        output_type=output_type,
        add_resolution_template=args.enable_resolution_template,
        add_duration_template=args.enable_duration_template,
        enable_safety_check=args.enable_safety_check,
        callback_on_step_end=on_step_end if profile_steps else None,
    )
    cuda_sync(args.device)
    seconds = time.perf_counter() - call_start
    record = {
        "pipeline": pipeline_kind,
        "kind": run_kind,
        "index": run_index,
        "seconds": seconds,
        "video_shape": video_shape_of(result, output_type),
        "steps": summarize_steps(step_end_offsets, seconds),
        "cuda": cuda_stats(args.device),
    }
    print(
        json.dumps(
            {
                "event": "run_complete",
                "pipeline": pipeline_kind,
                "run_kind": run_kind,
                "run_index": run_index,
                "seconds": seconds,
                "video_shape": record["video_shape"],
                "max_memory_allocated_gib": record["cuda"].get("max_memory_allocated_gib"),
            }
        ),
        flush=True,
    )
    return result, record


def run_pipeline_runs(
    pipe,
    args: argparse.Namespace,
    *,
    prompt: str,
    negative_prompt: str,
    output_type: str,
    pipeline_kind: str,
) -> tuple[Any, dict[str, Any]]:
    warmup_records = []
    for run_index in range(args.warmup_runs):
        warmup_result, warmup_record = run_pipeline_call(
            pipe,
            args,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_type=output_type,
            pipeline_kind=pipeline_kind,
            run_kind="warmup",
            run_index=run_index,
            profile_steps=False,
        )
        warmup_records.append(warmup_record)
        del warmup_result
        gc.collect()

    measured_records = []
    result = None
    for run_index in range(args.runs):
        if result is not None:
            del result
            gc.collect()
        result, record = run_pipeline_call(
            pipe,
            args,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_type=output_type,
            pipeline_kind=pipeline_kind,
            run_kind="measured",
            run_index=run_index,
            profile_steps=args.profile_steps,
        )
        measured_records.append(record)

    measured_seconds = [record["seconds"] for record in measured_records]
    timings = {
        "pipeline_call_seconds": statistics.fmean(measured_seconds),
        "pipeline_call_seconds_median": statistics.median(measured_seconds),
        "pipeline_call_seconds_min": min(measured_seconds),
        "pipeline_call_seconds_max": max(measured_seconds),
    }
    run_summary = {
        "timings": timings,
        "warmup_runs": warmup_records,
        "measured_runs": measured_records,
        "steps": measured_records[0]["steps"],
        "cuda": measured_records[-1]["cuda"],
        "video_shape": measured_records[-1]["video_shape"],
    }
    return result, run_summary


def export_video(video: Any, path: Path, fps: float) -> float:
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    export_to_video(video, str(path), fps=int(fps), macro_block_size=1)
    return time.perf_counter() - start


def read_video_frames(path: Path) -> list[Any]:
    import imageio
    from PIL import Image

    frames = []
    reader = imageio.get_reader(str(path))
    try:
        for frame in reader:
            frames.append(Image.fromarray(frame).convert("RGB"))
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def resize_pair_for_sheet(left, right, *, max_pair_width: int = 640):
    total_width = left.width + right.width
    if total_width <= max_pair_width:
        return left.copy(), right.copy()
    scale = max_pair_width / total_width
    new_left = left.resize((max(1, int(left.width * scale)), max(1, int(left.height * scale))))
    new_right = right.resize((max(1, int(right.width * scale)), max(1, int(right.height * scale))))
    return new_left, new_right


def pair_frame(left, right, *, shrink_for_sheet: bool):
    from PIL import Image, ImageDraw, ImageFont

    left = left.convert("RGB")
    right = right.convert("RGB")
    if shrink_for_sheet:
        left, right = resize_pair_for_sheet(left, right)
    width = left.width + right.width
    height = max(left.height, right.height)
    panel = Image.new("RGB", (width, height), "black")
    panel.paste(left, (0, 0))
    panel.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.rectangle((4, 4, 80, 20), fill="black")
    draw.text((8, 6), "baseline", fill="white", font=font)
    draw.rectangle((left.width + 4, 4, left.width + 92, 20), fill="black")
    draw.text((left.width + 8, 6), "taylorseer", fill="white", font=font)
    return panel


def save_contact_sheet(images: list[Any], path: Path, *, columns: int) -> None:
    from PIL import Image

    if not images:
        raise RuntimeError(f"No images for contact sheet {path}")
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "black")
    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def fraction_indices(frame_count: int, fractions: tuple[float, ...]) -> list[int]:
    if frame_count < 1:
        return []
    return [min(frame_count - 1, max(0, int(round(fraction * (frame_count - 1))))) for fraction in fractions]


def motion_indices(frame_count: int, *, center_fraction: float, count: int) -> list[int]:
    if frame_count < 1:
        return []
    actual_count = min(count, frame_count)
    center = int(round(center_fraction * (frame_count - 1)))
    start = center - actual_count // 2
    start = max(0, min(start, frame_count - actual_count))
    return list(range(start, start + actual_count))

def evenly_spaced_indices(frame_count: int, count: int) -> list[int]:
    if frame_count < 1 or count < 1:
        return []
    if count == 1:
        return [frame_count // 2]
    if count >= frame_count:
        return list(range(frame_count))
    indices = [int(round(i * (frame_count - 1) / (count - 1))) for i in range(count)]
    result: list[int] = []
    seen: set[int] = set()
    for index in indices:
        index = min(frame_count - 1, max(0, index))
        if index not in seen:
            result.append(index)
            seen.add(index)
    return result


def compute_frame_metrics(baseline_frame: Any, taylorseer_frame: Any) -> dict[str, float]:
    import numpy as np

    baseline = np.asarray(baseline_frame.convert("RGB"), dtype=np.float32)
    taylorseer = np.asarray(taylorseer_frame.convert("RGB"), dtype=np.float32)
    if baseline.shape != taylorseer.shape:
        raise RuntimeError(f"Frame shape mismatch for metrics: baseline={baseline.shape}, taylorseer={taylorseer.shape}")
    diff = baseline - taylorseer
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = 100.0 if mse == 0 else float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))

    baseline_mean = float(np.mean(baseline))
    taylorseer_mean = float(np.mean(taylorseer))
    baseline_var = float(np.var(baseline))
    taylorseer_var = float(np.var(taylorseer))
    covariance = float(np.mean((baseline - baseline_mean) * (taylorseer - taylorseer_mean)))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    global_ssim = ((2 * baseline_mean * taylorseer_mean + c1) * (2 * covariance + c2)) / (
        (baseline_mean**2 + taylorseer_mean**2 + c1) * (baseline_var + taylorseer_var + c2)
    )

    return {
        "mae": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "mse": mse,
        "rmse": rmse,
        "psnr_db": psnr,
        "global_ssim": float(global_ssim),
    }


def summarize_frame_metrics(frame_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric_keys = ("mae", "max_abs_diff", "mse", "rmse", "psnr_db", "global_ssim")
    summary: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [float(item[key]) for item in frame_metrics if key in item]
        if not values:
            continue
        summary[key] = {
            "min": min(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return summary


def generate_visual_artifacts(args: argparse.Namespace, *, job_dir: Path, baseline_video: Path, taylorseer_video: Path) -> dict[str, Any]:
    from diffusers.utils import export_to_video

    baseline_frames = read_video_frames(baseline_video)
    taylorseer_frames = read_video_frames(taylorseer_video)
    baseline_frame_count = len(baseline_frames)
    taylorseer_frame_count = len(taylorseer_frames)
    if baseline_frame_count != taylorseer_frame_count:
        raise RuntimeError(
            "Baseline/TaylorSeer frame count mismatch: "
            f"baseline={baseline_frame_count}, taylorseer={taylorseer_frame_count}"
        )
    if baseline_frame_count != args.num_frames:
        raise RuntimeError(f"Decoded frame count {baseline_frame_count} does not match requested {args.num_frames}")
    frame_count = baseline_frame_count

    side_by_side_path = job_dir / "side_by_side.mp4"
    side_frames = [pair_frame(baseline_frames[i], taylorseer_frames[i], shrink_for_sheet=False) for i in range(frame_count)]
    export_to_video(side_frames, str(side_by_side_path), fps=int(args.fps), macro_block_size=1)
    del side_frames

    frames_dir = job_dir / "frames"
    pair_dir = frames_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = frames_dir / "frame_metrics.json"
    global_contact_sheet = frames_dir / "global_contact_sheet.png"
    motion_early = frames_dir / "motion_strip_early.png"
    motion_mid = frames_dir / "motion_strip_mid.png"
    motion_late = frames_dir / "motion_strip_late.png"

    frame_metrics: list[dict[str, Any]] = []
    pair_images: list[str] = []
    for index in evenly_spaced_indices(frame_count, args.qa_frame_count):
        pair_path = pair_dir / f"frame_{index:06d}.png"
        pair_frame(baseline_frames[index], taylorseer_frames[index], shrink_for_sheet=False).save(pair_path)
        metrics = compute_frame_metrics(baseline_frames[index], taylorseer_frames[index])
        metrics.update({"frame_index": index, "time_seconds": index / args.fps if args.fps else None})
        frame_metrics.append(metrics)
        pair_images.append(str(pair_path))

    metrics_payload = {
        "frame_count": frame_count,
        "sampled_frame_count": len(frame_metrics),
        "sampled_frame_indices": [item["frame_index"] for item in frame_metrics],
        "frames": frame_metrics,
        "summary": summarize_frame_metrics(frame_metrics),
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.qa_save_contact_sheets:
        global_panels = [
            pair_frame(baseline_frames[i], taylorseer_frames[i], shrink_for_sheet=True)
            for i in fraction_indices(frame_count, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0))
        ]
        save_contact_sheet(global_panels, global_contact_sheet, columns=len(global_panels))

        for path, center_fraction in (
            (motion_early, 0.15),
            (motion_mid, 0.50),
            (motion_late, 0.85),
        ):
            indices = motion_indices(frame_count, center_fraction=center_fraction, count=args.qa_frame_count)
            panels = [pair_frame(baseline_frames[i], taylorseer_frames[i], shrink_for_sheet=True) for i in indices]
            save_contact_sheet(panels, path, columns=len(panels))

    contact_sheet_paths = {
        "global_contact_sheet": str(global_contact_sheet) if args.qa_save_contact_sheets else None,
        "motion_strip_early": str(motion_early) if args.qa_save_contact_sheets else None,
        "motion_strip_mid": str(motion_mid) if args.qa_save_contact_sheets else None,
        "motion_strip_late": str(motion_late) if args.qa_save_contact_sheets else None,
    }

    return {
        "paths": {
            "side_by_side": str(side_by_side_path),
            "frame_pairs_dir": str(pair_dir),
            "frame_pair_images": pair_images,
            "frame_metrics_json": str(metrics_json),
            **contact_sheet_paths,
        },
        "frame_metrics": metrics_payload,
    }


def write_review_html(job_dir: Path, summary: dict[str, Any]) -> str:
    paths = summary["paths"]

    def rel(path: str | None) -> str:
        if path is None:
            return ""
        try:
            return html.escape(os.path.relpath(path, job_dir))
        except ValueError:
            return html.escape(path)

    prompt_text = html.escape(summary["prompt"])
    config_text = html.escape(json.dumps(summary["taylorseer_config"], sort_keys=True))
    seed_text = html.escape(str(summary["seed"]))
    status_text = html.escape(summary["status"])
    pair_images = paths.get("frame_pair_images") or []
    pair_image_html = "\n".join(
        f"  <figure><img src=\"{rel(path)}\" alt=\"baseline vs taylorseer frame pair\"></figure>"
        for path in pair_images
    )
    metrics_payload = summary.get("frame_metrics") or {}
    metrics_text = html.escape(json.dumps(metrics_payload.get("summary", {}), indent=2, sort_keys=True))
    body = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>Cosmos3 TaylorSeer QA - {html.escape(summary['config_label'])} / {html.escape(summary['prompt_label'])}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    video, img {{ display: block; max-width: 100%; margin: 16px 0; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Cosmos3 TaylorSeer QA</h1>
  <p><strong>Status:</strong> {status_text}</p>
  <p><strong>Prompt label:</strong> {html.escape(summary['prompt_label'])}</p>
  <p><strong>Prompt:</strong> {prompt_text}</p>
  <p><strong>Seed:</strong> {seed_text}</p>
  <p><strong>Config:</strong> <code>{config_text}</code></p>
  <h2>Side-by-side video</h2>
  <video controls src=\"{rel(paths.get('side_by_side'))}\"></video>
  <h2>Frame-pair review images</h2>
  <p>Each image contains exactly one baseline frame on the left and the matching TaylorSeer frame on the right.</p>
{pair_image_html}
  <h2>Reference frame metrics</h2>
  <p><a href="{rel(paths.get('frame_metrics_json'))}">frame_metrics.json</a></p>
  <pre>{metrics_text}</pre>
</body>
</html>
"""
    path = job_dir / "review.html"
    path.write_text(body, encoding="utf-8")
    return str(path)


def is_memory_skip(exc: BaseException) -> bool:
    return "TaylorSeer cache estimate exceeds limit" in str(exc)


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or "cublas_status_alloc_failed" in text


def scheduler_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {"class": "UniPCMultistepScheduler", "flow_shift": args.flow_shift}


def _stats_count(stats: dict[str, Any], key: str) -> int:
    values = stats.get(key, {})
    if not isinstance(values, dict):
        return 0
    return sum(int(value) for value in values.values())


def validate_taylorseer_stats(config: dict[str, Any], stats: dict[str, Any], *, num_steps: int) -> None:
    errors: list[str] = []
    for key in (
        "interval",
        "fresh_threshold",
        "force_scheduler",
        "max_order",
        "first_enhance",
        "last_enhance",
        "force_final_full",
        "branches",
        "delta_change_threshold",
        "prediction_target",
        "cache_und",
        "stagger_layers",
        "slope_scale",
    ):
        if stats.get(key) != config.get(key):
            errors.append(f"{key}: requested {config.get(key)!r}, got {stats.get(key)!r}")
    if not stats.get("enabled"):
        errors.append("stats report TaylorSeer disabled")
    expected_layers = config.get("layer_indices")
    actual_layers = stats.get("selected_layers")
    if expected_layers is not None and actual_layers != expected_layers:
        errors.append(f"selected_layers: requested {expected_layers!r}, got {actual_layers!r}")
    elif expected_layers is None and not actual_layers:
        errors.append("selected_layers is empty")

    predicted_calls = _stats_count(stats, "predicted_layer_calls_by_branch")
    if config["prediction_target"] != "und_cache" and config["interval"] == 1 and predicted_calls != 0:
        errors.append(f"interval=1 must not predict any layer calls, got {predicted_calls}")
    cooldown_steps = max(1, int(config["last_enhance"])) if config.get("force_final_full") else 0
    prediction_window = max(0, num_steps - int(config["first_enhance"]) - cooldown_steps)
    if config["prediction_target"] == "und_cache":
        if predicted_calls <= 0:
            errors.append("TaylorSeer UND cache produced no cached layer calls")
    elif config["interval"] > 1 and prediction_window > 1 and predicted_calls <= 0:
        errors.append("TaylorSeer produced no predicted layer calls despite an accelerated schedule")
    if errors:
        raise RuntimeError("TaylorSeer stats validation failed: " + "; ".join(errors))


def aggregate_consistency_key(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": entry.get("model"),
        "height": entry.get("height"),
        "width": entry.get("width"),
        "num_frames": entry.get("num_frames"),
        "fps": entry.get("fps"),
        "num_inference_steps": entry.get("num_inference_steps"),
        "guidance_scale": entry.get("guidance_scale"),
        "dtype": entry.get("dtype"),
        "scheduler": entry.get("scheduler"),
        "seed": entry.get("seed"),
        "negative_prompt": entry.get("negative_prompt"),
        "add_resolution_template": entry.get("add_resolution_template"),
        "add_duration_template": entry.get("add_duration_template"),
        "enable_safety_check": entry.get("enable_safety_check"),
        "taylorseer_config": entry.get("taylorseer_config"),
    }


def pair_summary_base(
    args: argparse.Namespace,
    *,
    prompt_entry: dict[str, str],
    config: dict[str, Any],
    job_dir: Path,
) -> dict[str, Any]:
    paths = {
        "baseline": str(job_dir / "baseline.mp4"),
        "taylorseer": str(job_dir / "taylorseer.mp4"),
        "side_by_side": str(job_dir / "side_by_side.mp4"),
        "review_html": str(job_dir / "review.html"),
        "frame_pairs_dir": str(job_dir / "frames" / "pairs"),
        "frame_pair_images": [],
        "frame_metrics_json": str(job_dir / "frames" / "frame_metrics.json"),
        "global_contact_sheet": None,
        "motion_strip_early": None,
        "motion_strip_mid": None,
        "motion_strip_late": None,
        "summary_json": str(job_dir / "summary.json"),
    }
    return {
        "prompt_label": prompt_entry["label"],
        "prompt": prompt_entry["prompt"],
        "negative_prompt": None,
        "seed": args.seed,
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "add_resolution_template": args.enable_resolution_template,
        "add_duration_template": args.enable_duration_template,
        "enable_safety_check": args.enable_safety_check,
        "scheduler": scheduler_summary(args),
        "config_label": config["label"],
        "taylorseer_config": config,
        "taylorseer_stats": None,
        "baseline_seconds": None,
        "taylorseer_seconds": None,
        "speedup": None,
        "baseline_max_cuda_memory_gib": None,
        "taylorseer_max_cuda_memory_gib": None,
        "baseline": None,
        "taylorseer": None,
        "status": "error",
        "error": None,
        "frame_metrics": None,
        "paths": paths,
        "manual_review": dict(MANUAL_REVIEW_PENDING),
    }


def write_summary_json(summary: dict[str, Any]) -> None:
    path = Path(summary["paths"]["summary_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_manual_review(summary: dict[str, Any], review_data: Any) -> None:
    if not isinstance(review_data, dict):
        return
    key = f"{summary.get('config_label')}/{summary.get('prompt_label')}"
    candidates = [
        review_data.get(key),
        review_data.get(summary.get("config_label"), {}).get(summary.get("prompt_label"))
        if isinstance(review_data.get(summary.get("config_label")), dict)
        else None,
        review_data.get(summary.get("prompt_label")),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            merged = dict(MANUAL_REVIEW_PENDING)
            merged.update(candidate.get("manual_review", candidate))
            summary["manual_review"] = merged
            return


def load_review_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))



def run_pair_job(
    args: argparse.Namespace,
    *,
    prompt_entry: dict[str, str],
    config: dict[str, Any],
    review_data: Any,
    diffusers_source: str,
) -> dict[str, Any]:
    dtype = dtype_from_name(args.dtype)
    negative_prompt = load_prompt(args.negative_prompt_json, args.negative_prompt)
    job_dir = args.qa_output_dir / config["label"] / prompt_entry["label"]
    job_dir.mkdir(parents=True, exist_ok=True)
    summary = pair_summary_base(args, prompt_entry=prompt_entry, config=config, job_dir=job_dir)
    summary["negative_prompt"] = negative_prompt
    apply_manual_review(summary, review_data)

    print(
        json.dumps(
            {
                "event": "qa_job_start",
                "qa_mode": args.qa_mode,
                "config_label": config["label"],
                "prompt_label": prompt_entry["label"],
                "diffusers_source": diffusers_source,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )

    baseline_pipe = None
    taylorseer_pipe = None
    try:
        baseline_pipe, baseline_load_seconds = load_baseline_pipeline(args, dtype)
        baseline_result, baseline_run = run_pipeline_runs(
            baseline_pipe,
            args,
            prompt=prompt_entry["prompt"],
            negative_prompt=negative_prompt,
            output_type="pil",
            pipeline_kind="baseline",
        )
        baseline_export_seconds = export_video(baseline_result.video, Path(summary["paths"]["baseline"]), args.fps)
        summary["baseline"] = {
            "load_seconds": baseline_load_seconds,
            "export_seconds": baseline_export_seconds,
            **baseline_run,
        }
        summary["baseline_seconds"] = baseline_run["timings"]["pipeline_call_seconds"]
        summary["baseline_max_cuda_memory_gib"] = baseline_run["cuda"].get("max_memory_allocated_gib")
        del baseline_result
        baseline_pipe = None
        clear_cuda(args.device)

        taylorseer_pipe, taylorseer_load_seconds = load_taylorseer_pipeline(args, dtype, config)
        taylorseer_result, taylorseer_run = run_pipeline_runs(
            taylorseer_pipe,
            args,
            prompt=prompt_entry["prompt"],
            negative_prompt=negative_prompt,
            output_type="pil",
            pipeline_kind="taylorseer",
        )
        taylorseer_stats = taylorseer_pipe.get_taylorseer_stats()
        validate_taylorseer_stats(config, taylorseer_stats, num_steps=args.num_inference_steps)
        taylorseer_export_seconds = export_video(taylorseer_result.video, Path(summary["paths"]["taylorseer"]), args.fps)
        summary["taylorseer"] = {
            "load_seconds": taylorseer_load_seconds,
            "export_seconds": taylorseer_export_seconds,
            **taylorseer_run,
        }
        summary["taylorseer_stats"] = taylorseer_stats
        summary["taylorseer_seconds"] = taylorseer_run["timings"]["pipeline_call_seconds"]
        summary["taylorseer_max_cuda_memory_gib"] = taylorseer_run["cuda"].get("max_memory_allocated_gib")
        if summary["baseline_seconds"] and summary["taylorseer_seconds"]:
            summary["speedup"] = summary["baseline_seconds"] / summary["taylorseer_seconds"]
        del taylorseer_result
        taylorseer_pipe = None
        clear_cuda(args.device)

        artifact_data = generate_visual_artifacts(
            args,
            job_dir=job_dir,
            baseline_video=Path(summary["paths"]["baseline"]),
            taylorseer_video=Path(summary["paths"]["taylorseer"]),
        )
        summary["paths"].update(artifact_data["paths"])
        summary["frame_metrics"] = artifact_data["frame_metrics"]
        summary["status"] = "complete"
        summary["paths"]["review_html"] = write_review_html(job_dir, summary)
        print(
            json.dumps(
                {
                    "event": "manual_review_required",
                    "config_label": config["label"],
                    "prompt_label": prompt_entry["label"],
                    "paths": summary["paths"],
                }
            ),
            flush=True,
        )
    except RuntimeError as exc:
        if is_memory_skip(exc):
            summary["status"] = "skipped_memory"
        elif is_oom(exc):
            summary["status"] = "oom"
        else:
            summary["status"] = "error"
        summary["error"] = repr(exc)
        clear_cuda(args.device)
        try:
            summary["paths"]["review_html"] = write_review_html(job_dir, summary)
        except Exception:
            pass
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = repr(exc)
        clear_cuda(args.device)
        try:
            summary["paths"]["review_html"] = write_review_html(job_dir, summary)
        except Exception:
            pass
    finally:
        if baseline_pipe is not None:
            baseline_pipe = None
            clear_cuda(args.device)
        if taylorseer_pipe is not None:
            taylorseer_pipe = None
            clear_cuda(args.device)
        apply_manual_review(summary, review_data)
        write_summary_json(summary)
        print(
            json.dumps(
                {
                    "event": "qa_job_complete",
                    "config_label": config["label"],
                    "prompt_label": prompt_entry["label"],
                    "status": summary["status"],
                    "speedup": summary["speedup"],
                    "summary_json": summary["paths"]["summary_json"],
                }
            ),
            flush=True,
        )
    return summary


def aggregate_summaries(root: Path, *, review_data: Any = None) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            apply_manual_review(summary, review_data)
            if review_data is not None:
                path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summaries.append(summary)
        except Exception as exc:
            summaries.append({"status": "error", "error": repr(exc), "paths": {"summary_json": str(path)}})

    by_config: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        label = summary.get("config_label") or summary.get("taylorseer_config", {}).get("label") or "unknown"
        by_config.setdefault(label, []).append(summary)

    config_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for label, entries in sorted(by_config.items(), key=lambda item: QA_CONFIG_PRIORITY.get(item[0], 10_000)):
        status_counts: dict[str, int] = {}
        manual_counts: dict[str, int] = {}
        total_baseline = 0.0
        total_taylorseer = 0.0
        prompt_labels: list[str] = []
        reject_reasons: list[str] = []
        consistency_keys: list[dict[str, Any]] = []
        for entry in entries:
            status = entry.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            manual_status = entry.get("manual_review", {}).get("status", "pending")
            manual_counts[manual_status] = manual_counts.get(manual_status, 0) + 1
            prompt_label = entry.get("prompt_label")
            if prompt_label is not None:
                prompt_labels.append(prompt_label)
                consistency_keys.append(aggregate_consistency_key(entry))
            if status != "complete":
                reject_reasons.append(f"{entry.get('prompt_label', 'unknown')}: status {status}")
            if manual_status != "pass":
                reject_reasons.append(f"{entry.get('prompt_label', 'unknown')}: manual_review {manual_status}")
            baseline_seconds = entry.get("baseline_seconds")
            taylorseer_seconds = entry.get("taylorseer_seconds")
            if isinstance(baseline_seconds, (int, float)) and isinstance(taylorseer_seconds, (int, float)):
                if baseline_seconds > 0 and taylorseer_seconds > 0:
                    total_baseline += float(baseline_seconds)
                    total_taylorseer += float(taylorseer_seconds)
        duplicate_prompt_labels = sorted({prompt for prompt in prompt_labels if prompt_labels.count(prompt) > 1})
        if duplicate_prompt_labels:
            reject_reasons.append(f"duplicate prompt summaries: {', '.join(duplicate_prompt_labels)}")
        if consistency_keys:
            reference_key = consistency_keys[0]
            for index, key in enumerate(consistency_keys[1:], start=1):
                if key != reference_key:
                    reject_reasons.append(f"mixed generation/config settings in summary index {index}")
                    break
        expected_prompt_labels = {entry["label"] for entry in QA_PROMPTS} if label in QA_CONFIG_PRIORITY else set()
        missing_prompt_labels = sorted(expected_prompt_labels.difference(prompt_labels))
        if missing_prompt_labels:
            reject_reasons.append(f"missing full QA prompts: {', '.join(missing_prompt_labels)}")
        speedup = total_baseline / total_taylorseer if total_taylorseer > 0 else None
        record = {
            "label": label,
            "priority": QA_CONFIG_PRIORITY.get(label),
            "summary_count": len(entries),
            "prompt_labels": sorted(set(prompt_labels)),
            "missing_prompt_labels": missing_prompt_labels,
            "status_counts": status_counts,
            "manual_review_counts": manual_counts,
            "aggregate_speedup": speedup,
            "rejected": bool(reject_reasons),
            "reject_reasons": reject_reasons,
        }
        config_records.append(record)
        if not reject_reasons and speedup is not None:
            candidates.append(record)

    selected = None
    if candidates:
        best_speedup = max(record["aggregate_speedup"] for record in candidates if record["aggregate_speedup"] is not None)
        near_best = [
            record
            for record in candidates
            if record["aggregate_speedup"] is not None and record["aggregate_speedup"] >= best_speedup * 0.97
        ]
        selected = sorted(near_best, key=lambda record: record["priority"] if record["priority"] is not None else 10_000)[0]

    aggregate = {
        "event": "aggregate",
        "qa_output_dir": str(root),
        "summary_count": len(summaries),
        "configs": config_records,
        "selected_config": selected,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return aggregate


def run_single(args: argparse.Namespace, *, diffusers_source: str) -> dict[str, Any]:
    import torch

    dtype = dtype_from_name(args.dtype)
    prompt = load_prompt(args.prompt_json, args.prompt)
    negative_prompt = load_prompt(args.negative_prompt_json, args.negative_prompt)
    config = build_taylorseer_config(args)
    num_frames = args.num_frames
    frame_count_source = "benchmark_spec_default" if num_frames == 189 else "num_frames"

    print(
        json.dumps(
            {
                "model": args.model,
                "height": args.height,
                "width": args.width,
                "num_frames": num_frames,
                "frame_count_source": frame_count_source,
                "fps": args.fps,
                "seconds_requested": num_frames / args.fps if args.fps else None,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "output_type": args.output_type,
                "safety_check": args.enable_safety_check,
                "device": args.device,
                "dtype": args.dtype,
                "profile_steps": args.profile_steps,
                "warmup_runs": args.warmup_runs,
                "runs": args.runs,
                "qa_mode": args.qa_mode,
                "taylorseer_config": config,
                "diffusers_source": diffusers_source,
                "pid": os.getpid(),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    timings: dict[str, float] = {}
    pipe, load_seconds = load_taylorseer_pipeline(args, dtype, config)
    timings["load_pipeline_seconds"] = load_seconds
    clear_cuda(args.device)

    result, run_summary = run_pipeline_runs(
        pipe,
        args,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_type=args.output_type,
        pipeline_kind="taylorseer",
    )
    timings.update(run_summary["timings"])
    taylorseer_stats = pipe.get_taylorseer_stats()
    validate_taylorseer_stats(config, taylorseer_stats, num_steps=args.num_inference_steps)

    export_seconds = None
    if args.save_video is not None and args.output_type != "latent":
        cuda_sync(args.device)
        export_seconds = export_video(result.video, args.save_video, args.fps)

    summary: dict[str, Any] = {
        "timings": timings,
        "warmup_runs": run_summary["warmup_runs"],
        "measured_runs": run_summary["measured_runs"],
        "steps": run_summary["steps"],
        "cuda": run_summary["cuda"],
        "video_shape": run_summary["video_shape"],
        "export_seconds": export_seconds,
        "taylorseer_config": config,
        "taylorseer_stats": taylorseer_stats,
        "negative_prompt": negative_prompt,
        "add_resolution_template": args.enable_resolution_template,
        "add_duration_template": args.enable_duration_template,
        "enable_safety_check": args.enable_safety_check,
    }
    reference = maybe_h200_reference(args, num_frames, timings["pipeline_call_seconds"])
    if reference is not None:
        summary["h200_reference"] = reference

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del result
    pipe = None
    clear_cuda(args.device)
    return summary


def run_pair_mode(args: argparse.Namespace, *, diffusers_source: str) -> dict[str, Any]:
    review_data = load_review_json(args.qa_review_json)
    prompts = select_prompts(args)
    config = build_taylorseer_config(args)
    summaries = [
        run_pair_job(
            args,
            prompt_entry=prompt_entry,
            config=config,
            review_data=review_data,
            diffusers_source=diffusers_source,
        )
        for prompt_entry in prompts
    ]
    aggregate = aggregate_summaries(args.qa_output_dir, review_data=review_data)
    aggregate["job_summaries"] = [summary["paths"]["summary_json"] for summary in summaries]
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def matrix_jobs(args: argparse.Namespace) -> list[tuple[int, dict[str, Any], dict[str, str]]]:
    configs = QA_CONFIGS_SMOKE if args.qa_matrix_kind == "smoke" else QA_CONFIGS_FULL
    prompts = select_prompts(args, matrix_kind=args.qa_matrix_kind)
    jobs: list[tuple[int, dict[str, Any], dict[str, str]]] = []
    job_index = 0
    for raw_config in configs:
        for prompt_entry in prompts:
            if job_index % args.qa_num_workers == args.qa_worker_id:
                config = build_taylorseer_config(
                    args,
                    interval=raw_config["interval"],
                    layer_spec_text=raw_config["layers"],
                    label=raw_config["label"],
                    first_enhance=raw_config.get("first_enhance"),
                    last_enhance=raw_config.get("last_enhance"),
                    max_order=raw_config.get("max_order"),
                    fresh_threshold=raw_config.get("fresh_threshold"),
                    force_scheduler=raw_config.get("force_scheduler"),
                    branches=raw_config.get("branches"),
                    delta_change_threshold=raw_config.get("delta_change_threshold"),
                    prediction_target=raw_config.get("prediction_target"),
                    cache_und=raw_config.get("cache_und"),
                    stagger_layers=raw_config.get("stagger_layers"),
                    slope_scale=raw_config.get("slope_scale"),
                )
                jobs.append((job_index, config, prompt_entry))
            job_index += 1
    return jobs


def run_matrix_worker(args: argparse.Namespace, *, diffusers_source: str) -> dict[str, Any]:
    review_data = load_review_json(args.qa_review_json)
    jobs = matrix_jobs(args)
    print(
        json.dumps(
            {
                "event": "matrix_worker_start",
                "worker_id": args.qa_worker_id,
                "num_workers": args.qa_num_workers,
                "job_count": len(jobs),
                "qa_matrix_kind": args.qa_matrix_kind,
                "diffusers_source": diffusers_source,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    summaries = []
    for job_index, config, prompt_entry in jobs:
        print(
            json.dumps(
                {
                    "event": "matrix_job_shard_start",
                    "job_index": job_index,
                    "worker_id": args.qa_worker_id,
                    "config_label": config["label"],
                    "prompt_label": prompt_entry["label"],
                }
            ),
            flush=True,
        )
        summaries.append(
            run_pair_job(
                args,
                prompt_entry=prompt_entry,
                config=config,
                review_data=review_data,
                diffusers_source=diffusers_source,
            )
        )
    aggregate = aggregate_summaries(args.qa_output_dir, review_data=review_data)
    aggregate["worker_id"] = args.qa_worker_id
    aggregate["job_summaries"] = [summary["paths"]["summary_json"] for summary in summaries]
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    args = parse_args()
    if args.qa_mode == "aggregate-only":
        review_data = load_review_json(args.qa_review_json)
        aggregate = aggregate_summaries(args.qa_output_dir, review_data=review_data)
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    local_diffusers_dir = ensure_local_diffusers_source()

    import diffusers

    diffusers_source = assert_local_diffusers(diffusers.__file__, local_diffusers_dir)

    if args.qa_mode == "single":
        run_single(args, diffusers_source=diffusers_source)
    elif args.qa_mode == "pair":
        run_pair_mode(args, diffusers_source=diffusers_source)
    elif args.qa_mode == "matrix-worker":
        run_matrix_worker(args, diffusers_source=diffusers_source)
    else:
        raise ValueError(f"Unsupported qa mode: {args.qa_mode}")


if __name__ == "__main__":
    main()
