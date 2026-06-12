#!/usr/bin/env python3
"""Minimal Cosmos3 Diffusers text-to-video benchmark.

Defaults match the public Nano T2V Diffusers benchmark profile: single-GPU
H200-class 720p generation with 189 frames at 24 FPS.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    local_diffusers_dir = ensure_local_diffusers_source()

    import diffusers
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    diffusers_source = assert_local_diffusers(diffusers.__file__, local_diffusers_dir)

    prompt = load_prompt(args.prompt_json, args.prompt)
    negative_prompt = load_prompt(args.negative_prompt_json, args.negative_prompt)
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
                "diffusers_source": diffusers_source,
                "pid": os.getpid(),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype),
        device_map=args.device if args.device.startswith("cuda") else None,
        enable_safety_checker=args.enable_safety_check,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    cuda_sync(args.device)
    timings["load_pipeline_seconds"] = time.perf_counter() - t0
    print(json.dumps({"event": "loaded", "seconds": timings["load_pipeline_seconds"], **cuda_stats(args.device)}), flush=True)

    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    def video_shape_of(result) -> Any:
        if args.output_type in {"latent", "pt", "np"}:
            return list(result.video.shape)
        return [len(result.video), getattr(result.video[0], "height", None), getattr(result.video[0], "width", None)]

    def make_generator():
        if args.device.startswith("cuda"):
            return torch.Generator(device=args.device).manual_seed(args.seed)
        return torch.Generator().manual_seed(args.seed)

    def run_once(kind: str, index: int, *, profile_steps: bool):
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
                            "run_kind": kind,
                            "run_index": index,
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
            num_frames=num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            enable_sound=False,
            generator=make_generator(),
            output_type=args.output_type,
            add_resolution_template=args.enable_resolution_template,
            add_duration_template=args.enable_duration_template,
            enable_safety_check=args.enable_safety_check,
            callback_on_step_end=on_step_end if profile_steps else None,
        )
        cuda_sync(args.device)
        seconds = time.perf_counter() - call_start
        record = {
            "kind": kind,
            "index": index,
            "seconds": seconds,
            "video_shape": video_shape_of(result),
            "steps": summarize_steps(step_end_offsets, seconds),
            "cuda": cuda_stats(args.device),
        }
        print(
            json.dumps(
                {
                    "event": "run_complete",
                    "run_kind": kind,
                    "run_index": index,
                    "seconds": seconds,
                    "video_shape": record["video_shape"],
                    "max_memory_allocated_gib": record["cuda"].get("max_memory_allocated_gib"),
                }
            ),
            flush=True,
        )
        return result, record

    warmup_records = []
    for run_index in range(args.warmup_runs):
        warmup_result, warmup_record = run_once("warmup", run_index, profile_steps=False)
        warmup_records.append(warmup_record)
        del warmup_result
        gc.collect()

    measured_records = []
    result = None
    for run_index in range(args.runs):
        if result is not None:
            del result
            gc.collect()
        result, record = run_once("measured", run_index, profile_steps=args.profile_steps)
        measured_records.append(record)

    measured_seconds = [record["seconds"] for record in measured_records]
    timings["pipeline_call_seconds"] = statistics.fmean(measured_seconds)
    timings["pipeline_call_seconds_median"] = statistics.median(measured_seconds)
    timings["pipeline_call_seconds_min"] = min(measured_seconds)
    timings["pipeline_call_seconds_max"] = max(measured_seconds)

    export_seconds = None
    if args.save_video is not None and args.output_type != "latent":
        from diffusers.utils import export_to_video

        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        cuda_sync(args.device)
        export_start = time.perf_counter()
        export_to_video(result.video, str(args.save_video), fps=int(args.fps), macro_block_size=1)
        export_seconds = time.perf_counter() - export_start

    summary: dict[str, Any] = {
        "timings": timings,
        "warmup_runs": warmup_records,
        "measured_runs": measured_records,
        "steps": measured_records[0]["steps"],
        "cuda": measured_records[-1]["cuda"],
        "video_shape": measured_records[-1]["video_shape"],
        "export_seconds": export_seconds,
    }
    reference = maybe_h200_reference(args, num_frames, timings["pipeline_call_seconds"])
    if reference is not None:
        summary["h200_reference"] = reference

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
