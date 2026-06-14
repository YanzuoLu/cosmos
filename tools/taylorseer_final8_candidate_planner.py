#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_SELECTION_ROOT = (
    REPO_ROOT
    / "assets"
    / "cosmos3_t2v_baseline_runs"
    / "hf_download_0614d"
    / "cosmos3_t2v_baseline_runs"
    / "0614d_structured_json_seed100_0383c17"
    / "final_selection"
)
FINAL_SELECTION_JSON = FINAL_SELECTION_ROOT / "final_selection.json"
RUNNER = Path("tools/diffusers_taylorseer_t2v_benchmark.py")
PYTHON_EXECUTABLE = "python3"
DEFAULT_OUTPUT_ROOT = Path("artifacts/cosmos3_taylorseer_final8_candidate_only")
COMMON_RUNNER_ARGS = [
    "--qa-mode",
    "single",
    "--model",
    "nvidia/Cosmos3-Nano",
    "--height",
    "720",
    "--width",
    "1280",
    "--fps",
    "24",
    "--num-frames",
    "189",
    "--num-inference-steps",
    "35",
    "--guidance-scale",
    "6.0",
    "--flow-shift",
    "10.0",
    "--seed",
    "1234",
    "--dtype",
    "bf16",
]

CANDIDATE_CONFIGS: tuple[dict[str, Any], ...] = (
    {"label": "i2_o0_w9_c9_l35", "interval": 2, "max_order": 0, "first_enhance": 9, "last_enhance": 9, "layer_indices": "35"},
    {"label": "i2_o0_w12_c12_l35", "interval": 2, "max_order": 0, "first_enhance": 12, "last_enhance": 12, "layer_indices": "35"},
    {"label": "i2_o0_w9_c9_l34_35", "interval": 2, "max_order": 0, "first_enhance": 9, "last_enhance": 9, "layer_indices": "34-35"},
    {"label": "i2_o0_w12_c12_l34_35", "interval": 2, "max_order": 0, "first_enhance": 12, "last_enhance": 12, "layer_indices": "34-35"},
    {"label": "i3_o0_w12_c12_l35", "interval": 3, "max_order": 0, "first_enhance": 12, "last_enhance": 12, "layer_indices": "35"},
    {"label": "i3_o0_w12_c12_l34_35", "interval": 3, "max_order": 0, "first_enhance": 12, "last_enhance": 12, "layer_indices": "34-35"},
    {"label": "i2_o0_w14_c14_l33_35", "interval": 2, "max_order": 0, "first_enhance": 14, "last_enhance": 14, "layer_indices": "33-35"},
    {"label": "i2_o0_w16_c16_l32_35", "interval": 2, "max_order": 0, "first_enhance": 16, "last_enhance": 16, "layer_indices": "32-35"},
)


def load_prompt_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return json.dumps(json.load(handle))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-selection-json", type=Path, default=FINAL_SELECTION_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="Run the planned commands instead of printing JSON only.")
    return parser.parse_args()


def load_final_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_command(prompt_json_path: Path, summary_json_path: Path, video_path: Path, config: dict[str, Any]) -> list[str]:
    command = [
        PYTHON_EXECUTABLE,
        str(RUNNER),
        *COMMON_RUNNER_ARGS,
        "--prompt-json",
        str(prompt_json_path),
        "--summary-json",
        str(summary_json_path),
        "--save-video",
        str(video_path),
        "--taylorseer-interval",
        str(config["interval"]),
        "--taylorseer-max-order",
        str(config["max_order"]),
        "--taylorseer-first-enhance",
        str(config["first_enhance"]),
        "--taylorseer-last-enhance",
        str(config["last_enhance"]),
        "--taylorseer-layer-indices",
        config["layer_indices"],
    ]
    return command


def build_jobs(final_selection: dict[str, Any], output_root: Path, run_root: Path) -> list[dict[str, Any]]:
    items_by_id = {int(item["prompt_id"]): item for item in final_selection["items"]}
    jobs: list[dict[str, Any]] = []
    for prompt_id in final_selection["final_prompt_ids"]:
        item = items_by_id[int(prompt_id)]
        prompt_label = f"prompt_{int(prompt_id):03d}"
        prompt_json_path = run_root / item["prompt_json"]
        prompt_text = load_prompt_text(prompt_json_path)
        prompt_hash = sha256_text(prompt_text)
        baseline_summary_path = run_root / item["summary"]
        baseline_video_path = run_root / item["video"]
        baseline_seconds = item["pipeline_call_seconds"]

        for config in CANDIDATE_CONFIGS:
            summary_json_path = output_root / config["label"] / prompt_label / "summary.json"
            video_path = output_root / config["label"] / prompt_label / "taylorseer.mp4"
            command = build_command(prompt_json_path, summary_json_path, video_path, config)
            jobs.append(
                {
                    "prompt_id": int(prompt_id),
                    "prompt_label": prompt_label,
                    "config_label": config["label"],
                    "prompt_json_path": str(prompt_json_path),
                    "prompt_text": prompt_text,
                    "prompt_hash": prompt_hash,
                    "baseline_summary_path": str(baseline_summary_path),
                    "baseline_video_path": str(baseline_video_path),
                    "baseline_seconds": baseline_seconds,
                    "summary_json_path": str(summary_json_path),
                    "video_path": str(video_path),
                    "command": command,
                    "command_str": shlex.join(command),
                }
            )
    return jobs


def build_plan(final_selection_path: Path, output_root: Path) -> dict[str, Any]:
    final_selection = load_final_selection(final_selection_path)
    jobs = build_jobs(final_selection, output_root, final_selection_path.parent.parent)
    return {
        "run_id": final_selection["run_id"],
        "final_selection_json": str(final_selection_path),
        "output_root": str(output_root),
        "candidate_configs": [config["label"] for config in CANDIDATE_CONFIGS],
        "final_prompt_ids": final_selection["final_prompt_ids"],
        "jobs": jobs,
    }


def main() -> None:
    args = parse_args()
    plan = build_plan(args.final_selection_json, args.output_root)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return

    for job in plan["jobs"]:
        subprocess.run(job["command"], check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
