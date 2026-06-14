import hashlib
import json

from tools import taylorseer_final8_candidate_planner as planner


def test_final8_planner_uses_raw_prompt_json_and_baseline_metadata(monkeypatch):
    monkeypatch.setattr(planner.sys, "executable", "/tmp/not-python3")

    plan = planner.build_plan(planner.FINAL_SELECTION_JSON, planner.DEFAULT_OUTPUT_ROOT)
    assert plan["final_prompt_ids"] == [28, 14, 48, 79, 49, 39, 68, 6]
    assert len(plan["jobs"]) == 64

    manifest = planner.load_final_selection(planner.FINAL_SELECTION_JSON)
    items_by_id = {int(item["prompt_id"]): item for item in manifest["items"]}

    for job in plan["jobs"]:
        prompt_id = job["prompt_id"]
        item = items_by_id[prompt_id]
        run_root = planner.FINAL_SELECTION_JSON.parent.parent
        prompt_path = run_root / item["prompt_json"]
        expected_text = json.dumps(json.loads(prompt_path.read_text(encoding="utf-8")))

        assert job["command"][0] == "python3"
        assert job["prompt_json_path"] == str(prompt_path)
        assert job["prompt_text"] == expected_text
        assert job["prompt_hash"] == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
        assert job["baseline_summary_path"] == str(run_root / item["summary"])
        assert job["baseline_video_path"] == str(run_root / item["video"])
        assert job["baseline_seconds"] == item["pipeline_call_seconds"]


def test_final8_planner_commands_stay_candidate_only_single_mode():
    plan = planner.build_plan(planner.FINAL_SELECTION_JSON, planner.DEFAULT_OUTPUT_ROOT)

    for job in plan["jobs"]:
        command = job["command"]
        command_text = job["command_str"]

        assert command[1].endswith("tools/diffusers_taylorseer_t2v_benchmark.py")
        assert command.count("--prompt-json") == 1
        assert command.count("--summary-json") == 1
        assert "--prompt" not in command
        assert "--qa-mode" in command and command[command.index("--qa-mode") + 1] == "single"
        assert "--qa-output-dir" not in command
        assert "--qa-review-json" not in command
        assert "pair" not in command_text
        assert "matrix-worker" not in command_text
        assert "aggregate-only" not in command_text
        assert "source_prompt" not in command_text


def test_final8_planner_resolves_paths_relative_to_custom_final_selection_json(tmp_path, monkeypatch):
    run_root = tmp_path / "baseline-run"
    final_selection_dir = run_root / "final_selection"
    prompt_dir = final_selection_dir / "prompts"
    summary_dir = final_selection_dir / "summaries"
    video_dir = final_selection_dir / "videos"
    for directory in (prompt_dir, summary_dir, video_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prompt_path = prompt_dir / "prompt_001.json"
    summary_path = summary_dir / "prompt_001.json"
    video_path = video_dir / "prompt_001.mp4"
    prompt_path.write_text(json.dumps({"prompt": "hello"}), encoding="utf-8")
    summary_path.write_text(json.dumps({"summary": "baseline"}), encoding="utf-8")
    video_path.write_text("video", encoding="utf-8")

    custom_final_selection_json = final_selection_dir / "final_selection.json"
    custom_final_selection_json.write_text(
        json.dumps(
            {
                "run_id": "custom-run",
                "final_prompt_ids": [1],
                "items": [
                    {
                        "prompt_id": 1,
                        "prompt_json": "final_selection/prompts/prompt_001.json",
                        "summary": "final_selection/summaries/prompt_001.json",
                        "video": "final_selection/videos/prompt_001.mp4",
                        "pipeline_call_seconds": 12.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(planner, "FINAL_SELECTION_ROOT", tmp_path / "wrong" / "final_selection")

    plan = planner.build_plan(custom_final_selection_json, tmp_path / "outputs")

    job = plan["jobs"][0]
    assert job["prompt_json_path"] == str(run_root / "final_selection/prompts/prompt_001.json")
    assert job["baseline_summary_path"] == str(run_root / "final_selection/summaries/prompt_001.json")
    assert job["baseline_video_path"] == str(run_root / "final_selection/videos/prompt_001.mp4")
