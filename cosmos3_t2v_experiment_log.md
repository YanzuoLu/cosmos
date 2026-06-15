# Cosmos3 T2V Experiment Log

Running log for Cosmos3-Nano T2V Diffusers benchmark and optimization experiments. Append new entries below; keep each entry short and data-focused.

## 2026-06-12 — Vanilla Diffusers BF16 baseline

Purpose: establish the measured baseline for later training-free acceleration work.

Spec:

```text
model: nvidia/Cosmos3-Nano
pipeline: diffusers.Cosmos3OmniPipeline
profile: T2V 720p/1, 189 frames @ 24 FPS
resolution: 1280x720
steps: 35
guidance_scale: 6.0
flow_shift: 10.0
precision: BF16
batch_size: 1
prompt: root README Diffusers example
negative_prompt: empty
safety_check: disabled
sound: disabled
```

Command shape:

```bash
python tools/diffusers_t2v_benchmark.py \
  --disable-progress-bar \
  --warmup-runs 1 \
  --runs 1 \
  --summary-json <summary.json>
```

Result:

```text
published_reference: 239.60s
warmup_run_0: 255.87s
measured_run_0: 252.16s
absolute_gap_vs_reference: +12.56s
relative_gap_vs_reference: +5.2%
peak_allocated_memory: 41.82 GiB
output_shape: [189, 720, 1280]
```

Baseline for follow-up experiments: **252.16s** end-to-end generation time, excluding model load, after one full pipeline warmup. Report future speedups against this measured baseline and also include the published 239.60s reference for context.

## 2026-06-15 — TaylorSeer feature-cache Pareto candidates

Purpose: record three TaylorSeer cache configs on the Pareto frontier of the final8 prompt sweep, with reproduction recipe.

Candidates (mean speedup over the 8 final8 prompts):

| candidate | mean speedup | interval | max_order | layers | first_enhance (w) | last_enhance (c) | slope_scale (s) |
|---|---|---|---|---|---|---|---|
| i2_o1_all_w1_c5 | 1.605x | 2 | 1 | all (36) | 1 | 5 | 1.0 |
| i2_o1_all_w1_c1_s0.5 | 1.733x | 2 | 1 | all (36) | 1 | 1 | 0.5 |
| i2_o1_all_w1_c1 | 1.758x | 2 | 1 | all (36) | 1 | 1 | 1.0 |

Measurement convention:

```text
mean speedup = arithmetic mean over the 8 final8 prompts of
               (0614d baseline timings.pipeline_call_seconds) / (config timings.pipeline_call_seconds)
```

Technical scheme:

TaylorSeer is feature-cache acceleration on Cosmos3-Nano T2V (1280x720, 189 frames, 35 inference steps, guidance 6.0, flow-shift 10.0, bf16, seed 1234). Every `interval` denoising steps it does one full backbone compute; the intermediate steps PREDICT each layer's generation-pathway residual (`prediction_target=layer_delta`) via Taylor extrapolation from the cached factors. `first_enhance` (w) forces full compute for the first w steps; `last_enhance` (c) forces full compute for the last c steps; `slope_scale` (s) damps the linear extrapolation term (1.0 = undamped); all 36 MoT layers are cached.

Reproduce one candidate via the benchmark tool:

```bash
python tools/diffusers_taylorseer_t2v_benchmark.py --qa-mode single \
  --taylorseer-interval <I> \
  --taylorseer-max-order <O> \
  --taylorseer-first-enhance <w> \
  --taylorseer-last-enhance <c> \
  [--taylorseer-slope-scale <s>] \
  --taylorseer-cache-max-gib 120
```

Note: omit `--taylorseer-layer-indices` entirely to mean "all layers" (the literal value `all` is invalid).
