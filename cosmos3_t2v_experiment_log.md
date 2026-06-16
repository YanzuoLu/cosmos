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

## 2026-06-15 — M1 (quant+kernel roadmap): W8A8 FP8 generation-pathway linears — **LOCKED**

Purpose: first module of the quant+kernel roadmap (`cosmos3_quant_kernel_roadmap.md`). Quantize the 252
generation-pathway linears to FP8 W8A8, measured **on top of the frozen cache pipeline** `i2_o1_all_w1_c5`.

**Locked config: `cache + FP8(252 gen linears) + per-layer compile + SmoothQuant(alpha=0.5)`**

- **final8 QA: 8/8 PASS** (reference = cache pipeline output, same rubric/bar as the cache phase),
  overall mean 7.625, min 7 (every dimension >= 7, no cliff).
- **Speedup: 1.2585x over the cache pipeline** (per-prompt mean), i.e. ~**2.02x over BF16** stacked on
  the cache's 1.605x.

QA per prompt (overall, all acceptable):

| 006 | 014 | 028 | 039 | 048 | 049 | 068 | 079 |
|---|---|---|---|---|---|---|---|
| 7 | 8 | 8 | 8 | 8 | 7 | 7 | 8 |

Speedup convention (identical structure to the cache phase, baseline swapped to the cache pipeline):

```text
mean speedup = arithmetic mean over the 8 final8 prompts of
               (cache-only i2_o1_all_w1_c5 timings.pipeline_call_seconds) / (cache+M1 timings.pipeline_call_seconds)
```

Technical scheme:

- **FP8 W8A8 via TorchAO** `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())` (torchao
  0.17.0, version=2 API) applied with `quantize_(..., filter_fn=...)` to exactly the **252 generation
  linears** (36 layers x 7: `add_q/k/v_proj`, `to_add_out`, `mlp_moe_gen.{gate,up,down}_proj`). E4M3 both
  sides, per-token dynamic activation + per-channel weight scale, lowers to `torch._scaled_mm` rowwise
  (trace-verified: real `aten::_scaled_mm`, not fake-quant). Understanding/causal pathway, all RMSNorms,
  softmax, VAE stay BF16.
- **torch.compile is mandatory AND must wrap the real hot path.** The TaylorSeer full step runs
  `Cosmos3VLTextMoTDecoderLayer.forward_with_gen_delta`, NOT the plain `forward`. Regional-compiling the
  plain forward (`compile_repeated_blocks`) leaves the quant-cast prologue un-fused and makes FP8 *slower*
  than BF16. Fix: `torch.compile(layer.forward_with_gen_delta, fullgraph=True, dynamic=False)` per layer —
  fuses the per-token cast into the `_scaled_mm` prologue (quant overhead 18.6% -> 4.4%) while leaving
  TaylorSeer's python control flow outside the graph (avoids recompile-limit blowups).
- **SmoothQuant (offline, calibrated on the 100 structured prompts)** rescues the high-motion hardcase 068,
  which FP8-only fails. Per-input-channel activation absmax is collected over all 100 prompts (8-GPU
  parallel, ~16min, taylorseer cache OFF during calibration to walk the true denoising trajectory) and
  stored ONCE at `/root/cosmos/m1_smoothquant_stats/act_absmax.pt`. The migration scale
  `s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)` is then derived per alpha (no recalibration) and folded:
  `1/s` into the preceding RMSNorm weight, `s` into the linear's input columns. Migration applies only to
  the **norm-fed** linears (`add_q/k/v_proj` via `input_layernorm_moe_gen`; `gate/up_proj` via
  `post_attention_layernorm_moe_gen`); `to_add_out` and `down_proj` (no preceding norm) keep plain FP8.
  SmoothQuant is **fold-free** (zero speed cost; 0.999x vs FP8-only).

alpha sweep (speed is alpha-invariant; alpha is purely a quality knob, sweet spot at 0.5):

| config | 068 overall | result | mean speedup |
|---|---|---|---|
| FP8-only (alpha=0) | 6 | 7/8 fail (activation outliers under-protected) | 1.233x |
| **SmoothQuant alpha=0.5** | **7** | **8/8 PASS (locked)** | **1.2585x** |
| SmoothQuant alpha=0.6 | 4 | 7/8 fail (over-migration -> weight outliers, 068 melts) | 1.2532x |

Reproduce the locked M1 (one prompt per GPU; copy-per-module benchmark, cache transformer/pipeline reused,
quant applied as a torchao tensor-subclass without editing the transformer source):

```bash
python tools/diffusers_m1fp8_t2v_benchmark.py \
  --taylorseer-interval 2 --taylorseer-max-order 1 --taylorseer-first-enhance 1 \
  --taylorseer-last-enhance 5 --taylorseer-force-final-full --taylorseer-cache-und \
  --taylorseer-cache-max-gib 120 --taylorseer-branches both --taylorseer-prediction-target layer_delta \
  --m1-fp8 --m1-compile --m1-smoothquant --smoothquant-alpha 0.5 \
  --smoothquant-stats-path /root/cosmos/m1_smoothquant_stats/act_absmax.pt \
  --height 720 --width 1280 --num-frames 189 --fps 24 --num-inference-steps 35 \
  --guidance-scale 6.0 --flow-shift 10.0 --seed 1234 --dtype bf16 --warmup-runs 1 --runs 1
```

Key learnings:
- 068 (high-motion white cat) is the universal quantization stress-test, exactly as it was for cache.
  Every quant error source first shows up there. M1's fix is SmoothQuant; M2's attention quant needs its
  own accuracy fix (see M2).
- Weight-only quant is pointless on this compute-bound workload — activation quant (W8A8) is the win, and
  it only pays off once compile fuses the cast into the GEMM prologue on the real hot path.

## 2026-06-15 — M2 (quant+kernel roadmap): sparse/quant attention — **LOCKED (Phase A dense)**

Purpose: second module of the quant+kernel roadmap. Replace the generation full-attention
(`is_causal=False`) with a quantized attention kernel behind diffusers `dispatch_attention_fn`, measured
**on top of cache+M1**.

**Locked config: M2 = SageAttention2 dense quantized attention (INT8 QK^T + FP8 PV, sm90 Hopper path)**

- Installed behind `dispatch_attention_fn` (SPARGE backend in `attention_dispatch.py`). Intercepts ONLY the
  gen `is_causal=False` call -> `sageattn_qk_int8_pv_fp8_cuda_sm90`; und/causal path stays native. Receives
  already-RoPE'd, already-QK-normed q/k and does NOT re-apply them. GQA expanded via `repeat_interleave`.
  NHD layout, bf16 in/out. Single-op cosine vs SDPA = 0.99925.
- **Phase B (block-sparsity) is DEFERRED, not delivered.** SpargeAttn's SM90 block-sparse kernel
  **deadlocks** on Cosmos's real fullshape cross-attention (KV=44189) at ~denoising step 4 — a kernel-level
  hang (`torch.cuda.synchronize` never returns), reproducible without cache and without gpuburn. Bare-kernel
  probes at the same shapes do NOT hang, so the trigger is the real inference path's state, not simple tail
  misalignment. Upstream fix #77 (BLKQ=64/BLKK=128 + V pad128) is already present; #116 (H100 stuck) is
  open. Pursuing it is high-risk CUDA debugging with no upstream resolution -> parked. M2's safe, viable
  config is the **dense** quant attention (Phase A).

**Integration result — cache + M1 + M2 (the operative deliverable):**

- **final8 QA: 8/8 PASS**, overall mean 8.0, min 7 (068 confirmed by 3 independent raters: 7/8/8 -> all
  acceptable).
- **Speedup: 1.528x over the cache pipeline** (per-prompt mean, range 1.487-1.575), i.e. **~2.45x over
  BF16** stacked on the cache's 1.605x. (M1 1.2585x x M2 ~1.18x, and compile helps the attention path too.)

QA per prompt (cache+M1+M2, overall, all acceptable):

| 006 | 014 | 028 | 039 | 048 | 049 | 068 | 079 |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 8 | 8 | 8 | 8 | 7 | 9 |

Technical scheme:

- **compile + custom CUDA op composition (the key integration mechanism, reusable for M3).** M1 compiles
  `forward_with_gen_delta` with `fullgraph=True`; M2's SageAttention call is a custom CUDA op inside that
  region. A naive call graph-breaks fullgraph. Fix: register the sage call as an **opaque
  `torch.library.custom_op`** (`m1m2::sparge_fp8_dense_sm90`) with an explicit schema + fake/meta impl, so
  Inductor treats it like `_scaled_mm` (black box) while still fusing the FP8 GEMM prologue. Do NOT call
  `set_attention_backend(enum)` inside the compiled path (Dynamo chokes on the enum) — install SPARGE as the
  active-registry backend and let the compiled path take the `backend=None` branch. Trace-verified: both
  `aten::_scaled_mm` (linears) AND the sage sm90 kernel fire, 36 layers fullgraph with no graph break.
- **FP16-PV is NOT viable on Hopper.** `sageattn_qk_int8_pv_fp16_cuda` lowers to an SM80-style kernel (no
  sm90 fast path) -> **negative speedup** (0.75-0.89x vs cache). It does recover 068 quality on its own, but
  costs more than it saves, so it is rejected. The FP8 sm90 path is the only positive-speedup option.

Reproduce the integration (cache+M1+M2, one prompt per GPU):

```bash
M2_PV_VARIANT=fp8 python tools/diffusers_m1m2_t2v_benchmark.py \
  --attention-backend sparge --m1-fp8 --m1-compile --m1-smoothquant --smoothquant-alpha 0.5 \
  --smoothquant-stats-path /root/cosmos/m1_smoothquant_stats/act_absmax.pt \
  --taylorseer-interval 2 --taylorseer-max-order 1 --taylorseer-first-enhance 1 --taylorseer-last-enhance 5 \
  --taylorseer-cache-und --height 720 --width 1280 --num-frames 189 --num-inference-steps 35 \
  --guidance-scale 6.0 --flow-shift 10 --seed 1234
```

Key learnings:
- 068 (high-motion white cat) again the binding constraint. M2's attention quant alone fails it (warping),
  but **M1's SmoothQuant + M2 compose favorably** — the SmoothQuant-cleaned activations feeding attention
  give 068 just enough margin to pass in the integrated stack. Co-design synergy, measured not assumed.
- The diffusers `dispatch_attention_fn` plug-in point let us swap the attention kernel with zero model
  surgery — exactly as the roadmap predicted.

## 2026-06-15 — M3 (quant+kernel roadmap): kernel fusion + attention-sparsity attempt — **LOCKED (no further speedup; compile-fusion delivered & verified)**

Purpose: third module. Evaluation is a **correctness check** (fusion is meant to be lossless), but the
**implementation went all-out to write/fix kernels and attack the profiled bottlenecks.**

**Profiling (steady-state denoising step, 720p real shape) — where the time actually goes:**

| kernel | share | verdict |
|---|---|---|
| sage attention (dense INT8 QK^T + FP8 PV, sm90) | **70.5%** | THE bottleneck; only big lever = sparsity |
| FP8 GEMM (`_scaled_mm` CUTLASS sm90) | 20.3% | compute-bound (FLOPs fixed); horizontal gate/up fusion ~0 gain on large-M |
| FP8 cast prologue (single fused triton kernel) | 3.68% | only sizeable non-attention target; folding into GEMM mainloop = hard CUTLASS, <4% ceiling, poor risk/reward |
| RoPE / QK-norm / elementwise+norm | 3.14% (true RoPE/QK-norm-adjacent only 0.62%) | already fused by Inductor |
| other (copy/cat/memcpy) | 2.37% | — |

**3a/3b compile-fusion is DELIVERED & verified in the locked cache+M1+M2.** Per-layer
`torch.compile(forward_with_gen_delta, fullgraph=True)` already fuses residual-adds, RMSNorm,
SwiGLU-mul, RoPE/QK-norm elementwise, and the RMSNorm->amax->FP8-cast prologue between the `_scaled_mm`
and sage kernels (this is most of why dense-integrated 110s << dense-standalone 136s). **Correctness
verified against an fp32 gold judge** (NOT eager-FP8, which is not a ground truth in the FP8 domain):
compile-FP8 rel-L2=0.146 is *closer* to fp32 than eager-FP8's 0.312 — compile-fusion is numerically
faithful and in fact more accurate than eager.

**Attention block-sparsity (the 70.5% lever) — attempted ALL-OUT, NO-GO on quality.**

1. **SM90 block-sparse deadlock — root-caused & cleanly bypassed.** SpargeAttn's fused **PV-threshold**
   WGMMA kernel deadlocks on Cosmos's real fullshape cross-attention (KV=44189) at ~step 4 — a
   low-probability async-barrier race (heisenbug; matches upstream #116, open, no maintainer fix). The
   KV-tail-misalignment hypothesis was disproven (padding K to 128 still hangs). **The bug lives only in
   the PV-threshold path**; switching to the **non-PV** SM90 binding
   (`...block_sparse_attn_inst_buf_fuse_v_scale_sm90`) bypasses it cleanly — no hand-fixing the WGMMA
   barrier (which a council unanimously flagged as a silent-wrong-on-untested-shapes landmine).
2. **Sparse integrated into the compiled stack** (opaque `custom_op` + non-PV kernel + adaptive
   block-map): graph-break 0, trace shows `_scaled_mm` + non-PV block-sparse kernel, finite.
3. **Speed lever is real** (fair comparison sparse-integrated vs dense-integrated, NOT the earlier
   apples-to-oranges sparse-standalone-vs-dense-integrated): crude topk=0.5 gave ~1.25-1.32x over
   dense-integrated.
4. **But quality collapses on the high-motion hardcase 068 at ANY speed-positive sparsity:**
   - crude topk=0.5 / 0.65 (drop 50% / 35%): 068 overall **2-3** (massive melting/warping/ghosting).
   - content-adaptive meansim (CDF tau + self-similarity theta=0.6 fix-block guard ON) at tau=0.90/0.95/0.98
     (drop only 3-8%, guard covering ~90%): 068 STILL **3-4**, AND the 3-8% skip is eaten by block-map
     overhead -> ~net-zero speed. Even the most conservative 3.4%-skip setting melts 068.
   - **Root cause:** 068 (high motion) has flat, critical attention with NO safely-skippable blocks; the
     self-similarity guard's heuristic mis-classifies its critical blocks as skippable. To keep 068's
     quality requires ~0% skip = dense. This is the roadmap's predicted failure mode ("busy/fast content
     has flatter attention with fewer skippable blocks") and 068 is the universal hardcase across cache,
     M1, M2. Since QA is strict 8/8 equal-weight, 068 gates the whole config -> **GLOBAL block-sparse
     cannot yield an 8/8 config faster than dense+M1+M2.**

**Per-layer selective sparsity (Phase C) — the promising lever, but NO-GO on a falsified proxy.**
Hypothesis: 068's sparsity-intolerance is concentrated in some layers; sparsify only the tolerant ones.
- Per-layer 068 sensitivity (per-layer attention-output rel-L1 from sparsifying one layer) IS concentrated:
  boundary layers (L33/31/02/34/32/30/01, near input/output) are critical; mid layers L05-L20 are tolerant.
- Sparsifying only the 6 least-sensitive layers ({9,13,15,16,17,20}, ~8% overall skip) **rescued 068 to a
  robust overall-7 (3 independent raters: 7/7/7)** — the first speed-positive non-melting 068 config,
  1.096x over dense. Step-gated (late-steps-dense) variants FAILED (double-cat/ghosting). Speed floors at
  ~1.10x over dense at ~8% skip.
- **But the full final8 was 7/8: the held-out 028 (turtle) MELTED (overall 6)** despite the sparse set being
  068-calibrated. Decisive finding: those same 6 layers are **028's OWN lowest-error / "safest" layers
  (rank 31-36, rel-L1 0.05-0.08)** — the L1 proxy is **silent exactly where it should warn**. **The
  attention-output L1 proxy is falsified as a per-layer quality selector.** Root cause is a metric category
  error (not too few calibration prompts): the melting is **compound cross-layer x cross-step structural /
  high-frequency texture degradation**, which an isolated per-layer, few-step, mean-magnitude L1 cannot
  integrate (blind to phase/frequency). Worst-case-over-N aggregation cannot fix it (028's failing layers
  already have ~zero proxy error), and a "clean" robust-set held-out eval would be a TRAP (false-pass on
  easy content -> melt the next 028-like prompt in production). Reviving per-layer sparsity needs a
  **quality-correlated proxy** (end-to-end per-layer ablation human-eval, or a texture/frequency-sensitive
  metric) -- a research pivot, not worth blocking M3. **Per-layer sparsity NO-GO under this proxy.**

**Lossless GEMM/cast fusion — implemented & measured, ~0 e2e gain.** gate/up horizontal GEMM fusion (concat
gate_proj+up_proj into one `[2I,H]` FP8 GEMM; composes with SmoothQuant + per-layer compile + sage custom_op)
was implemented and proven **bitwise lossless** (in-process forward equivalence: max-abs=0, cosine=1.0; the
eager cross-process drift seen first was harness non-determinism, not fusion). But e2e gain is **~0-2% (in
measurement noise)** because the 720p large-M (44160-token) GEMMs are compute-bound -- the saved launch
overhead is amortized away. Folding the FP8 cast prologue (3.68%) into the GEMM mainloop has no TorchAO
switch and would require hand-CUTLASS SM90 work (high risk, <4% ceiling) -> skipped. Neither is delivered.

**Decision:** M3 ships the **verified compile-fused cache+M1+M2** (no additional approximation). The
dominant bottleneck (attention) is a tuned INT8/FP8 Hopper dense kernel whose only big lever (sparsity)
is quality-infeasible here; the remaining lossless targets are <4% on compute-bound GEMMs / hard CUTLASS
with poor risk/reward. The sparse benchmark + non-PV bypass are kept as an **opt-in research artifact**
(`diffusers_m1m2sparse_t2v_benchmark.py`, env-gated), default path byte-identical to dense.

**Final delivered stack: cache + M1 + M2 = 8/8 @ 1.528x over cache (~2.45x over BF16).**

## Candidate 1 (STA static 3D sliding-tile block-sparse quant kernel) — **LOCKED (8/8) — real speedup over dense baseline @ w20-20-34**

Purpose: revisit the M3 attention-sparsity lever (the profiled 70.5% bottleneck) with a **static**, content-independent
block-sparse mask, sidestepping the adaptive-proxy failures that sank M3's global/per-layer attempts.

**Mechanism.** A precomputed static **int32 LUT** drives a 3D sliding-tile (STA) block-sparse pattern through the
SageAttn2 **no_pv** INT8-QK + FP8-PV sm90 kernel. The sparse call is registered as an opaque
`torch.library.custom_op` (`m1m2::sparge_sta_static_sm90`) so it composes inside the M1 fullgraph compile exactly like
the dense M2 custom_op. The mask is **static per geometry** -> **zero per-step mask overhead** (no block-map recompute,
the cost that ate M3's adaptive skip). A mask-of-all-1s is **bit-equivalent to dense** (clean fallback / correctness
anchor). Geometry: T48/H23/W40 visual grid, und/text prefix = K-idx 0-42 (force-kept for every query), blkq64/blkk128,
mask 690x346. Selected at runtime via `STA_SPARSE=1 STA_WT=<t> STA_WH=<h> STA_WW=<w>`.

**Key finding — high-motion video is sparsity-tolerant on the TEMPORAL axis ONLY.** Sweeping the window showed a sharp
quality cliff governed entirely by temporal coverage at full spatial extent: the cliff sits **between +/-16 (fail) and
+/-20 (pass) temporal frames**. Spatial sparsity is *not* safely skippable — shrinking the spatial window collapses
fast-moving subjects into ghosting/melting. Cubic windows that skip spatially fail the high-motion hardcases at any
speed-positive setting: w16-23-40 (skip 0.43) 068=4/028=3; w12-23-40 (0.54) 068=5/028=2; w8-23-40 (0.67) 068=3/028=2;
the original cubic sweep (w12-12-12 / w6-6-6 / w3-3-3 / w2-2-2, skip 0.57-0.94) all land 068/028 at 2-3. So the full
spatial window (WH=23, WW=40) is mandatory and **temporal is the only axis with headroom** — ceiling ~32.5% block skip.

**Pareto config: w20-20-34** (full spatial + temporal +/-20 frames, **32.5% block skip**). Final8 QA, strict 8/8 gate:

| 006 | 014 | 028 | 039 | 048 | 049 | 068 | 079 |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 8 | 8 | 8 | 8 | 7 | 8 |

8/8 PASS (overall mean 7.875, min 7). The two universal hardcases **028 (high-motion turtle) = 8 and 068 (high-motion
white cat) = 7 both match dense — no regression**. Per-prompt speedup vs dense-integrated ranges 1.10-1.26x.

**Speed:** mean **1.17x over dense-integrated** = **~1.79x over cache** = **~2.87x over BF16**. STA lifts the locked
stack from 1.528x to ~1.79x over cache while holding 8/8.

**Integration-bug hypothesis REFUTED (independent diagnosis).** Before accepting the cliff as honest, the obvious
"it's a plumbing bug" theory was attacked and falsified: und/text tokens *are* the prefix K-idx 0-42 (matches the mask,
force-kept for all queries); the visual flatten order is **t-major / h-mid / w-minor**, confirmed against the real
patchify + mrope; LUT / GQA / axis mapping all correct. **Decisive falsification:** a near-full window (24,23,40, skip
0.24) is clean and ~=dense, and quality degrades **monotonically** with skip — that is honest coverage loss, not a bug.
The cliff is a real mechanism of fast motion, not an artifact.

**Decision:** **STA LOCKED at w20-20-34.** Mechanism takeaway: for high-motion video the only safe sparsity axis is
temporal; full spatial coverage is required to track fast-moving subjects. Sparsity ceiling ~32.5% block skip.

Reproduce (one prompt per GPU; same m1m2sparse runner flags as the dense stack, plus the STA env):

```bash
M2_PV_VARIANT=fp8 STA_SPARSE=1 STA_WT=20 STA_WH=20 STA_WW=34 \
python tools/diffusers_m1m2sparse_t2v_benchmark.py \
  --attention-backend sparge --m1-fp8 --m1-compile --m1-smoothquant --smoothquant-alpha 0.5 \
  --smoothquant-stats-path /root/cosmos/m1_smoothquant_stats/act_absmax.pt \
  --taylorseer-interval 2 --taylorseer-max-order 1 --taylorseer-first-enhance 1 --taylorseer-last-enhance 5 \
  --taylorseer-cache-und --height 720 --width 1280 --num-frames 189 --num-inference-steps 35 \
  --guidance-scale 6.0 --flow-shift 10 --seed 1234
```

## Candidate 2 (axial / decomposed attention) — **NO-GO (high-motion collapse)**

**Mechanism.** A new mask builder (`tools/axial_mask.py:build_axial_block_mask`) emits a pure **axial / decomposed**
block pattern — each query attends only to the horizontal, vertical, and temporal axis-lines through its own position
(a 3D "plus"/cross), not its full 2D spatial neighborhood. Driven through the **same** no_pv INT8-QK + FP8-PV LUT path
and `m1m2::sparge_sta_static_sm90` custom_op as STA, selected via `AXIAL_SPARSE=1`. Realized **76.95% block skip**
(all-1 mask bit-equiv to dense; kernel-landed + NaN-free). Speed-positive: 068 1.43x, 028 1.55x over dense-integrated
(~1.5x over dense, well above STA's 1.17x — **if quality had held**).

**Binding-gate result (unanimous fail).** The two high-motion hardcases both collapse: **068 = 3** (raters 3,3 —
cat melts into white blob, ghosted hood duplicates, horizontal smearing) and **028 = 2** (raters 2,2 — vertical
stacked ghost/duplicate turtles, shell melting, water banding). The failure mode is **off-axis 2D spatial-neighbor
loss** manifesting as axis-line ghosting (vertical-stacked duplicates, horizontal smears).

**Verdict: NO-GO.** Pure axial is fast but catastrophically destroys high-motion binding. This **reinforces the
Candidate-1 finding**: high-motion video requires FULL local 2D spatial coverage — the cross/plus axis-lines are
insufficient. Thickening the axial bands toward full-spatial just converges back to STA, so axial is **not a
distinct viable lever**. Remaining axial sweeps skipped.

## Candidate 3 (dilated / strided temporal) — **NO-GO (high-motion temporal ghosting)**

**Mechanism.** A new mask builder (`tools/dilated_mask.py:build_dilated_temporal_block_mask`) keeps **full spatial**
coverage (WH=23, WW=40) but makes the **temporal** axis dilated: a contiguous near-frame **core** plus **strided far
frames** out to a range. Driven through the **same** no_pv INT8-QK + FP8-PV LUT path and `m1m2::sparge_sta_static_sm90`
custom_op as STA, selected via `DILATED_SPARSE=1` + `DIL_CORE`/`DIL_STRIDE`/`DIL_RANGE` (all-1 mask bit-equiv to dense;
kernel-landed + NaN-free). Two configs: **D1** core10/stride2/range23 = **41.5% block skip** (~1.20x), **D2**
core6/stride3/range23 = **57.2% block skip** (~1.33x) — both beat STA's 32.5% skip / 1.17x.

**Binding-gate result.** The high-motion hardcase **068 collapses** with **temporal ghosting / cat duplication**:
**D1 068 = [4,6]** (fail), **D2 068 = [6,7,6,5]** (3/4 raters fail — two distinct white-cat bodies, clearest frame 96).
**028 passes both** (D1 [7,8], D2 [8,7]). The strided far frames (sees t-3,t-6 but not t-1,t-2,t-4,t-5) prevent the
model from interpolating fast motion, so it ghosts/doubles the subject. This artifact is **absent in STA's contiguous
window** (068 clean = 7) and **absent in dense** (068 = 7,8,8).

**Verdict: NO-GO.** Dilated-temporal buys more skip/speed than STA and 028 passes, but high-motion binding needs
**dense contiguous near-frame temporal context** — striding the far frames causes ghosting/duplication. **Contiguous
STA (w20-20-34) stays best.**
