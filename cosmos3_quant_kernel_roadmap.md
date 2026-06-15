# Cosmos3-Nano T2V — Quantization + Kernel Optimization Roadmap (H200)

> Audience: a future session with **no prior context**. Read §0 first — it explains why every
> choice below is the way it is. This is a **plan**, not a changelog; no code has been written yet
> for M1–M4. The cache layer it builds on top of is already done and frozen.

---

## 0. Context — why this exists (read this first)

### 0.1 Starting point: we already have a cache-accelerated pipeline
We are accelerating **NVIDIA Cosmos3-Nano text-to-video** at the public profile (1280×720, 189 frames,
24fps, 35 steps, guidance 6.0, flow-shift 10.0, bf16) on a **single H200**. **A working
cache-accelerated pipeline already exists** and is this roadmap's frozen starting point: the TaylorSeer
feature-cache config **`i2_o1_all_w1_c5` @ 1.605× speedup, 8/8 on the final8 visual-QA suite**. (How we
got there: `_compare/frontier_summary.md`; log: `cosmos3_t2v_experiment_log.md`.)

This roadmap adds **two more optimizations on top of that cached pipeline**, plus the fusion that welds
them — all measured **end-to-end on the cached pipeline** (so cache×quant×kernel interactions are
tested, not assumed):
- **M1 — quantize the generation-pathway linears** (W8A8 FP8).
- **M2 — replace the generation attention with a sparse-FP8 kernel.**
- **M3 — fuse the seams** (norm / RoPE / quant prologues) that M1+M2 leave behind.

Their speedups stack multiplicatively on the 1.605×. (Context only, **not** a QA rule: the cache's
hardest prompt happens to be the high-motion 068; the QA in §0.5 scores all 8 final8 prompts **equally**,
identical to the cache phase.)

### 0.2 Hardware reality (H200 = Hopper, sm90)
- Native low-precision tensor cores: **FP8 (E4M3/E5M2) and INT8 only**. **NO native FP4/NVFP4/FP6**
  (Blackwell-only) and **no hardware block/microscaling (MXFP8)** (Blackwell-only). INT4 is not a
  first-class Hopper format. → **The precision ceiling here is FP8.** (Cosmos3's advertised NVFP4 "2×"
  is unreachable on H200; that is why B200 ≈ 2× H200 in NVIDIA's own benchmarks.)
- **Memory is not the bottleneck** (141 GB; model is 8B-class). The T2V diffusion workload is
  **compute-bound** with large GEMM M (720p × 189 frames → huge token count). Therefore
  **weight-only quantization gives ≈0 speedup** — we must quantize **activations too (W8A8)** to feed
  the FP8 tensor cores, and make **attention** cheaper.

### 0.3 Architecture facts that shape everything (from `transformer_cosmos3.py`)
The generator is **NOT a standalone DiT** — it is a **Qwen3-VL Mixture-of-Transformers (MoT)** decoder
with **dual-pathway** packed attention. Each `Cosmos3VLTextMoTDecoderLayer` carries **two separate
weight sets**:
- *understanding / causal* pathway (text conditioning): `to_q/k/v/out`, `mlp` — **static across
  denoising steps; leave in BF16**.
- *generation / full-attention* pathway (the denoising hot path): `add_q_proj/add_k_proj/add_v_proj`,
  `to_add_out`, `mlp_moe_gen` (SwiGLU = `gate_proj/up_proj/down_proj`).

Consequences that drive M1/M2/M3:
1. **Quantize only the 252 generation-pathway linears** (36 layers × 7). Halves the quant surface;
   the und/text path and all norms/softmax/VAE stay BF16.
2. **The bottleneck is the gen full-attention**: gen-query attends to **concatenated `[und+gen]`**
   K/V with **`is_causal=False`**, O(N²) over a very long 3-D (T×H×W) token sequence. (Hypothesized
   ~majority of step time, per the STA paper's 720p finding — the real share is learned for free from M1's vs M2's e2e delta).
3. The model already applies **QK-norm** (`norm_added_q/k`, per-head RMSNorm) **and applies RoPE
   OUTSIDE the attention kernel** (in `Cosmos3AttnProcessor`). Two implications:
   (a) QK-norm bounds q/k magnitude → **low-bit attention is unusually safe on this model**;
   (b) any swapped attention kernel receives **already-RoPE'd, already-QK-normed** q/k and **must not
   re-apply** them.
4. The attention backend is **pluggable via diffusers `dispatch_attention_fn` / `attention_backend`**
   → we can install a custom attention kernel **without model surgery**, intercepting only the gen
   `is_causal=False` call.
5. The two RMSNorms feeding gen GEMMs are `input_layernorm_moe_gen` (→ qkv-proj) and
   `post_attention_layernorm_moe_gen` (→ SwiGLU gate/up). `to_add_out`'s input is the attention output
   (no preceding norm). This is exactly where M3-3b (norm→quant fusion) applies.

### 0.4 Guiding principles
- **Quant ↔ kernel are coupled through the numeric-format contract.** Quantization picks the format;
  the kernel cashes in the speed. A quantized op without a matching kernel can be *slower* (dequant
  overhead). On this compute-bound workload, weight-only is pointless; activation-quant + tensor-core
  kernel is the win.
- **Fusion deletes operator boundaries — so freeze the data-flow contracts BEFORE building, and keep
  an un-fused, already-verified stack to diff against.** Fusion is a silent-correctness-bug hotspot;
  every fuse must be numerically diffed vs the un-fused reference. This is why M1/M2 land **un-fused
  and verified first**, and M3 only welds the seams afterward.
- **Each module ships as its OWN copied pipeline + benchmark — never by editing a shared tool.** Exactly
  as the cache phase forked `*_taylorseer` copies of the transformer / pipeline / benchmark off the base
  files, each module is built by **copying the previous stage's pipeline + benchmark and adapting it**:
  M1 copies the cache pipeline; M2 copies the cache+M1 pipeline; M3 copies cache+M1+M2. Every prior stage
  stays intact and independently runnable, and each stage gets an isolated, reproducible harness to
  measure and diff against.

### 0.5 Measurement & evaluation (identical to the cache phase; reference = the cache pipeline)
Every module is judged by the **exact same evaluation pipeline used in the cache phase** — with **one
change: the comparison and speed baseline is the cache pipeline's own result (`i2_o1_all_w1_c5`), not
the cache-free BF16 reference.** Scoring each new module against the cache output (rather than pristine
BF16) deliberately **relaxes the bar** — a module only has to avoid degrading *further* than the cache
already does — so the optimization loop keeps converging instead of chasing an unreachable pixel-match.

Per-module loop (run on vr-1/vr-2, 8× H200, one prompt per GPU):
1. **Generate** the 8 final8 videos (ids `006 014 028 039 048 049 068 079`) with the module ON, on the
   cached pipeline, **identical gen settings** to the baseline (720p, 189 frames, 35 steps, guidance
   6.0, flow-shift 10.0, bf16, seed 1234) via **that module's own copied pipeline + benchmark** (the
   copy convention in §0.4 — do not overload the shared TaylorSeer tool). Archive to HF
   `oliveryanzuolu/data`, download locally.
2. **Speed:** per-prompt `speedup = cache_pipeline pipeline_call_seconds / (cache+module)
   pipeline_call_seconds`, then mean over the 8. (Absolute headline = ×1.605 for the cache gain.)
3. **Reference videos = the existing `i2_o1_all_w1_c5` final8 outputs** (already generated in the cache
   phase — `cosmos_taylorseer_final8_frontier/0615b_wc_rescue/final8_matrix/i2_o1_all_w1_c5/`); no need
   to regenerate them.
4. **Build comparison sheets** per prompt with the fixed ffmpeg recipe — **LEFT = cache-pipeline output
   (reference), RIGHT = cache+module output**: each frame scaled 512×288 and hstacked → 1024-wide rows;
   `sheetA_early` tiles frames 0/24/48/72 (`tile=1x4` → 1024×1152 PNG), `sheetB_late` tiles
   96/120/144/168; plus a full-res 2560×720 side-by-side mp4 for human spot-checks. (Same recipe as the
   cache phase, so scores stay comparable across phases.)
5. **Score with subagents — one per prompt, NOT codex.** The orchestrator **never opens the images**
   (keeps its context clean); all visual judgment is delegated. Each subagent reads that prompt's two
   sheets and returns strict JSON scoring how CLOSE the RIGHT (module) is to the LEFT (reference): four
   dimensions **picture_quality / structural_integrity / motion_temporal / semantic_fidelity**, each
   0–10 (10 = indistinguishable from the reference); anchors **9-10 indistinguishable · 7-8 minor but
   acceptable · 5-6 borderline · 3-4 clear artifacts (not acceptable) · 0-2 severe collapse/drift**;
   plus a holistic **overall** (0–10) and **acceptable** (bool — true only if overall ≥ 7 and no dim ≤ 4).
   Minor softness is fine; smearing / ghosting / warping / melting / structure-collapse / identity- or
   content-drift are not. **All 8 prompts scored equally.**
6. **Aggregate & gate:** mean overall, min overall, acceptable count (X/8). **PASS = all 8 acceptable
   (8/8)** — the same bar as the cache phase. Persist to `_compare/<module>/qa_scores.json`.

**Incremental order:** cache → +M1 → +M1+M2 → +M1+M2+M3, scoring at each step (gives attribution + the
time-split for free; no standalone profiling). **SmoothQuant calibration uses the full 100 structured
prompts** (`assets/cosmos3_t2v_prompts_seed100_upsampled/prompts/`, the 0614d set); **final8 is the
held-out QA subset** — calibrate on 100, judge on 8.

---

## M1 — W8A8 FP8 linears: **SmoothQuant + TorchAO** (tuning module)

**Goal:** FP8 W8A8 on the 252 generation-pathway linears, calibration-smoothed so even the
outlier-heavy channels quantize cleanly, **without** foreclosing later fusion.

**Method**
1. **SmoothQuant smoothing (offline, calibrated on the 100 structured prompts).** Collect per-channel
   activation magnitudes on the gen-pathway linear inputs; compute the SmoothQuant migration scale and
   **fold it into the preceding RMSNorm weight** (`input_layernorm_moe_gen`,
   `post_attention_layernorm_moe_gen`). Folding into the norm is free and **synergizes with M3-3b**
   (the norm→quant prologue is already being fused). `to_add_out` (no preceding norm) takes the scale
   on its weight directly.
2. **TorchAO FP8 on top** — `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())` via
   `quantize_(transformer, cfg, filter_fn=…)` (use the `version=2` config class; the string API was
   removed in torchao 0.16). **E4M3 both sides, per-token dynamic activation scale + per-channel weight
   scale, `use_fast_accum=True`.** Lowers to `torch._scaled_mm` rowwise on Hopper (NT layout: pass
   `B.t()`).
3. **torch.compile is mandatory** (`compile_repeated_blocks(fullgraph=True)`), with **constant-batch
   CFG** (run cond+uncond as one doubled batch) to avoid B↔2B recompiles. Without compile the eager
   FP8-cast overhead cancels the GEMM win.
4. **Bring-up check (part of implementing M1, not the QA loop):** confirm from the trace that FP8
   actually lowered to `torch._scaled_mm` — **not** fake-quant (Q/DQ inserted but the matmul still BF16
   → accuracy hit, zero speed) — and that `torch.compile` didn't silently fall back to eager. Do this
   once when the kernel is wired up; a "≈0 speedup" is debugged here, before any QA run.

**Why TorchAO and not TransformerEngine / a custom kernel:** TorchAO is a tensor-subclass over
`nn.Linear` — it keeps the region as ordinary ATen ops, so (a) Inductor fuses RMSNorm→amax→FP8-cast
into one prologue kernel **for free** (this is most of M3-3b), and (b) we can later add a deeper fused
kernel as an Inductor pattern-match pass **without touching attention/cache**. TransformerEngine's
`te.LayerNormLinear` *is* a ready-made norm→GEMM fusion but it **graph-breaks under compile** and its
DelayedScaling amax history **desyncs when the cache skips steps** — it freezes the very fusion freedom
we want. Hand-writing CUTLASS/Triton now is premature (do it only if Phase-0/M3 profiling proves the
two-kernel split is binding). ModelOpt's PyTorch path is fake-quant (no in-framework speedup) — not a
runtime option.

**What to tune in M1**
| Knob | Range / default | Effect |
|---|---|---|
| SmoothQuant **migration strength α** | 0.5 default; sweep ~0.4–0.8 | balances how much outlier difficulty moves activation→weight; too high hurts weights, too low leaves activation outliers |
| **Layer inclusion / BF16 keep-list** | start: all 252; demote on regression | `mlp_moe_gen` SwiGLU is the GEMM-FLOP bulk (quantize first); `add_q/k/v_proj` are small + outlier-sensitive (quantize + QA, keep BF16 if FID/LPIPS regress); protect any timestep/modulation-adjacent linear if it regresses |
| FP8 **granularity** | PerRow (fixed best practice) | per-token act + per-channel weight; per-tensor is brittle — not really a tuning axis, stated for completeness |

(Activation quant itself is **calibration-free / dynamic** — SmoothQuant is the *only* calibrated part,
and it uses the 100 structured prompts.)

**Acceptance:** final8 QA no-regression vs the **cache-pipeline reference** (§0.5) + measured speedup over
the cache pipeline.

**Known traps:** `RMSNorm + compile + float8 rowwise` has a high-priority NaN bug (pytorch #150859) —
test for it. torchao API break #13286 (diffusers). Compile is non-optional.

---

## M2 — Sparse FP8 attention: **SpargeAttn (built on SageAttention2++)** (tuning module)

**Goal:** replace the gen full-attention (`is_causal=False`) with a quantized + block-sparse kernel,
installed behind `dispatch_attention_fn` — **no model surgery**.

**Why SpargeAttn:** it is the THU-ml SageAttention group's training-free sparse attention that layers
**block-sparsity on top of SageAttention2++'s quantized kernels** (INT8 QK^T + FP8 PV). On Hopper it
has an FP8 path (needs CUDA ≥12.3), **no Blackwell/FP4 requirement**, **accepts raw q/k/v with no
RoPE/QK-norm coupling** (exactly our case — RoPE/QK-norm already applied outside), supports
`is_causal=False`, and is evaluated on long-sequence **video** diffusion. It is a strict superset of
SageAttention2 (quantization × sparsity vs quantization-only). Repo: `github.com/thu-ml/SpargeAttn`.
(Do **not** use SpargeAttention**2** / arXiv:2602.13515 — it is a *trainable* redesign requiring a
distillation fine-tune of the model; that crosses the "kernel only" boundary. Defer as a later option.)

**Method — phased, to isolate the risky part**
- **Phase A — quant-only (guaranteed win):** run SpargeAttn with sparsity ≈ 0 (≈ SageAttention2++).
  Validate near-lossless on final8. The quantized speedup is the safe part (QK-norm helps).
- **Phase B — add sparsity, conservatively:** enable block-sparsity, **keep the self-similarity
  "fix-block" guard ON**, and ramp the sparsity target up, **re-running the §0.5 final8 QA at each
  step**. (Sparsity quality is content-dependent — busy/fast content has flatter attention with fewer
  skippable blocks — so let the standard equal-weight final8 QA catch regressions rather than
  special-casing prompts.)
- **Phase C — per-layer tuned (optional):** calibrate per-layer thresholds to an attention L1-error
  budget if the global setting leaves speed on the table.

**Bring-up check (part of implementing M2, not the QA loop):** before trusting any speed number, confirm
the **SpargeAttn CUDA op is actually in the trace** — not a silent fallback to PyTorch SDPA.

**What to tune in M2**
| Knob | Range / default | Effect |
|---|---|---|
| **τ** (CDF / cumulative-prob threshold) | low → conservative | how aggressively whole QK^T blocks are skipped; the primary sparsity↔quality dial |
| **θ** (self-similarity threshold for "fix blocks") | keep guard ON | blocks with low internal similarity are always computed (safety); turning this off collapses quality |
| **λ** (softmax-skip threshold) | conservative default | skips PV at warp granularity when a row's local max lags — secondary dial |
| **block size** | 128×64 default | kernel tiling; usually leave default |
| **no-tune fast path** | `topk` fraction (e.g. 0.5) | single-knob alternative to τ/θ/λ — good for Phase B bring-up |
| **target sparsity ramp** | start low | the headline speed knob; bounded by high-motion QA |
| **Hilbert token permutation** | on/off | raises block similarity (more sparsity) **but reorders tokens** — must stay consistent with the 3-D order / mRoPE / TaylorSeer cache (see Integration note); off if not |
| per-layer vs global thresholds | global first | per-layer calibration recovers extra speed at calibration cost |

**Integration with the cache:** because the TaylorSeer cache reuses the layer-delta on predicted steps,
the sparse attention (like M1's GEMMs) **runs only on full steps automatically** once M2 is built inside
the cached transformer — no extra cache-gating. The one real interaction risk is the **Hilbert
permutation** (above): its token reordering must stay consistent across steps with the fixed 3-D (T×H×W)
order, the mRoPE positions, and TaylorSeer's by-position residual cache — if it can't, **disable Hilbert**.

**Acceptance:** final8 QA no-regression vs the cache-pipeline reference (§0.5, all 8 equal) + measured
attention + e2e speedup.

---

## M3 — Kernel fusion (**correctness-verification module**, little tuning)

Welds the seams between M1 and M2 to remove HBM round-trips. **No new approximation is introduced —
the job is to fuse without changing numerics**, so every fuse is **numerically + visually diffed
against the un-fused M1+M2 stack** (silent bugs: wrong scale, flipped rotate-half, inconsistent
permutation only surface against that reference).

**First freeze the two contracts** (deferred from the start, since M1/M2 land un-fused): (i) RoPE/QK-norm
placement — keep in the processor (contract A) or fold into the attention prologue (contract B = 3c);
(ii) the torch.compile boundary (what Inductor fuses vs what is hand-written). All of M3 depends on these.

- **3a — free compile fusion:** regional-compile the repeated MoT block; Inductor fuses residual-adds,
  RMSNorm, SwiGLU mul, RoPE elementwise *between* the custom kernels. Cheap, eats half the
  fragmentation tax.
- **3b — norm→quant prologue:** **mostly already free** via M1's TorchAO+compile (RMSNorm→amax→FP8-cast
  fuse into one kernel). The deeper step — folding the cast *into* the GEMM mainloop — stays opaque
  behind `_scaled_mm`; only do it **if the measured prologue/GEMM two-kernel split is binding**, and
  then as an **Inductor pattern-match pass** (precedent: vLLM PR #10906), not model surgery.
- **3c — qknorm+rope→attn prologue (the one real hand-fuse):** fold QK-norm + RoPE into SpargeAttn's
  CUDA load stage (contract B from Phase 0). Touches CUDA source — diff against un-fused every time.

**Acceptance:** bit-for-bit-close vs un-fused stack on a few steps + final8 QA unchanged + net speedup.

---

## Sequence & dependency summary
```
M1  SmoothQuant(calib=100 prompts) + TorchAO FP8 W8A8 on 252 gen linears   ── un-fused, verified, trace-gated
M2  SpargeAttn (SageAttention2++ kernels), gen is_causal=False, phased A→B→C ── un-fused, verified, trace-gated
   │   (M1, M2 independent — can build in parallel; both diffed vs BF16+cache baseline)
M3  fusion = 3a compile-glue(free) + 3b norm→quant(mostly free) + 3c qknorm+rope→attn(hand)  ── diff vs un-fused
```
Everything is measured on the cached pipeline, so there is **no separate integration phase**: M1's
GEMMs and M2's attention run only on full steps automatically, and the final **+fusion** increment IS
the combined high-motion measurement on the full cache+M1+M2+M3 stack.
**Tuning lives in M1 (α, BF16 keep-list) and M2 (τ/θ/λ, sparsity ramp, Hilbert); M3 is correctness
verification.**

## Key external references
- FP8 formats arXiv:2209.05433 · SVDQuant (diffusion outliers) arXiv:2411.05007 · SmoothQuant
  arXiv:2211.10438 · SpargeAttn arXiv:2502.18137 (repo github.com/thu-ml/SpargeAttn) ·
  SageAttention2 arXiv:2411.10958 · TorchAO Float8 docs + `_scaled_mm` rowwise · RMSNorm+rowwise NaN
  pytorch#150859 · vLLM RMSNorm→fp8 Inductor pass PR#10906.
