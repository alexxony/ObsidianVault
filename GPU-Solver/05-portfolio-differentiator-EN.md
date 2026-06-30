---
title: GPU-Solver — Portfolio Differentiator (EN)
created: 2026-06-28
tags: [gpu-solver, portfolio, differentiator, english]
status: active
---

# GPU-Solver — What I Proved, and What I Didn't

> 🎯 Single portfolio narrative for an AX / agentic-systems audience.
> Tone: not *"I built impressive infrastructure"* but *"I form a hypothesis, isolate the mechanism
> with a controlled experiment, and when it doesn't work I prove it with measurement."*
> Korean original: [[05-portfolio-differentiator]]. Hub: [[GPU-Solver-MOC]].

## One line

I built a loop where an LLM agent **evolves its own classification rules from profiling
measurements**, and proved that evolution mechanism with a **controlled ON/OFF experiment**.
The performance gain was **measured to be null** and attributed to an environment limit —
stated honestly as future work.

## What I proved (claimable)

### 1. Measurement-feedback rule-evolution meta-loop — mechanism proven (✅)

Prior art (CUDAMaster, arXiv 2603.07169) already implements the "deterministic rule label →
LLM rewrite → measure-verify" pipeline. **My contribution = the classification rule table itself
evolves from measurement feedback.** All 6 surveyed prior systems use static rules.

**Evidence (real A100, sigmoid, 14 rounds, `run_gain_compare.py`):**
- Evolution **ON** = a wrong `fp32_no_tensorcore` rule fired 4 times, failed, was retired →
  auto-switched to `memory_bound_fusable`.
- Evolution **OFF** = the same fp32 rule misfired 6 times forever (fake signal).
- = "a wrong static rule gets retired by measurement and replaced by the right one" — closed on real GPU.

### 2. Self-validation via ablation (✅ — the core selling point)

The ON/OFF above is not a demo. It is an **ablation that isolates which component (evolution)
produces the effect.** The same variant queue was fed fairly to both tracks, so the difference
is controlled to the single variable of evolution presence/absence.

### 3. The loop reaches a faster kernel by measurement — gain layer first breakthrough (✅ 2026-06-29)

| Problem | Attempt | Result |
|---|---|---|
| **matmul (4096² fp32)** | loop R0(fp32)→`fp32_no_tensorcore` fires→R1(TF32) | ✅ **9.6ms→1.5ms = 6.4× gain** |
| sigmoid | `--latency` 12R, BLOCK variants | null — memory-bound ceiling (no headroom) |
| groupnorm | split-parallel (3 algorithms) | null — DRAM BW ceiling |
| llama | TF32 OFF vs ON | null — attention (flash) 54% dominates, matmul 23% not bottleneck |

**matmul: the loop autonomously reached a 6.4×-faster kernel via measure→hypothesis→rewrite.**
First demonstration of "measure → hypothesis → rewrite → faster kernel." sigmoid/groupnorm nulls are
real ceilings (memory-bound); llama is attention-dominated — **not a loop defect, a problem property.**

> ⚠️ **Measurement integrity:** matmul first read 95ms parity ("environment limit"), but the cause was
> a **bug** — `_profile_ncu` measured only 1 kernel (`--launch-count 1`). Fixing it to sum all-kernel
> durations revealed the 6.4×. Cross-checked by ncu kernel names (sgemm vs tensorop) and theoretical
> FLOP. = a bug caught and corrected by measurement.

### 4. Environment (chip) as a first-class rule input — chip guard (✅ 2026-06-30, real T4)

The optimal prescription for the *same signal* depends on the chip. fp32 matmul "slow" → A100: enable TF32;
T4: no TF32. This dependency goes into the rule `cond` as a first-class guard (`chip_cap`) — but the rule
**content is not authored by a human; measurement→evolution discovers it** (the golden rule).

| Chip | fp32 matmul signal → fired rule | Result |
|---|---|---|
| A100 (sm_80) | `fp32_no_tensorcore` fires | TF32 → 6.4× gain |
| **T4 (sm_75)** | guard (`chip_cap="tf32"`) **blocks** → `reg_pressure` | bogus TF32 hypothesis blocked upfront |

**Changing only the Context (chip) flips the fired rule — no rule edited.** Demonstrates "the user sets only
the stage (environment); the optimal rule is discovered by measurement." CHIP_CAPS (NVIDIA public spec) filters
the bogus hypothesis. Prior art (CUDAMaster/CudaForge) is fixed to CUDA + a specific GPU → **chip portability
is a genuine open slot.**

**Dual defense:** chip guard = block *known* bogus hypotheses upfront (declarative) + evolution retire =
discard *unknown* bogus hypotheses post-hoc (by measurement).

> ⚠️ **Honesty:** T4 `reg_pressure` matches its cond (`occ<0.5 & reg>64`) but is bogus — compute is saturated
> (96.7%), so it cannot help. Evolution should retire it, but `CONVERGE_PATIENCE=3`==`RETIRE_AFTER_FAILS=3`
> makes convergence fire before retire → T4 retire unobserved (retire itself proven on sigmoid). Claim only the
> 1 measured chip (T4); N-chip generalization is unproven.

## What I did NOT prove (never claim)

- ❌ **Evolution superiority ON>OFF.** On matmul the first fired rule is correct → no retire needed → ON≈OFF.
  Evolution's benefit is shown separately by sigmoid's misfire-retire. **The gain axis and the evolution
  axis are not demonstrated together on a single problem.**
- ❌ **Multi-problem generalization.** gain=matmul (1 problem), evolution=sigmoid (1 problem).
- ⚠️ **Two axes, separated:** loop improves (gain, matmul ✅) + evolution beats static (sigmoid ✅) =
  each a single-problem demonstration. Both-on-one-problem + multi-problem generalization = future work.
  - **Both-on-one confirmed structurally hard (by measurement).** Tried a deliberately uncoalesced Triton
    matmul to get "wrong rule fires → retire → right rule → gain," but the seed rules fit so well that the
    first fired rule is always correct → no retire. **Wrong-rule-fires-naturally = memory-bound (gain
    ceiling) vs right-rule = compute (no retire) = mutually exclusive.** Both on one problem without staging
    = not possible. The limit itself was established by measurement.

> **Cross this boundary and you get downgraded.** Do not say "a system that auto-optimizes GPU kernels."
> It collapses under a speedup question. Keep claims confined to the **mechanism / methodology layer.**

## Audience positioning

**AX / agentic systems (primary audience):**
> "An agentic system with an LLM in a measurement-feedback loop. The core = the agent's
> classification rules self-evolve from profiling results. I proved the mechanism with an ablation,
> and I quantified the limit of the performance gain by measurement."

**Compiler / perf experts (secondary, defensive):**
> Concede the absence of performance numbers first. Present "null in this environment, cause is the
> measurement environment" as data. = Trust through measurement integrity, not exaggeration.
> Do not enter the speedup competition.

## Why this holds up even though I'm a beginner

The differentiator is not lines of code or harness sophistication — it's **methodology**:
1. Form a hypothesis (if the rule table evolves, there's a gain).
2. Isolate the mechanism with a controlled experiment (ON/OFF ablation).
3. When it fails, prove it fails with measurement (3 problems null, cause attributed).
4. Separate claims by layer (mechanism vs gain, single-problem vs generalization).

= A senior-researcher way of thinking. The harness (git-mailbox, ncu pipeline) is a tool that
supports this thinking — not the thing to show off.

## Reproducibility (when the environment is ready)

The gain-measurement infrastructure is ready — retry as-is once the environment is PoC-grade (1.4ms):
- `loop/run_gain_hypcond.py` = hypothesis-conditional callback driver (selects code by fired rule label).
- Full retry procedure = [[PROGRESS]] §performance-gain infrastructure.

## Links

- [[GPU-Solver-MOC]] — document hub
- [[04-multiproblem-round-design]] — A/B experiment detail (mechanism evidence)
- [[02-prior-art-survey]] — prior-art comparison, differentiator check
- [[PROGRESS]] — resume entry point
