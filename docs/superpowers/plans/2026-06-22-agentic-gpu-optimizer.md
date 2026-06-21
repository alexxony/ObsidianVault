# Agentic GPU Kernel Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agent loop that optimizes GPU kernels via authored trace→hypothesis rules (LLM rewrites code only), producing a two-track result: percentile-improvement curves for optimizable problems and STOP-accuracy for saturated ones.

**Architecture:** A Python harness orchestrates a round loop: LLM generates a kernel → correctness gate → submit/measure → **Trace Parser** normalizes metrics → **Hypothesis Engine** matches authored seed rules to pick one bottleneck + targeted prompt → loop. A **Rule Evolver** meta-loop updates rule confidence from the run ledger. The three authored components (Parser, Engine, Evolver) are decoupled from external services and TDD'd against fixtures; LeetGPU and the LLM are thin, swappable adapters behind interfaces.

**Tech Stack:** Python 3.11+, pytest, dataclasses, JSON ledger. LLM via a single API (DeepSeek or Codex — decided in Task 1). Measurement via LeetGPU API and/or local `ncu`/`nvprof` (decided in Task 0).

## Global Constraints

- Language: Python 3.11+. No heavy frameworks — stdlib + pytest only for the harness. (Generated *kernels* are CUDA C++; the harness never writes CUDA.)
- The three authored components (Trace Parser, Hypothesis Engine, Rule Evolver) MUST be unit-testable with zero network/GPU access — all external I/O goes through adapter interfaces injected at construction.
- Every authored rule carries a one-line `rationale` string. A rule without a rationale is a plan failure.
- Problem set is NEVER filtered by result. The system classifies; the human does not pre-select. (Spec §4.)
- No new runtime dependency may be added for what ~20 lines of stdlib does.
- Ledger is append-only JSON Lines (`.jsonl`), one record per round.
- Korean is fine in notes/docs; code identifiers and commit messages in English.

---

### Task 0: Measurement feasibility spike (BLOCKING — do first, no code merge)

**Why:** Spec §6 top risk. If LeetGPU's free tier exposes no percentile/trace, and local `ncu` is unavailable, the hypothesis engine has no signal and the whole project is invalid. Resolve before building.

**Files:**
- Create: `GPU-Solver/docs/00-measurement-feasibility.md`

- [ ] **Step 1: Probe LeetGPU free-tier signal**

Manually (or via curl/browser) submit one known kernel to LeetGPU. Record: does the response include a runtime number? a percentile? any trace (Perfetto/metrics)? Is any of it gated behind Pro?

- [ ] **Step 2: Probe local measurement fallback**

Run: `which ncu nvprof nvidia-smi 2>/dev/null; nvcc --version 2>/dev/null`
Record whether an NVIDIA toolchain + a GPU is reachable locally (incl. cloud notebooks). Note: the *target* chip is A100 (spec); local need only be CUDA-capable enough to produce metrics for relative comparison.

- [ ] **Step 3: Decide the metric source and write it down**

In `00-measurement-feasibility.md` record one of:
- **A)** LeetGPU free tier gives percentile + enough metrics → primary source = LeetGPU.
- **B)** LeetGPU gives only pass/fail → primary source = local `ncu` for the signal dict; LeetGPU for percentile only (if any).
- **C)** Neither works → STOP. Project blocked; escalate to user with options (buy Pro / rent GPU / pivot).

Also record the concrete field names available (e.g. `sm__throughput`, `dram__bytes`) — Task 4 maps these into the signal dict.

- [ ] **Step 4: Commit**

```bash
git add GPU-Solver/docs/00-measurement-feasibility.md
git commit -m "docs: measurement feasibility spike — decide metric source"
```

**GATE:** Do not proceed to Task 2+ unless Step 3 chose A or B. The signal field names from Step 3 are consumed by Task 4.

---

### Task 1: Project skeleton + LLM adapter interface + bake-off

**Files:**
- Create: `GPU-Solver/src/gpuopt/__init__.py`
- Create: `GPU-Solver/src/gpuopt/llm.py`
- Create: `GPU-Solver/tests/test_llm_adapter.py`
- Create: `GPU-Solver/pyproject.toml`
- Create: `GPU-Solver/docs/01-llm-bakeoff.md`

**Interfaces:**
- Produces:
  - `class LLMAdapter(Protocol)`: method `generate(self, prompt: str) -> str` (returns kernel source).
  - `class FakeLLM(LLMAdapter)`: constructed with `FakeLLM(responses: list[str])`; `generate` pops next response. For tests.
  - `class APILLM(LLMAdapter)`: constructed with `APILLM(model: str, api_key: str)`; real call. (Body may be thin; not unit-tested against network.)

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "gpuopt"
version = "0.1.0"
requires-python = ">=3.11"
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write failing test for FakeLLM**

```python
# tests/test_llm_adapter.py
from gpuopt.llm import FakeLLM

def test_fake_llm_returns_scripted_responses():
    llm = FakeLLM(responses=["kernel_v1", "kernel_v2"])
    assert llm.generate("p1") == "kernel_v1"
    assert llm.generate("p2") == "kernel_v2"
```

- [ ] **Step 3: Run test, verify it fails**

Run: `cd GPU-Solver && pytest tests/test_llm_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.llm'`

- [ ] **Step 4: Implement llm.py**

```python
# src/gpuopt/llm.py
from typing import Protocol

class LLMAdapter(Protocol):
    def generate(self, prompt: str) -> str: ...

class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0
    def generate(self, prompt: str) -> str:
        r = self._responses[self._i]
        self._i += 1
        return r

class APILLM:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    def generate(self, prompt: str) -> str:
        # ponytail: real HTTP call filled in when an API key exists;
        # network path is not unit-tested. Use the provider SDK or requests.
        raise NotImplementedError("wire to provider API before live runs")
```

- [ ] **Step 5: Run test, verify it passes**

Run: `cd GPU-Solver && pytest tests/test_llm_adapter.py -v`
Expected: PASS

- [ ] **Step 6: Bake-off note (decide the model)**

Manually run 2-3 Easy problems through DeepSeek and Codex by hand. Record pass rate + subjective quality in `docs/01-llm-bakeoff.md`. Pick one. (No code; this picks the `model` string for `APILLM`.)

- [ ] **Step 7: Commit**

```bash
git add GPU-Solver/pyproject.toml GPU-Solver/src/gpuopt/ GPU-Solver/tests/test_llm_adapter.py GPU-Solver/docs/01-llm-bakeoff.md
git commit -m "feat: project skeleton + LLM adapter interface with bake-off"
```

---

### Task 2: Data model (Signal, Hypothesis, RoundRecord)

**Files:**
- Create: `GPU-Solver/src/gpuopt/model.py`
- Create: `GPU-Solver/tests/test_model.py`

**Interfaces:**
- Produces:
  - `@dataclass Signal`: fields `occupancy: float, reg_per_thread: int, l2_hit: float, global_load_eff: float, stall_reason: str, achieved_bw_pct: float, compute_tput: float`.
  - `@dataclass Hypothesis`: fields `bottleneck: str, prompt: str, rationale: str, rule_id: str`.
  - `@dataclass RoundRecord`: fields `problem_id: str, round_idx: int, kernel_src: str, passed: bool, percentile: float | None, signal: Signal | None, hypothesis: Hypothesis | None, outcome: str` (`"improved" | "no_change" | "worse" | "stop" | "fail"`).
  - `RoundRecord.to_json(self) -> str` and `RoundRecord.from_json(s: str) -> RoundRecord`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_model.py
from gpuopt.model import Signal, Hypothesis, RoundRecord

def test_roundrecord_json_roundtrip():
    sig = Signal(0.38, 96, 0.41, 0.55, "sync", 0.30, 0.15)
    hyp = Hypothesis("reg_pressure", "lower threads", "occupancy limited by registers", "r_reg")
    rec = RoundRecord("p_sigmoid", 0, "src", True, 42.0, sig, hyp, "improved")
    back = RoundRecord.from_json(rec.to_json())
    assert back == rec

def test_roundrecord_handles_none_signal():
    rec = RoundRecord("p1", 0, "src", False, None, None, None, "fail")
    assert RoundRecord.from_json(rec.to_json()) == rec
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.model'`

- [ ] **Step 3: Implement model.py**

```python
# src/gpuopt/model.py
from dataclasses import dataclass, asdict
import json

@dataclass
class Signal:
    occupancy: float
    reg_per_thread: int
    l2_hit: float
    global_load_eff: float
    stall_reason: str
    achieved_bw_pct: float
    compute_tput: float

@dataclass
class Hypothesis:
    bottleneck: str
    prompt: str
    rationale: str
    rule_id: str

@dataclass
class RoundRecord:
    problem_id: str
    round_idx: int
    kernel_src: str
    passed: bool
    percentile: float | None
    signal: Signal | None
    hypothesis: Hypothesis | None
    outcome: str

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "RoundRecord":
        d = json.loads(s)
        sig = Signal(**d["signal"]) if d["signal"] is not None else None
        hyp = Hypothesis(**d["hypothesis"]) if d["hypothesis"] is not None else None
        d["signal"] = sig
        d["hypothesis"] = hyp
        return cls(**d)
```

- [ ] **Step 4: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add GPU-Solver/src/gpuopt/model.py GPU-Solver/tests/test_model.py
git commit -m "feat: data model — Signal, Hypothesis, RoundRecord with JSON roundtrip"
```

---

### Task 3: Run Ledger (append-only JSONL)

**Files:**
- Create: `GPU-Solver/src/gpuopt/ledger.py`
- Create: `GPU-Solver/tests/test_ledger.py`

**Interfaces:**
- Consumes: `RoundRecord` from Task 2.
- Produces:
  - `class Ledger`: `Ledger(path: str)`; `append(self, rec: RoundRecord) -> None`; `all(self) -> list[RoundRecord]`; `for_rule(self, rule_id: str) -> list[RoundRecord]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_ledger.py
from gpuopt.model import RoundRecord
from gpuopt.ledger import Ledger

def _rec(rule_id, outcome):
    from gpuopt.model import Hypothesis
    hyp = Hypothesis("b", "p", "why", rule_id)
    return RoundRecord("p1", 0, "src", True, 50.0, None, hyp, outcome)

def test_ledger_append_and_read(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    led.append(_rec("r_a", "improved"))
    led.append(_rec("r_b", "no_change"))
    assert len(led.all()) == 2

def test_ledger_filters_by_rule(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    led.append(_rec("r_a", "improved"))
    led.append(_rec("r_b", "worse"))
    led.append(_rec("r_a", "no_change"))
    assert len(led.for_rule("r_a")) == 2
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.ledger'`

- [ ] **Step 3: Implement ledger.py**

```python
# src/gpuopt/ledger.py
from gpuopt.model import RoundRecord

class Ledger:
    def __init__(self, path: str):
        self.path = path

    def append(self, rec: RoundRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")

    def all(self) -> list[RoundRecord]:
        try:
            with open(self.path, encoding="utf-8") as f:
                return [RoundRecord.from_json(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def for_rule(self, rule_id: str) -> list[RoundRecord]:
        return [r for r in self.all()
                if r.hypothesis is not None and r.hypothesis.rule_id == rule_id]
```

- [ ] **Step 4: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_ledger.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add GPU-Solver/src/gpuopt/ledger.py GPU-Solver/tests/test_ledger.py
git commit -m "feat: append-only JSONL run ledger with per-rule filtering"
```

---

### Task 4: Trace Parser (raw metrics → Signal)

**Files:**
- Create: `GPU-Solver/src/gpuopt/parser.py`
- Create: `GPU-Solver/tests/test_parser.py`
- Create: `GPU-Solver/tests/fixtures/trace_sigmoid.json`
- Create: `GPU-Solver/tests/fixtures/trace_uncoalesced.json`

**Interfaces:**
- Consumes: `Signal` from Task 2; raw field names from Task 0 Step 3.
- Produces:
  - `def parse_trace(raw: dict) -> Signal`. Maps raw metric keys → Signal fields, with safe defaults (0.0 / 0 / "none") for missing keys.

> NOTE: the raw key names below are placeholders matching common `ncu` metrics. If Task 0 recorded different field names, substitute them here — the test fixtures and the mapping must agree.

- [ ] **Step 1: Create fixtures**

```json
// tests/fixtures/trace_sigmoid.json  (saturated case)
{"sm__occupancy": 0.6, "registers_per_thread": 32, "l2_hit_rate": 0.9,
 "global_load_efficiency": 1.0, "dominant_stall": "none",
 "dram_throughput_pct": 0.85, "sm_throughput_pct": 0.15}
```

```json
// tests/fixtures/trace_uncoalesced.json  (optimizable case)
{"sm__occupancy": 0.7, "registers_per_thread": 40, "l2_hit_rate": 0.5,
 "global_load_efficiency": 0.4, "dominant_stall": "memory",
 "dram_throughput_pct": 0.45, "sm_throughput_pct": 0.3}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_parser.py
import json, pathlib
from gpuopt.parser import parse_trace
from gpuopt.model import Signal

FIX = pathlib.Path(__file__).parent / "fixtures"

def _load(name):
    return json.loads((FIX / name).read_text())

def test_parse_sigmoid_saturated():
    sig = parse_trace(_load("trace_sigmoid.json"))
    assert sig.achieved_bw_pct == 0.85
    assert sig.global_load_eff == 1.0
    assert sig.compute_tput == 0.15

def test_parse_missing_keys_get_defaults():
    sig = parse_trace({})
    assert isinstance(sig, Signal)
    assert sig.achieved_bw_pct == 0.0
    assert sig.stall_reason == "none"
```

- [ ] **Step 3: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.parser'`

- [ ] **Step 4: Implement parser.py**

```python
# src/gpuopt/parser.py
from gpuopt.model import Signal

def parse_trace(raw: dict) -> Signal:
    g = raw.get
    return Signal(
        occupancy=float(g("sm__occupancy", 0.0)),
        reg_per_thread=int(g("registers_per_thread", 0)),
        l2_hit=float(g("l2_hit_rate", 0.0)),
        global_load_eff=float(g("global_load_efficiency", 0.0)),
        stall_reason=str(g("dominant_stall", "none")),
        achieved_bw_pct=float(g("dram_throughput_pct", 0.0)),
        compute_tput=float(g("sm_throughput_pct", 0.0)),
    )
```

- [ ] **Step 5: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_parser.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add GPU-Solver/src/gpuopt/parser.py GPU-Solver/tests/test_parser.py GPU-Solver/tests/fixtures/
git commit -m "feat: trace parser — raw metrics to normalized Signal with defaults"
```

---

### Task 5: Hypothesis Engine (seed rules → one Hypothesis)

**Files:**
- Create: `GPU-Solver/src/gpuopt/rules.py`
- Create: `GPU-Solver/src/gpuopt/engine.py`
- Create: `GPU-Solver/tests/test_engine.py`

**Interfaces:**
- Consumes: `Signal`, `Hypothesis` from Task 2.
- Produces:
  - `rules.py`: `SEED_RULES: list[Rule]` where `@dataclass Rule` has `id: str, match: Callable[[Signal], bool], bottleneck: str, prompt: str, rationale: str, priority: int`.
  - `engine.py`: `def diagnose(sig: Signal, rules: list[Rule], disabled: set[str] = frozenset()) -> Hypothesis`. Returns the highest-priority matching, non-disabled rule's hypothesis. If none match (or the only match is `memory_saturated`), returns a STOP hypothesis (`bottleneck="memory_saturated"` or `bottleneck="none"`, `prompt="STOP"`).

- [ ] **Step 1: Write failing test**

```python
# tests/test_engine.py
from gpuopt.model import Signal
from gpuopt.rules import SEED_RULES
from gpuopt.engine import diagnose

def _sig(**kw):
    base = dict(occupancy=0.7, reg_per_thread=32, l2_hit=0.9,
               global_load_eff=1.0, stall_reason="none",
               achieved_bw_pct=0.3, compute_tput=0.3)
    base.update(kw)
    return Signal(**base)

def test_saturated_returns_stop():
    h = diagnose(_sig(achieved_bw_pct=0.85, global_load_eff=1.0), SEED_RULES)
    assert h.prompt == "STOP"
    assert h.bottleneck == "memory_saturated"

def test_uncoalesced_detected():
    h = diagnose(_sig(global_load_eff=0.4), SEED_RULES)
    assert h.bottleneck == "uncoalesced"
    assert "tiling" in h.prompt.lower() or "coalesce" in h.prompt.lower()
    assert h.rationale  # rationale is mandatory

def test_disabled_rule_is_skipped():
    rid = next(r.id for r in SEED_RULES if r.bottleneck == "uncoalesced")
    h = diagnose(_sig(global_load_eff=0.4), SEED_RULES, disabled={rid})
    assert h.bottleneck != "uncoalesced"
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.rules'`

- [ ] **Step 3: Implement rules.py**

```python
# src/gpuopt/rules.py
from dataclasses import dataclass
from typing import Callable
from gpuopt.model import Signal

@dataclass
class Rule:
    id: str
    match: Callable[[Signal], bool]
    bottleneck: str
    prompt: str
    rationale: str
    priority: int  # higher = checked first

SEED_RULES: list[Rule] = [
    Rule("r_uncoalesced",
         lambda s: s.global_load_eff < 0.7,
         "uncoalesced",
         "Rewrite global memory indexing to be coalesced; add shared-memory tiling.",
         "Low global load efficiency means scattered DRAM access wasting bandwidth.",
         priority=30),
    Rule("r_reg_pressure",
         lambda s: s.occupancy < 0.5 and s.reg_per_thread > 64,
         "reg_pressure",
         "Reduce threads per block or add __launch_bounds__ to cut register use.",
         "Low occupancy with high registers/thread means occupancy is register-limited.",
         priority=20),
    Rule("r_oversync",
         lambda s: s.stall_reason == "sync",
         "oversync",
         "Remove unnecessary __syncthreads or apply double buffering.",
         "Sync-dominated stalls indicate over-synchronization.",
         priority=15),
    Rule("r_compute_bound",
         lambda s: s.achieved_bw_pct < 0.6 and s.compute_tput > 0.7,
         "compute_bound",
         "Use FMA, lower precision where allowed, reduce redundant math.",
         "Bandwidth headroom but high compute throughput means arithmetic-bound.",
         priority=10),
    # Saturation guard is handled in engine.diagnose (not a normal rule):
    # high bandwidth + already coalesced => nothing to optimize.
]
```

- [ ] **Step 4: Implement engine.py**

```python
# src/gpuopt/engine.py
from gpuopt.model import Signal, Hypothesis
from gpuopt.rules import Rule

def _is_saturated(s: Signal) -> bool:
    return s.achieved_bw_pct > 0.8 and s.global_load_eff > 0.9

def diagnose(sig: Signal, rules: list[Rule], disabled=frozenset()) -> Hypothesis:
    if _is_saturated(sig):
        return Hypothesis("memory_saturated", "STOP",
                          "Bandwidth saturated and already coalesced — no headroom.",
                          "r_saturated")
    for r in sorted(rules, key=lambda r: r.priority, reverse=True):
        if r.id in disabled:
            continue
        if r.match(sig):
            return Hypothesis(r.bottleneck, r.prompt, r.rationale, r.id)
    return Hypothesis("none", "STOP",
                      "No known bottleneck signature matched.", "r_nomatch")
```

- [ ] **Step 5: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_engine.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add GPU-Solver/src/gpuopt/rules.py GPU-Solver/src/gpuopt/engine.py GPU-Solver/tests/test_engine.py
git commit -m "feat: hypothesis engine — seed rules + saturation guard + priority/disable"
```

---

### Task 6: Rule Evolver (ledger → confidence + disable list)

**Files:**
- Create: `GPU-Solver/src/gpuopt/evolver.py`
- Create: `GPU-Solver/tests/test_evolver.py`

**Interfaces:**
- Consumes: `Ledger` from Task 3.
- Produces:
  - `def rule_stats(ledger) -> dict[str, dict]`: per `rule_id` → `{"applied": int, "improved": int, "confidence": float}` where `confidence = improved / applied` (0.0 if applied==0).
  - `def disabled_rules(ledger, min_applied: int = 3, max_conf: float = 0.2) -> set[str]`: rules applied ≥ `min_applied` times with confidence ≤ `max_conf`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_evolver.py
from gpuopt.model import RoundRecord, Hypothesis
from gpuopt.ledger import Ledger
from gpuopt.evolver import rule_stats, disabled_rules

def _rec(rule_id, outcome):
    return RoundRecord("p", 0, "s", True, 1.0, None,
                       Hypothesis("b", "p", "why", rule_id), outcome)

def test_confidence_computed(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    for o in ["improved", "improved", "no_change", "worse"]:
        led.append(_rec("r_x", o))
    stats = rule_stats(led)
    assert stats["r_x"]["applied"] == 4
    assert stats["r_x"]["improved"] == 2
    assert stats["r_x"]["confidence"] == 0.5

def test_bad_rule_gets_disabled(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    for o in ["worse", "no_change", "worse", "no_change"]:
        led.append(_rec("r_bad", o))
    assert "r_bad" in disabled_rules(led)

def test_unproven_rule_not_disabled(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    led.append(_rec("r_new", "worse"))  # only 1 application
    assert "r_new" not in disabled_rules(led)
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_evolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.evolver'`

- [ ] **Step 3: Implement evolver.py**

```python
# src/gpuopt/evolver.py
from collections import defaultdict

def rule_stats(ledger) -> dict:
    stats = defaultdict(lambda: {"applied": 0, "improved": 0, "confidence": 0.0})
    for rec in ledger.all():
        if rec.hypothesis is None or rec.hypothesis.prompt == "STOP":
            continue
        s = stats[rec.hypothesis.rule_id]
        s["applied"] += 1
        if rec.outcome == "improved":
            s["improved"] += 1
    for s in stats.values():
        s["confidence"] = s["improved"] / s["applied"] if s["applied"] else 0.0
    return dict(stats)

def disabled_rules(ledger, min_applied: int = 3, max_conf: float = 0.2) -> set:
    return {rid for rid, s in rule_stats(ledger).items()
            if s["applied"] >= min_applied and s["confidence"] <= max_conf}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_evolver.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add GPU-Solver/src/gpuopt/evolver.py GPU-Solver/tests/test_evolver.py
git commit -m "feat: rule evolver — confidence stats + auto-disable underperforming rules"
```

---

### Task 7: Measurement adapter interface + fakes

**Files:**
- Create: `GPU-Solver/src/gpuopt/measure.py`
- Create: `GPU-Solver/tests/test_measure.py`

**Interfaces:**
- Consumes: nothing from prior tasks (pure boundary).
- Produces:
  - `@dataclass MeasureResult`: `passed: bool, percentile: float | None, raw_trace: dict`.
  - `class Measurer(Protocol)`: `submit(self, kernel_src: str, problem_id: str) -> MeasureResult`.
  - `class FakeMeasurer(Measurer)`: `FakeMeasurer(results: list[MeasureResult])` pops next on each `submit`. For tests + dry runs.
  - `class LeetGPUMeasurer` / `class LocalNcuMeasurer`: real adapters — thin, body deferred to live phase per Task 0 decision (`NotImplementedError` stub with a `ponytail:` note).

- [ ] **Step 1: Write failing test**

```python
# tests/test_measure.py
from gpuopt.measure import FakeMeasurer, MeasureResult

def test_fake_measurer_scripts_results():
    m = FakeMeasurer([
        MeasureResult(True, 40.0, {"dram_throughput_pct": 0.5}),
        MeasureResult(True, 65.0, {"dram_throughput_pct": 0.5}),
    ])
    assert m.submit("k1", "p").percentile == 40.0
    assert m.submit("k2", "p").percentile == 65.0
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_measure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.measure'`

- [ ] **Step 3: Implement measure.py**

```python
# src/gpuopt/measure.py
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class MeasureResult:
    passed: bool
    percentile: float | None
    raw_trace: dict = field(default_factory=dict)

class Measurer(Protocol):
    def submit(self, kernel_src: str, problem_id: str) -> MeasureResult: ...

class FakeMeasurer:
    def __init__(self, results: list[MeasureResult]):
        self._results = list(results)
        self._i = 0
    def submit(self, kernel_src: str, problem_id: str) -> MeasureResult:
        r = self._results[self._i]
        self._i += 1
        return r

class LeetGPUMeasurer:
    def submit(self, kernel_src: str, problem_id: str) -> MeasureResult:
        # ponytail: wire to LeetGPU API per Task 0 decision (path A). Not unit-tested.
        raise NotImplementedError

class LocalNcuMeasurer:
    def submit(self, kernel_src: str, problem_id: str) -> MeasureResult:
        # ponytail: nvcc compile + ncu metrics per Task 0 decision (path B). Not unit-tested.
        raise NotImplementedError
```

- [ ] **Step 4: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_measure.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add GPU-Solver/src/gpuopt/measure.py GPU-Solver/tests/test_measure.py
git commit -m "feat: measurement adapter interface + fake + real-adapter stubs"
```

---

### Task 8: Optimization loop (wires everything, fully fake-driven test)

**Files:**
- Create: `GPU-Solver/src/gpuopt/loop.py`
- Create: `GPU-Solver/tests/test_loop.py`

**Interfaces:**
- Consumes: `LLMAdapter` (T1), `Signal`/`Hypothesis`/`RoundRecord` (T2), `Ledger` (T3), `parse_trace` (T4), `diagnose` + `SEED_RULES` (T5), `disabled_rules` (T6), `Measurer`/`MeasureResult` (T7).
- Produces:
  - `def optimize_problem(problem_id, base_prompt, llm, measurer, ledger, rules, max_rounds=5, patience=2) -> list[RoundRecord]`. Per round: build prompt (base + last hypothesis.prompt) → `llm.generate` → `measurer.submit` → on fail record `outcome="fail"` and regenerate (counts as a round) → on pass parse trace, classify outcome vs previous percentile (`improved`/`no_change`/`worse`), `diagnose` next hypothesis using `disabled_rules(ledger)`. STOP when hypothesis.prompt == "STOP", or `patience` rounds without improvement, or `max_rounds`.

- [ ] **Step 1: Write failing test — optimizable problem improves then converges**

```python
# tests/test_loop.py
from gpuopt.llm import FakeLLM
from gpuopt.measure import FakeMeasurer, MeasureResult
from gpuopt.ledger import Ledger
from gpuopt.rules import SEED_RULES
from gpuopt.loop import optimize_problem

def test_optimizable_problem_improves_then_stops(tmp_path):
    llm = FakeLLM(["k1", "k2", "k3"])
    # round0: uncoalesced (eff 0.4), pct 40 -> round1: coalesced+saturated, pct 80 (STOP)
    measurer = FakeMeasurer([
        MeasureResult(True, 40.0, {"global_load_efficiency": 0.4, "dram_throughput_pct": 0.45}),
        MeasureResult(True, 80.0, {"global_load_efficiency": 1.0, "dram_throughput_pct": 0.85}),
    ])
    led = Ledger(str(tmp_path / "l.jsonl"))
    recs = optimize_problem("p_matmul", "solve matmul", llm, measurer, led, SEED_RULES)
    assert recs[0].outcome == "improved" or recs[1].outcome == "stop"
    assert recs[-1].hypothesis.prompt == "STOP"
    assert recs[1].percentile == 80.0

def test_saturated_problem_stops_immediately(tmp_path):
    llm = FakeLLM(["k1"])
    measurer = FakeMeasurer([
        MeasureResult(True, 90.0, {"global_load_efficiency": 1.0, "dram_throughput_pct": 0.85}),
    ])
    led = Ledger(str(tmp_path / "l.jsonl"))
    recs = optimize_problem("p_sigmoid", "solve sigmoid", llm, measurer, led, SEED_RULES)
    assert len(recs) == 1
    assert recs[0].hypothesis.bottleneck == "memory_saturated"

def test_failing_kernel_regenerates(tmp_path):
    llm = FakeLLM(["bad", "good"])
    measurer = FakeMeasurer([
        MeasureResult(False, None, {}),
        MeasureResult(True, 90.0, {"global_load_efficiency": 1.0, "dram_throughput_pct": 0.85}),
    ])
    led = Ledger(str(tmp_path / "l.jsonl"))
    recs = optimize_problem("p1", "solve", llm, measurer, led, SEED_RULES, max_rounds=5)
    assert recs[0].outcome == "fail"
    assert recs[-1].passed is True
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.loop'`

- [ ] **Step 3: Implement loop.py**

```python
# src/gpuopt/loop.py
from gpuopt.model import RoundRecord
from gpuopt.parser import parse_trace
from gpuopt.engine import diagnose
from gpuopt.evolver import disabled_rules

def _classify(prev_pct, new_pct):
    if prev_pct is None:
        return "improved"
    if new_pct > prev_pct:
        return "improved"
    if new_pct < prev_pct:
        return "worse"
    return "no_change"

def optimize_problem(problem_id, base_prompt, llm, measurer, ledger,
                     rules, max_rounds=5, patience=2) -> list[RoundRecord]:
    recs: list[RoundRecord] = []
    next_prompt = ""        # first round: no hypothesis yet
    best_pct = None
    no_improve = 0
    for idx in range(max_rounds):
        prompt = base_prompt + ("\n\n" + next_prompt if next_prompt else "")
        kernel = llm.generate(prompt)
        mr = measurer.submit(kernel, problem_id)

        if not mr.passed:
            rec = RoundRecord(problem_id, idx, kernel, False, None, None, None, "fail")
            ledger.append(rec); recs.append(rec)
            next_prompt = "Previous kernel failed correctness. Fix and retry."
            continue

        sig = parse_trace(mr.raw_trace)
        outcome = _classify(best_pct, mr.percentile)
        hyp = diagnose(sig, rules, disabled=disabled_rules(ledger))

        if hyp.prompt == "STOP":
            outcome = "stop"
        rec = RoundRecord(problem_id, idx, kernel, True, mr.percentile, sig, hyp, outcome)
        ledger.append(rec); recs.append(rec)

        if outcome == "improved":
            best_pct = mr.percentile
            no_improve = 0
        else:
            no_improve += 1

        if hyp.prompt == "STOP" or no_improve >= patience:
            break
        next_prompt = hyp.prompt
    return recs
```

- [ ] **Step 4: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_loop.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run full suite**

Run: `cd GPU-Solver && pytest -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 6: Commit**

```bash
git add GPU-Solver/src/gpuopt/loop.py GPU-Solver/tests/test_loop.py
git commit -m "feat: optimization loop wiring parser/engine/evolver/measurer with convergence"
```

---

### Task 9: Two-track report (curve data + STOP accuracy)

**Files:**
- Create: `GPU-Solver/src/gpuopt/report.py`
- Create: `GPU-Solver/tests/test_report.py`

**Interfaces:**
- Consumes: `Ledger` (T3), `RoundRecord` (T2).
- Produces:
  - `def optimization_curves(ledger) -> dict[str, list[tuple[int, float]]]`: per `problem_id`, list of `(round_idx, percentile)` for passed rounds — the rising-curve data.
  - `def stop_summary(ledger) -> dict`: `{"stopped_problems": [ids], "stop_count": int}` where a problem is "stopped" if its last record's hypothesis.bottleneck is `memory_saturated` or `none`.
  - `def rule_confidence_table(ledger) -> dict`: re-exports `rule_stats` (T6) for the report.

- [ ] **Step 1: Write failing test**

```python
# tests/test_report.py
from gpuopt.model import RoundRecord, Signal, Hypothesis
from gpuopt.ledger import Ledger
from gpuopt.report import optimization_curves, stop_summary

def _pass(pid, idx, pct, bottleneck):
    hyp = Hypothesis(bottleneck, "STOP" if bottleneck in ("memory_saturated","none") else "p",
                     "why", "r")
    return RoundRecord(pid, idx, "s", True, pct, None, hyp, "improved")

def test_curve_extracted_per_problem(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    led.append(_pass("matmul", 0, 40.0, "uncoalesced"))
    led.append(_pass("matmul", 1, 70.0, "memory_saturated"))
    curves = optimization_curves(led)
    assert curves["matmul"] == [(0, 40.0), (1, 70.0)]

def test_stop_summary_counts_saturated(tmp_path):
    led = Ledger(str(tmp_path / "l.jsonl"))
    led.append(_pass("sigmoid", 0, 90.0, "memory_saturated"))
    s = stop_summary(led)
    assert "sigmoid" in s["stopped_problems"]
    assert s["stop_count"] == 1
```

- [ ] **Step 2: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.report'`

- [ ] **Step 3: Implement report.py**

```python
# src/gpuopt/report.py
from collections import defaultdict
from gpuopt.evolver import rule_stats

def optimization_curves(ledger) -> dict:
    curves = defaultdict(list)
    for rec in ledger.all():
        if rec.passed and rec.percentile is not None:
            curves[rec.problem_id].append((rec.round_idx, rec.percentile))
    for v in curves.values():
        v.sort()
    return dict(curves)

def stop_summary(ledger) -> dict:
    last_by_problem = {}
    for rec in ledger.all():
        last_by_problem[rec.problem_id] = rec
    stopped = [pid for pid, rec in last_by_problem.items()
               if rec.hypothesis is not None
               and rec.hypothesis.bottleneck in ("memory_saturated", "none")]
    return {"stopped_problems": stopped, "stop_count": len(stopped)}

def rule_confidence_table(ledger) -> dict:
    return rule_stats(ledger)
```

- [ ] **Step 4: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_report.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add GPU-Solver/src/gpuopt/report.py GPU-Solver/tests/test_report.py
git commit -m "feat: two-track report — optimization curves + STOP summary + rule confidence"
```

---

### Task 10: CLI entrypoint + dry-run end-to-end

**Files:**
- Create: `GPU-Solver/src/gpuopt/cli.py`
- Create: `GPU-Solver/tests/test_cli_dryrun.py`
- Create: `GPU-Solver/problems/example_problems.json`

**Interfaces:**
- Consumes: everything. Wires real-or-fake adapters by a `--dry-run` flag.
- Produces: `def main(argv: list[str]) -> int`. `--dry-run` uses `FakeLLM` + `FakeMeasurer` seeded from the problems file so the whole pipeline runs with no GPU/network and writes a ledger + prints the two-track report.

- [ ] **Step 1: Create example problems file**

```json
// problems/example_problems.json
[
  {"id": "sigmoid", "base_prompt": "Write a CUDA kernel computing sigmoid elementwise.",
   "dry_fixture": [{"passed": true, "percentile": 90.0,
     "raw_trace": {"global_load_efficiency": 1.0, "dram_throughput_pct": 0.85}}]},
  {"id": "matmul", "base_prompt": "Write a CUDA kernel for matrix multiply.",
   "dry_fixture": [
     {"passed": true, "percentile": 40.0,
      "raw_trace": {"global_load_efficiency": 0.4, "dram_throughput_pct": 0.45}},
     {"passed": true, "percentile": 80.0,
      "raw_trace": {"global_load_efficiency": 1.0, "dram_throughput_pct": 0.85}}]}
]
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_cli_dryrun.py
import json, pathlib
from gpuopt.cli import main

def test_dry_run_produces_ledger_and_exits_zero(tmp_path, capsys, monkeypatch):
    problems = pathlib.Path(__file__).parent.parent / "problems" / "example_problems.json"
    ledger_path = tmp_path / "run.jsonl"
    rc = main(["--dry-run", "--problems", str(problems), "--ledger", str(ledger_path)])
    assert rc == 0
    assert ledger_path.exists()
    out = capsys.readouterr().out
    assert "sigmoid" in out          # report mentions both problems
    assert "matmul" in out
```

- [ ] **Step 3: Run, verify fail**

Run: `cd GPU-Solver && pytest tests/test_cli_dryrun.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuopt.cli'`

- [ ] **Step 4: Implement cli.py**

```python
# src/gpuopt/cli.py
import argparse, json
from gpuopt.llm import FakeLLM, APILLM
from gpuopt.measure import FakeMeasurer, MeasureResult
from gpuopt.ledger import Ledger
from gpuopt.rules import SEED_RULES
from gpuopt.loop import optimize_problem
from gpuopt.report import optimization_curves, stop_summary, rule_confidence_table

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--max-rounds", type=int, default=5)
    args = ap.parse_args(argv)

    problems = json.loads(open(args.problems, encoding="utf-8").read())
    ledger = Ledger(args.ledger)

    for p in problems:
        if args.dry_run:
            fixtures = [MeasureResult(f["passed"], f["percentile"], f["raw_trace"])
                        for f in p["dry_fixture"]]
            measurer = FakeMeasurer(fixtures)
            llm = FakeLLM([f"kernel_{i}" for i in range(len(fixtures))])
        else:
            # ponytail: select real adapters per Task 0 decision; key from env.
            raise SystemExit("live mode not wired yet — see docs/00-measurement-feasibility.md")
        optimize_problem(p["id"], p["base_prompt"], llm, measurer, ledger,
                         SEED_RULES, max_rounds=args.max_rounds)

    print("=== Optimization curves ===")
    for pid, curve in optimization_curves(ledger).items():
        print(f"{pid}: {curve}")
    print("=== STOP summary ===")
    print(stop_summary(ledger))
    print("=== Rule confidence ===")
    print(rule_confidence_table(ledger))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run, verify pass**

Run: `cd GPU-Solver && pytest tests/test_cli_dryrun.py -v`
Expected: PASS

- [ ] **Step 6: Run full suite + manual dry run**

Run: `cd GPU-Solver && pytest -v && python -m gpuopt.cli --dry-run --problems problems/example_problems.json --ledger /tmp/run.jsonl`
Expected: all tests PASS; dry run prints curves (matmul rises 40→80), STOP summary lists sigmoid.

- [ ] **Step 7: Commit**

```bash
git add GPU-Solver/src/gpuopt/cli.py GPU-Solver/tests/test_cli_dryrun.py GPU-Solver/problems/
git commit -m "feat: CLI entrypoint with dry-run end-to-end pipeline"
```

---

## Post-MVP (out of scope, do not build now)

Tracked so they aren't forgotten — each is a *future* plan, per spec §5 "제외":
- Wire `APILLM` + `LeetGPUMeasurer`/`LocalNcuMeasurer` real bodies (live phase; gated by Task 0).
- Run the real Easy problem set; produce actual curves for the portfolio.
- RAG/pattern DB, multi-LLM fallback, Medium/Hard, site crawling, principle notes.

## Self-Review

**Spec coverage:**
- §1 narrative / trace→hypothesis differentiator → Tasks 4,5,8. ✓
- §2 components (Generator/Gate/Submit/Parser/Engine/Evolver/Ledger) → T1(LLM=Generator), T7(Measurer=Submit+Gate via `passed`), T4,T5,T6,T3. ✓ (Correctness gate is folded into Measurer.passed — acceptable; no separate gate component needed for MVP.)
- §3 seed rules + rationale + evolver + self-defense (disable failed, convergence) → T5 (rationale mandatory, tested), T6 (disable), T8 (patience/STOP). ✓
- §4 no filtering + two-track classification → T8 (saturation STOP), T9 (two-track report). ✓
- §5 single LLM, exclude RAG/multi-LLM → honored (Post-MVP). ✓
- §6 Pro-trace risk → Task 0 BLOCKING gate. ✓
- §6 rule-evolver convergence evidence → T9 `rule_confidence_table` + curves. ✓

**Placeholder scan:** No "TBD/handle edge cases". Real-adapter bodies are explicit `NotImplementedError` stubs with `ponytail:` notes + a Post-MVP entry — deliberate, gated by Task 0, not a hidden placeholder.

**Type consistency:** `Signal`/`Hypothesis`/`RoundRecord` fields consistent across T2→T9. `diagnose(sig, rules, disabled=)` signature matches T5 def and T8 call. `MeasureResult(passed, percentile, raw_trace)` consistent T7→T8→T10. `rule_stats`/`disabled_rules` consistent T6→T9. ✓
