---
title: 칩×문법 컨텍스트 설계 — 환경을 룰의 1급 입력으로
created: 2026-06-30
tags: [gpu-solver, design, portability, rules, context]
status: 🟢 칩 가드 구현 + ctx 배선 완료 + T4 실측 입증 (2026-06-30)
---

# 칩×문법 컨텍스트 설계 (Context as first-class rule input)

> 🧭 병목 판별·해결법은 **(칩, 문법)에 의존**하는데, 현 룰은 A100·Triton/cuBLAS를
> 주석(rationale)에만 박아두고 cond엔 안 넣음. 이 누락이 난항 일부의 원인.
> 이 노트 = 환경을 룰의 1급 입력으로 올리되, **차별점(룰 진화)을 훼손하지 않는** 설계.
> 허브 [[GPU-Solver-MOC]], 재개 [[PROGRESS]]. 선행 교훈 [[04-multiproblem-round-design]].

## 1. 문제 — 환경 의존성이 룰에 숨어있다

병목의 진단·처방은 **(칩, 문법) 쌍의 함수.** 같은 신호라도 환경이 다르면 다른 처방:

| 신호 | A100 + Triton | T4 + Mojo | V100 + CUDA |
|---|---|---|---|
| fp32 matmul 느림 | TF32 켜라 (TC 있음) | **TF32 없음** → 알고리즘 변경 | TF32 없음, fp16 TC만 |
| `tl.dot` 자동 TF32 | 발화 막힘 (관측됨) | Mojo엔 `tl.dot` 없음 | Triton 아님 |

현 `rules.py`는 이 의존성을 **rationale 주석에만** 둠. 예:
- `tensorcore_saturated` rationale = "**A100** TF32 throughput ≈ bf16" ← A100 전용 사실.
- `fp32_no_tensorcore` = "텐서코어 놀고있음 → TF32" ← V100엔 TF32 없음, T4 TC 약함.

`Signal`에 **칩 필드 없음** = 룰이 "어느 칩이냐" 못 받음. cond 가드 부재.

### 이 누락이 실제 난항을 냈다 (정직)

[[04-multiproblem-round-design]] / [[PROGRESS]]의 두 난항이 정확히 이 누락에서 나옴:
- **matmul_tri 두 축 동시 실패** = Triton `tl.dot`이 TF32 텐서코어 **자동**(tc=True)으로 켜서
  `fp32_no_tensorcore`(cond=`not tc`) 발화 막힘. = **문법(Triton)이 신호를 바꿨는데** 룰은
  cuBLAS sgemm을 암묵 전제 → 헛다리. 문법을 안 봐서 생긴 혼선.
- **llama TF32 무효** = matmul이 이미 텐서코어 경로(A100 자동)라 "fp32→TF32" 룰 부적용 →
  `attention_dominant` 룰을 사후에 추가해 땜빵.

→ **치명타는 아님** (PoC가 A100 고정 + 진화가 사후 흡수). 단 **이식성 약점**이고
난항의 일부 원인인 건 맞음.

## 2. 핵심 분리 — 측정(Signal) vs 환경(Context)

두 종류 입력을 dataclass로 분리:

| | 출처 | 시점 | 예 |
|---|---|---|---|
| **Signal** (기존) | profiler가 측정 | 커널 실행 **후** | `bw_pct=0.83` |
| **Context** (신규) | 사람/자동 탐지 | 커널 실행 **전** | `chip="t4"` |

칩·문법은 **측정으로 안 나옴 — 미리 정하는 설정.** `Signal`에 욱여넣으면 의미가 섞임
(측정값+설정값 혼재). 별 dataclass = 깔끔히 분리.

```python
@dataclass
class Context:           # 측정 아닌 "환경" (사전 주어짐 / 자동 탐지)
    chip: str = ""       # "a100" | "t4" | "v100" | "h100"
    lang: str = ""       # "triton" | "cuda" | "mojo" | "torch"
    # 파생 capability는 CHIP_CAPS에서 조회 (아래)
```

## 3. 황금 규칙 — 사용자/LLM은 룰을 "넣지" 않는다

질문: "이용자가 칩·문법 말하면 그걸 룰에 넣나?"
**답: 룰엔 안 넣음. Context(환경)에 넣고, 룰은 그 Context를 *읽어서* 가드한다.**

뭐가 어디로 들어가는지 정확히:

| 항목 | 누가 채움 | 종류 | 검증 |
|---|---|---|---|
| `chip="t4"`, `lang="mojo"` | **사용자 발화 or 자동 탐지** | 설정값(사실) | nvidia-smi / 파일 스캔 |
| `CHIP_CAPS["t4"]={tf32:False}` | **사람이 1회 박음** | 공개 HW 스펙(고정) | NVIDIA 문서 |
| 룰 cond `if ctx.has_tf32...` | **이미 코드에 있음** | 읽기 가드 | self-check |
| **"T4서 뭐가 빠른가" 룰** | ❌ **아무도 손으로 안 넣음** | 학습 결과 | **측정→진화가 발견** |

→ 사용자 입력 = **"환경이 뭐냐"(사실)만.** "그 환경서 어떻게 최적화하냐"(지식)는
**측정이 생성.** 둘을 섞으면 = 사람이 답을 알려주는 꼴 = 진화 무의미 = 차별점 훼손
(정적 CUDAMaster로 회귀). **사용자는 무대만 설정, 룰은 그 무대서 뭐가 되는지 측정으로 학습.**

### 왜 LLM이 즉석 룰 작성을 안 하나

사용자 "T4, Mojo" → 내가 "T4는 TF32 없으니 bf16" 룰을 즉석 작성 = **검증 안 된 룰.**
틀릴 수 있고, 무엇보다 **룰은 측정으로 진화**해야 차별점이 산다. LLM이 손으로 넣으면
정적 룰표와 동일 = 차별점 소멸. → **금지.**

## 4. 칩 = 1급 변수 (자동 탐지 + 박힌 사실)

칩은 신호에 **안 나타나는 사실**(TF32 *가능 여부*는 측정 못 함) → 1급 필요.

```python
# 칩 → capability: 공개 스펙, 사람이 1회 박음 (고정값)
CHIP_CAPS = {
    "a100": {"tf32": True,  "tc_gen": 3, "bf16": True},
    "t4":   {"tf32": False, "tc_gen": 1, "bf16": False},  # Turing, TF32 없음
    "v100": {"tf32": False, "tc_gen": 1, "bf16": False},  # fp16 TC만
    "h100": {"tf32": True,  "tc_gen": 4, "bf16": True, "fp8": True},
}
```

탐지 = `nvidia-smi`가 이미 앎(`sm_75`=T4, `sm_80`=A100). watch가 셀1서 확인 중 →
**사용자 입력 불요, 자동.** 룰 cond에 가드:

```python
cond=lambda sig, ctx: CHIP_CAPS.get(ctx.chip, {}).get("tf32") \
                      and sig.compute_tput > 0 and not sig.tensorcore_active
#                     ^^^ T4면 False → fp32_no_tensorcore 발화 안 함 = 헛가설 차단
```

**측정 입증 범위**: Colab이 A100·T4·V100 무료 전환 → **칩 2~3칸 공짜 실측 가능.**
이게 진짜 이식성 입증.

## 5. 문법 = 1급 안 만든다 (신호가 흡수)

문법축은 칩축보다 **훨씬 약함.** 정직 평가:

| | 탐지 | 입증 가능성 |
|---|---|---|
| **칩** | nvidia-smi 1줄, 확실 | A100·T4 Colab 공짜 = 2칸 즉시 |
| **문법** | 정적 스캔 지저분, 파일 단위라 입도 안 맞음 | Triton·torch만. CUDA=inline 가능, **Mojo=환경 0** |

문법 탐지의 진짜 벽 2개:
1. **한 파일 다문법** — solve.py가 matmul=torch(cuBLAS) + elementwise=Triton 혼재.
   "이 solve의 문법"이 단일 답 없음. 룰은 *커널별* 문법이 필요한데 탐지는 파일 단위 = **입도 불일치.**
2. **Mojo = 환경 자체가 없음** — 탐지 코드 짜도 Mojo 깐 환경 없으면 측정 0칸. 주장하면 과장.

### 결론: 문법은 "신호로 간접 흡수"로 서술

**룰이 문법을 직접 알 필요 없다 — 문법 차이는 ncu 신호에 이미 나타난다.**
matmul_tri 교훈을 약점이 아니라 **신호 기반 설계의 강점**으로 프레임:

> Triton `tl.dot`이 TF32를 자동으로 켜면 → `tensorcore_active=True`가 측정됨 →
> 룰이 문법을 몰라도 신호가 달라서 다른 룰이 발화. 문법 차이 = 측정 신호가 관측.

→ **칩만 1급**(신호에 안 나오는 사실), **문법은 신호 흡수**(이미 신호에 반영됨).
입증 안 될 축(Mojo)을 주장 안 함 = 정직. 칩 이식성만 = 깔끔한 차별점.

## 6. 차별점에 미치는 효과 — 오히려 격상

현 차별점 = "룰표가 측정 피드백으로 진화" (정적 CUDAMaster 대비).
칩축 추가 시: **진화 = (칩) 신규 칸의 룰을 자동 학습.** 사람이 T4 룰표를 손으로 안 짜도,
루프가 측정→실패→retire→새 룰로 그 칸을 채움. = 진화가 **단일 칩 신뢰도 갱신을 넘어
미지의 HW로 일반화.** CUDAMaster·CudaForge는 CUDA+특정 GPU 고정 → **칩 이식성 = 진짜 빈 자리.**

## 7. 작업 계획 (최소, 회귀 없음)

1. `signals.py`에 `Context` dataclass + `CHIP_CAPS` 테이블 추가.
2. 칩 자동 탐지: watch가 `nvidia-smi`/`torch.cuda.get_device_capability`로 `ctx.chip` 채움.
3. `rules.py`의 `match(sig, rules)` → `match(sig, ctx, rules)`, 칩 의존 룰 cond에 `ctx` 가드.
   - 대상: `fp32_no_tensorcore`, `tensorcore_saturated` (TF32/TC 가정 룰).
   - A100 칸은 기존 동작 **보존** (회귀 없음 self-check).
4. 빈칸(T4 등) = "설계상 Context 1줄 + cond 가드, HW 생기면 즉시 측정" 명시.
5. 측정: Colab T4 전환으로 1칸 실측 (이식성 입증). 야망에 따라 V100 추가.

**범위 경고**: 칩 N종 제대로 = 그 HW 실측 필요. 지금 입증 = A100 1칸.
"N칸 일반화" 주장하되 1칸만 측정 = 과장 → **측정한 칸만 주장**(claim layering 규칙 준수,
[[04-multiproblem-round-design]] applicability vs gain 분리와 동일 정신).

## 7-b. 구현 완료 (2026-06-30, 회귀 0)

칩 가드 구현·검증. loop repo 커밋 (signals.py + rules.py).

**signals.py:**
- `Context(chip, lang)` dataclass — 측정 아닌 사전 환경.
- `CHIP_CAPS` = {a100, h100, t4, v100} capability (TF32/TC세대/bf16). 박힌 사실.
- `detect_chip(cc, name)` — compute capability 튜플/이름 → 칩 키 (watch 자동탐지용).
- `Context.cap(key)` 가드 규칙:
  - `key=""` (칩 무관 룰) → **항상 True**.
  - 칩 미지/미등록 → **True** (보수적 통과 = 회귀 0).
  - 칩 명시 + 능력 없음 → **False** (헛가설 차단).

**rules.py:**
- `Rule.chip_cap` 필드 — 신호 cond와 **분리**(lambda 11개 안 건드림, 가드 선언적 1필드).
- `fp32_no_tensorcore`·`tensorcore_saturated`에 `chip_cap="tf32"`.
- `match(sig, rules, ctx=None)` — `ctx.cap(chip_cap) AND cond(sig)`. ctx=None=DEFAULT_CTX=종전 동작.

**검증 (self-check 9~13):**
- A100/ctx없음 → `fp32_no_tensorcore` 발화 (회귀 0).
- **T4 → fp32/saturated 차단 → 같은 신호서 `memory_bound_fusable`로 자연 강등** = 헛가설 차단 + 진화 흡수 경로 입증.
- 회귀 0 확인: signals/rules/selfcheck/evolver/harness **전부 PASS**.

**버그 1개 잡음**: `cap("")`이 `CHIP_CAPS[chip].get("", False)`=False 반환 → 가드 없는 룰까지 차단.
→ `key=""` 早期 True 반환으로 수정.

**미완→완료 (2026-06-30 후속):**
- ✅ **ctx 배선 완료** — `harness.run_loop(ctx=)` → `match(sig, rules, ctx)`, `runner.run_problem(ctx=)`,
  `run_gain_compare --chip=<key>` → `Context` 생성 전달. self-check 7개 PASS (커밋 loop master).
- ⬜ **칩 자동탐지 미배선** — 지금 `--chip` 사용자 명시(런타임 선택과 일치). watch가
  `get_device_capability`로 자동 채워 RES에 싣는 건 미구현 (수동으로 충분히 입증됨).
- `lang` 필드 = 기록용만 (신호 흡수 설계대로 룰 안 씀).

## 7-c. T4 실측 입증 (2026-06-30) — 칩 가드 작동 확인

Colab **T4 (Tesla T4, sm_75)** 전환 후 matmul fp32 4096² 실행. `run_gain_compare matmul ... --chip=t4`.

**측정 신호 (RES JSON, 4096² fp32 sgemm on T4):**
| 신호 | 값 | 판정 |
|---|---|---|
| compute_tput | **0.967** | SM 연산 96.7% 포화 = sgemm 이미 최적 |
| occupancy | 0.489 | <0.5 |
| reg_per_thread | 122 | >64 |
| tensorcore_active | False | T4 = TF32 없음 |
| latency | 48.4 ms | 2.8 TFLOP/s = T4 fp32 피크(8.1) 35% |

**핵심 결과 = 칩 가드 ✅:**
- 같은 fp32 matmul 신호인데 **A100선 `fp32_no_tensorcore` 발화(→TF32 6.4×), T4선 차단.**
- T4 = `chip_cap="tf32"` 가드가 막음 → 다음 적격 룰 `reg_pressure` 발화 (cond `occ<0.5 & reg>64` 정확 매칭).
- = **"T4서 못 낼 TF32 헛가설 사전 차단"** 실측. 룰 손 안 대고 Context(칩)만 바꿔 발화 갈림.

**부가 발견 1 — reg_pressure는 헛가설 (compute 천장):**
compute_tput 96.7% = cuBLAS sgemm 포화. reg_pressure 가설("스레드↓로 occ↑")은 부적절
— occ 올려도 연산 포화라 latency 불변(48.4ms 평평 확인). 진짜 천장 = **T4 fp32 코어 throughput**
(텐서코어 없이 알고리즘 여지 0). = applicability(루프 돎)✓, gain(천장)=null = 정직한 결과.

**부가 발견 2 — retire 관찰 완료 (2026-06-30 후속, 첫 시도는 미관찰):**
첫 시도 = 8R 중 5R converged 조기종료 (retire 0). 원인 = `CONVERGE_PATIENCE=3` == `RETIRE_AFTER_FAILS=3`
→ 수렴정지가 retire와 동시/먼저 발동, reg_pressure fail 3 도달 직전 converged.
**픽스 = `CONVERGE_PATIENCE 3→6`** (retire 조건 `fail>=3 & conf<=0.25 & n>=4` 채울 4R+ 확보).
재측정 T4 8R: **ON retire=1 > OFF retire=0.** reg_pressure 헛가설 8R 발화, ON 트랙만 측정 누적해 폐기.
= **칩 가드(사전차단)+진화 retire(사후폐기)를 T4 한 환경서 동시 실측.** sigmoid(A100) retire와 별개 2번째 입증.
부수 인프라 = `watch.py raw_script` 분기(REQ에 임의 파이썬 실어 보냄=executor 우회) → 새 측정 추가 시 Colab 재시작 영원 불요.

**이중 방어 그림 (차별점 격상):**
- **칩 가드** = 알려진 헛가설(TF32 on non-TF32 chip) **사전 차단** (선언적, CHIP_CAPS).
- **진화 retire** = 모르는 헛가설 **사후 폐기** (측정 누적, sigmoid서 입증).

## 8. 한 줄 요약

- 환경(칩·문법) = **측정 아닌 사전 사실** → `Context`로 분리, `Signal`과 안 섞음.
- 사용자/LLM은 **무대(Context)만 설정**, 그 무대의 최적 룰은 **측정→진화가 발견** (룰 손수정 금지).
- **칩 = 1급**(신호에 안 나오는 사실, nvidia-smi 자동탐지 + CHIP_CAPS), **문법 = 신호 흡수**.
- 효과 = 진화가 미지의 칩으로 일반화 = 차별점 격상. 단 측정한 칸만 주장(정직).

## 링크
- [[GPU-Solver-MOC]] — 허브
- [[PROGRESS]] — 재개 진입점
- [[04-multiproblem-round-design]] — 진화 메커니즘 / claim layering(applicability vs gain)
- [[2026-06-22-agentic-gpu-optimizer-design]] — 설계 스펙 (룰엔진 §3.2)
- [[05-portfolio-differentiator]] — 포트폴리오 차별점 서술
