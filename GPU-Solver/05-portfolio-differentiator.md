---
title: GPU-Solver — 포트폴리오 차별점 (방법론 서술)
created: 2026-06-28
tags: [gpu-solver, portfolio, differentiator]
status: active
---

# GPU-Solver — 무엇을 입증했고, 무엇을 입증 못 했나

> 🎯 이 노트 = 포트폴리오 청중(AX/agentic 시스템)에게 보여줄 단일 서술.
> 톤 = **"멋진 인프라를 만들었다"가 아니라 "가설→통제실험→검증, 안 되면 측정으로 증명한다".**
> 허브 [[GPU-Solver-MOC]], 상세 [[04-multiproblem-round-design]], 재개 [[PROGRESS]].

## 한 줄

LLM 에이전트가 **프로파일링 측정값으로 자기 분류 룰을 진화시키는** 루프를 만들고,
그 진화 메커니즘을 **통제된 ON/OFF 실험으로 입증**했다.
성능 이득은 **측정으로 null임을 확정**해 환경 한계로 귀속시켰다 — future work로 정직히 명시.

## 무엇을 입증했나 (claim 가능)

### 1. 측정 피드백 진화 메타루프 — 메커니즘 입증 (✅)

선행(CUDAMaster, arXiv 2603.07169)은 "결정론 룰 라벨 → LLM 재작성 → 측정 검증" 파이프라인을
이미 구현. **내 기여 = 그 분류 룰표 자체가 측정 피드백으로 진화한다는 것.** 선행 6종 전부 정적 룰.

**증거 (진짜 A100, sigmoid 14R, `run_gain_compare.py`):**
- 진화 **ON** = 틀린 `fp32_no_tensorcore` 룰이 4회 실패 후 retire → `memory_bound_fusable`로 자동 전환.
- 진화 **OFF** = 같은 fp32 룰이 6회 영원 오발화 (fake 신호).
- = "틀린 정적 룰이 측정으로 폐기되고 맞는 룰로 갈아탐"이 실 GPU로 닫힘.

### 2. ablation으로 자기 시스템 검증 (✅ — 이게 핵심 어필)

위 ON/OFF = 단순 데모 아님. **내 시스템의 어느 부품(진화)이 효과 내는지 격리한 ablation.**
같은 variant 큐를 두 트랙에 공정 투입 → 차이의 원인을 진화 유무 단일 변수로 통제.

### 3. negative result를 측정으로 확정 (✅ — 정직성)

### 4. 루프가 측정으로 더 빠른 커널 도달 — gain layer 첫 돌파 (✅ 2026-06-29)

| 문제 | 시도 | 결과 |
|---|---|---|
| **matmul (4096² fp32)** | 루프 R0(fp32)→`fp32_no_tensorcore` 발화→R1(TF32) | ✅ **9.6ms→1.5ms = 6.4× gain** |
| sigmoid | `--latency` 12R, BLOCK 변형 | null — 메모리바운드 천장 (최적화 여지 없음) |
| groupnorm | 분할병렬 3알고리즘 | null — DRAM BW 천장 |
| llama | TF32 OFF vs ON | null — attention(flash) 54% 지배, matmul 23% 비병목 |

**matmul: 루프가 측정→가설→재작성→6.4× 빠른 커널 자율 도달.** = "measure → hypothesis → rewrite →
faster kernel" 첫 실증. sigmoid/groupnorm null은 진짜 천장(메모리바운드), llama는 attention 지배 —
**루프 결함 아니라 문제 특성.** matmul이 compute-bound headroom 있는 깨끗한 문제라 gain 드러남.

> ⚠️ **측정 정직성:** 처음엔 matmul도 95ms 동률("환경 한계")로 나왔으나, 원인 = `_profile_ncu`가
> `--launch-count 1`로 커널 1개만 측정하던 **버그.** 전체 커널 duration 합산으로 고치니 6.4× 드러남.
> ncu 커널명(sgemm vs tensorop)·이론 FLOP 검산으로 교차 확인. = 버그를 측정으로 잡고 정정한 기록.

## 무엇을 입증 못 했나 (절대 주장 금지)

- ❌ **진화 우월성 ON>OFF.** matmul은 첫 발화 룰이 맞아 retire 불요 → ON≈OFF. 진화 이득은 sigmoid
  오발화 retire가 별개 증거. **gain 축과 진화 축이 한 문제서 동시 입증된 건 아님.**
- ❌ **다문제 일반화.** gain=matmul 1문제, 진화=sigmoid 1문제. 일반 우수성 주장 불가.
- ⚠️ **두 축 분리:** loop improves(gain, matmul ✅) + evolution beats static(진화, sigmoid ✅) =
  각각 단일문제 실증. 한 문제서 둘 다 + 여러 문제 일반화 = future work.

> **이 경계를 넘는 순간 격하당한다.** "GPU 커널을 자동 최적화하는 시스템"이라 말하지 말 것.
> speedup 질문에 무너진다. 주장을 **메커니즘·방법론 레이어**에 가둔다.

## 청중별 포지셔닝

**AX/agentic 시스템 (메인 청중):**
> "LLM을 측정 피드백 루프에 끼운 agentic 시스템. 핵심 = 에이전트의 분류 룰이 프로파일링
> 결과로 자기 진화. 메커니즘을 ablation으로 입증하고, 성능 이득은 측정으로 한계를 정량화했다."

**컴파일러/perf 전문가 (보조, 방어용):**
> 성능 숫자 없음을 먼저 인정. "이 환경선 null이고 원인은 측정 환경"을 데이터로 제시.
> = 과장 대신 측정 윤리로 신뢰 확보. speedup 경쟁엔 진입 안 함.

## 왜 초보가 만들어도 격하 안 당하나

차별점은 코드 줄 수나 하네스 정교함이 아니라 **방법론**:
1. 가설을 세운다 (룰표가 진화하면 이득).
2. 통제 실험으로 메커니즘을 격리한다 (ON/OFF ablation).
3. 안 되면 안 된다고 측정으로 증명한다 (3문제 null, 원인 귀속).
4. claim을 레이어로 분리한다 (메커니즘 vs 이득, 단일문제 vs 일반화).

= 시니어 연구자 사고방식. 하네스(git-mailbox·ncu 파이프라인)는 이 사고를 받치는 도구일 뿐,
자랑거리가 아니다.

## 재현 가능성 (환경 갖춰지면)

성능 gain 측정 인프라는 준비됨 — 환경만 PoC급(1.4ms)이면 그대로 재시도:
- `loop/run_gain_hypcond.py` = 가설-조건부 콜백 드라이버 (발화 룰 라벨 보고 코드 선택).
- 상세 재시도 절차 = [[PROGRESS]] §성능 gain 인프라.

## 링크

- [[GPU-Solver-MOC]] — 문서 허브
- [[04-multiproblem-round-design]] — A/B 실험 상세 (메커니즘 증거)
- [[02-prior-art-survey]] — 선행연구 비교, 차별점 검증
- [[PROGRESS]] — 재개 진입점
