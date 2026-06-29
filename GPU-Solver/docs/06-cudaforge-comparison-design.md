---
title: CudaForge 성능 비교 실험 설계
status: design
created: 2026-06-29
tags: [gpu-solver, experiment, cudaforge, comparison, benchmark]
---

# CudaForge 성능 비교 실험 설계

> 🗂️ 인덱스: [[GPU-Solver-MOC]] · 선행: [[02-prior-art-survey]] · 진행: [[PROGRESS]]
> 목적: "우리 룰 기반 결정론 판정 vs CudaForge LLM-Judge 판정" — 어느 쪽이 더 나은 최적화 결정을 내나, **실측**으로 답.

## 0. 가설 (claim layering 준수)

미측정 = 모름. "확실히 나쁘다"도 "이긴다"도 데이터 없는 추측 (survey §정직한 리스크).
이 실험이 그 빈칸을 메움. 두 층 분리:

- **H1 (성능층)**: 같은 문제·같은 LLM·같은 측정 환경에서, 우리 루프 최종 speedup ≥ CudaForge 최종 speedup.
  - 예상: **불확실.** 우리=3소스(ncu/nsys/torch) 넓음 but 유한 룰. CudaForge=ncu 1소스 but LLM 자유추론. 트레이드.
- **H2 (품질층, 차별점 본질)**: 우리 판정이 **결정론·감사가능·재현**하면서 CudaForge와 **비등한 결정**(같은 병목 라벨 일치율 ≥ X%)을 낸다.
  - 이게 진짜 논점. **성능 동률이어도 H2 성립 시 차별점 살아있음.**

⚠️ **포트폴리오 금지 문구**: "CudaForge보다 빠르다" — 실측이 그렇게 안 나오면 못 씀. 결과대로만 서술.

## 1. 통제 변수 (공정성 — expert가 반박 못 하게)

| 변수 | 고정값 | 이유 |
|---|---|---|
| LLM 생성기 | **동일** (둘 다 같은 call_fn — GLM 또는 동일 Claude) | 코드 생성 품질 차이 제거. 판정 방식만 비교 |
| 측정 환경 | 동일 A100, 동일 ncu 설정 (launch-count 무제한) | 환경 parity (s1-284) |
| 문제 셋 | 동일 (아래 §3) | — |
| 라운드 예산 | 동일 (예: 각 문제 최대 8R) | 수렴 속도도 비교 |
| Correctness Gate | 동일 (robust-kbench 6체크 차용) | 리워드해킹 방지 (survey §1) |
| seed 코드 | 동일 fp32 naive | 출발점 동일 |

**유일 독립변수 = 병목 판정 주체.** A=우리 룰(결정론), B=CudaForge Judge LLM(자유).

## 2. CudaForge arm 구현 (B)

> ⚠️ 핵심 리스크 = CudaForge 재현. 두 경로.

- **(권장) 논문 충실 재구현**: Coder+Judge 2에이전트, training-free. Judge 프롬프트 = survey §핵심대조 verbatim
  ("identify exactly one highest-impact bottleneck by 3-4 most important metrics ... occupancy, coalescing, ... etc.").
  ncu 24-metric 제공 → Judge가 자유 선택·판정 → Coder가 재작성. **우리 harness에 'judge_mode' arm으로 끼움**
  (mailbox/gate/측정 전부 공유, 판정 단계만 LLM 호출로 교체).
- **(대안) 공식 repo 직접 실행**: `OptimAI-Lab/CudaForge`. 단 우리 문제 포맷·gate와 배선 비용 큼, 환경 차이 변수 유입.
  → **재구현이 통제 우수.** 같은 harness 안에서 판정만 갈음 = 가장 공정.

**판정 주체 외 전부 공유** = "판정 방식" 순수 비교. 이게 설계 핵심.

## 3. 문제 셋 (CudaForge 사각 노출 설계)

세 부류 — 각 부류가 다른 걸 드러냄:

| 부류 | 예 | 무엇 검증 | 예상 |
|---|---|---|---|
| **A. compute-bound matmul** | 4096²~8192² GEMM | ncu만으로 충분한 영역 | CudaForge 유연성 유리할 수 있음 |
| **B. ncu 사각 (우리 강점 후보)** | 작은 커널 다수 / launch 오버헤드 큰 워크로드 | nsys·torch 신호 필요 — CudaForge ncu 1소스 못 봄 | **우리 우위 가능** (launch_overhead/attention_dominant 룰) |
| **C. 함정 (오판 유도)** | 메모리바운드인데 compute처럼 보이는 것 / attention 지배 llama | 잘못된 판정 비용 | 결정론 룰 = 일관, LLM = 가끔 환각 |

**B가 a 작업의 존재 이유.** a 없으면 B에서 우리도 ncu만 → CudaForge와 동률. a로 nsys/torch 신호 룰화 → B에서 차별 가능.

## 4. 측정 지표

**H1 (성능):**
- 문제별 최종 speedup (vs seed), 평균·중앙값.
- 수렴 라운드 수 (적을수록 좋음).
- 헛 라운드 비율 (gate 통과 못 한 시도 / 전체).

**H2 (품질·차별점):**
- **판정 일치율**: 같은 신호에서 우리 룰 라벨 == CudaForge Judge 라벨? (병목 종류 기준)
- **재현성**: 같은 입력 N회 → 우리=100% 동일(결정론), CudaForge=? (LLM 비결정 측정).
- **감사 가능성**: 각 판정에 rationale 추적 가능 (우리 ✅ / CudaForge ❌ 블랙박스) — 정성.
- **오판 복구**: 틀린 판정 후 회복 — 우리=evolver retire, CudaForge=다음 라운드 LLM 재판단.

## 5. 결과 해석 매트릭스 (미리 정직하게)

| 결과 | 해석 (포트폴리오 서술) |
|---|---|
| H1 우리>CudaForge | "결정론 룰이 비등 이상 + 투명" — 강한 주장 (단 B부류 덕일 수 있음, 명시) |
| H1 동률 | "성능 동률, 우리는 결정론·감사·진화 추가" — **차별점 충분, 정직** |
| H1 우리<CudaForge | "성능은 LLM-Judge 우위. 우리 기여 = 투명·재현·진화 (성능 SOTA 경쟁 아님)" — survey §(4) 톤 |
| B부류서만 우리 우위 | "다신호 통합이 ncu-only Judge의 사각 메움" — **a의 가치 입증** |

**어느 결과든 차별점(H2)은 성능과 독립.** = 실험이 손해 안 봄. 정직이 곧 방어.

## 6. 단계 (실행 순서)

1. **a 마무리** — 새 룰(launch_overhead, attention_dominant) executor 배선 (현재 파서·룰만 있음, 측정 미연결).
   - nsys/torch 측정을 executor에 추가 → Signal 채움. **Colab 재시작 1회 필요** (executor 변경).
2. **judge_mode arm** — harness에 CudaForge 재구현 arm 추가 (판정만 LLM 호출).
3. **문제 셋 준비** — A/B/C 부류 solve.py seed (B부류=launch 오버헤드 워크로드 신규).
4. **공정성 검증** — 같은 LLM·환경·gate 확인 (변수 1개만).
5. **실행** — 부류×arm 매트릭스, mailbox 라운드.
6. **분석** — §4 지표 집계, §5 매트릭스로 해석.

## 7. 리스크·완화

- **CudaForge 재구현 ≠ 원본.** → 논문 프롬프트 verbatim + training-free 충실. "재구현이라 원본과 다를 수 있음" 명시.
- **B부류 cherry-picking 보임.** → A(공정)·C(함정)도 같이 = 우리 불리 케이스도 보고. 정직.
- **judge LLM 비결정 → 비교 불안정.** → arm마다 seed 고정·N회 반복 median.
- **a 측정 비용↑** (ncu+nsys+torch 3패스). → torch 싼 1차 스크리닝 → ncu 확정 계층화.

## 링크
- [[02-prior-art-survey]] — CudaForge 구조 (§핵심대조, Judge 프롬프트 verbatim)
- [[PROGRESS]] — 진행 현황
- [[GPU-Solver-MOC]] — 허브
