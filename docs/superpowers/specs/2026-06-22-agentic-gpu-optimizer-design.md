---
title: Agentic GPU Kernel Optimizer — 설계 (Design Spec)
status: proposal
created: 2026-06-22
tags: [gpu-solver, project, gpu, agent, ax, leetgpu, portfolio, design]
---

# Agentic GPU Kernel Optimizer — 설계

> 🗂️ **인덱스: [[GPU-Solver-MOC]]**
> ⚠️ **제안(proposal)이며 확정 아님.** 브레인스토밍 합의 결과를 정리한 한 갈래.
> 출발: [[agentic-gpu-solver-concept]] 컨셉 메모를 포트폴리오 목표로 재조준.
> 다음: [[2026-06-22-agentic-gpu-optimizer]] 구현 계획.

> **📌 Task 0 검증 반영 (2026-06-22, [[00-measurement-feasibility]]):**
> - 측정원 분리 확정: **Colab Pro A100 + ncu = 루프 신호원(무제한)**, **LeetGPU Pro = percentile 검증(가끔)**.
> - 포폴 곡선 = 단일 percentile이 아니라 **2종**: ncu 메트릭 개선 곡선(메인) + LeetGPU percentile 검증점.
> - 섹션6 최우선 리스크("Pro 트레이스 필요성") **해소.** A100 칩도 LeetGPU와 일치.
> - 아래 섹션0/5/6은 이 반영을 포함하도록 갱신됨.

## 0. 포지셔닝 (가장 중요 — 이게 모든 결정의 기준)

- **목표 = 포트폴리오.** 상업/학습 아님.
- **직군 = AX(에이전트/LLM) 엔지니어 메인 + 저수준 도메인 적용력 보조.**
  - 이 프로젝트는 "컴파일러 엔지니어" 어필을 **하지 않는다.** GPU 커널 최적화 ≠ 컴파일러 엔지니어링. 컴파일러 직군은 별도 프로젝트 필요.
  - "GPU perf 전문" 어필도 **하지 않는다.** perf는 목적지가 아니라 통로.
- **한 줄 셀링**: "흔한 텍스트 태스크 말고, GPU 커널 최적화처럼 **검증·측정이 빡센 도메인**에 에이전트 시스템을 붙여 percentile을 실제로 올렸다."
- **약점→강점 전환 원칙** (전 설계 관통):
  1. GPU 깊이 부족 → "커널은 LLM이 짠다. 나는 루프를 설계한다." (위임)
  2. 룩업룰 손작성 한계 → "룰을 사람이 다 못 채워서, 결과 피드백으로 룰이 진화하는 메타루프를 설계했다." (자기개선)

## 1. 핵심 서사 & 차별점

**한 줄 정의**: 트레이스를 해석하는 머리는 내가 짜고, 코드 재작성만 LLM에 시키는, **근거 기반 GPU 커널 최적화 루프.**

**수동 루프와의 결정적 차이** = 루프 안의 **가설 엔진**.
- 수동: "더 빠르게 해줘" (맹목)
- 시스템: `트레이스 숫자 → 병목 분류 → 타깃 변형` (인과)

이 트레이스→가설 매핑 로직 + 그 룰을 진화시키는 메타루프 = AX 깊이의 실체.

**포폴 결과물 = 곡선 + 로그.** 단순 "통과율" 아님:
- percentile 상승 곡선 — 각 상승점에 "어떤 병목을 어떤 가설로 고쳤나" 로그가 붙음.
- 이 로그가 수동 루프와의 격차이자 면접 방어선.

## 2. 아키텍처

```d2
direction: right
problem: "문제 + starter" {shape: document}
gen: "Generator\n(LLM 호출, 글루)"
gate: "Correctness Gate\n(challenge.py / 자체)"
submit: "LeetGPU 제출\n(API 글루)"
trace: "Trace Parser\n★내 코드★" {style.fill: "#ffe08a"}
hyp: "Hypothesis Engine\n★내 코드: 시드룰 매칭★" {style.fill: "#ffe08a"}
evolver: "Rule Evolver\n★내 코드: 메타루프★" {style.fill: "#ffd24a"}
log: "Run Ledger\n(JSON, 곡선 원천)" {shape: cylinder}

problem -> gen -> gate
gate -> submit: "pass"
gate -> gen: "fail (재생성)"
submit -> trace: "percentile + trace"
trace -> hyp: "구조화 병목"
hyp -> gen: "타깃 프롬프트 주입"
submit -> log
hyp -> log: "가설+근거"
log -> evolver: "성공/실패 이력"
evolver -> hyp: "룰 신뢰도 갱신/폐기/추가"
```

**컴포넌트** (굵게 = 내 코드 = 자랑거리, 나머지 = 교체 가능 글루):

1. **Generator** — 문제+(가설 프롬프트)→LLM→커널 코드. *글루.* 단일 LLM (멀티-LLM 폐기).
2. **Correctness Gate** — pass/fail. challenge.py 라이선스 막히면 자체 테스트 생성기(우회안 C). *얇음.*
3. **LeetGPU 제출** — API 제출→percentile+트레이스 회수. *글루.*
4. **★Trace Parser★** — 원시 트레이스/메트릭 → 정규화 신호 dict. "무슨 신호를 뽑을지" 선택 = perf 이해 증거.
5. **★Hypothesis Engine★** — 신호 → 시드룰 매칭 → 병목 라벨 → 변형 프롬프트. **A 깊이의 1차 심장.**
6. **★Rule Evolver★** — Ledger 이력 보고 룰 신뢰도 갱신/폐기/후보추가. **AX 깊이의 새 심장.**
7. **Run Ledger** — 매 라운드 (코드/percentile/가설/근거/결과) 적재. 곡선+로그 원천.

**경계 한 마디**: 4·5·6 = 내 지능(해석·판단·진화). 1(LLM) = 시키는 대로 코드만.

## 3. 가설 엔진 내부

3단계 + 메타루프. 전부 내 코드.

### 3.1 Trace Parser
원시 트레이스 → 정규화 신호:
```python
{ "occupancy": 0.38, "reg_per_thread": 96, "l2_hit": 0.41,
  "global_load_eff": 0.55, "stall_reason": "sync",
  "achieved_bw_pct": 0.30, "compute_tput": 0.15 }
```

### 3.2 시드 룩업 룰 (5~6개, 얇게 시작)
GPU 학습 부담 압축(3~5일). 각 룰에 **"왜 이 신호→이 병목" 근거 1줄 필수.** 근거 없으면 룩업테이블, 있으면 perf 이해.

```python
RULES = [
  # (조건,                          병목 라벨,           변형 가설 프롬프트,            근거)
  (lambda t: t.bw_pct>0.8,          "memory_saturated", "STOP: 대역폭 포화, 손댈 것 없음", "elementwise는 BW가 천장"),
  (lambda t: t.load_eff<0.7,        "uncoalesced",      "인덱싱 재배열 + shared 타일링",  "비합착 접근 = 대역폭 낭비"),
  (lambda t: t.occ<0.5 and t.reg>64,"reg_pressure",     "스레드 수↓ 또는 __launch_bounds__","점유율 제한 = 레지스터"),
  (lambda t: t.stall=="sync",       "oversync",         "__syncthreads 축소 / 더블버퍼링","과도한 동기화 = stall"),
  (lambda t: t.bw_pct>0.8 and 느림, "compute_bound",    "FMA 활용 / 정밀도 낮추기",       "대역폭 여유인데 느림 = 연산"),
  # ... 시드 5~6개
]
```
우선순위: 가장 지배적 병목 1개만 선택.

### 3.3 Rule Evolver (메타루프 — AX 정점)
Ledger 이력으로 룰 자체를 갱신:
- 가설 적용 후 percentile 올랐나 → 룰별 `성공/실패` 카운트.
- 신뢰도 높은 룰 우선 적용 (탐색-활용).
- 반복 실패 룰 → 폐기 후보.
- 새 패턴 후보 제안 (2차).

면접 한 마디: **"룰을 사람이 다 못 채워서, 결과 피드백으로 룰이 진화하는 메타루프를 설계했다."**

### 3.4 자기방어 장치
- **가설 검증 기록**: 실패한 룰은 같은 문제서 재시도 안 함.
- **수렴 정지**: N라운드 개선 없으면 종료 (무한 토큰 방지).

## 4. 문제 처리 정책 (체리피킹 금지)

**문제 선별 안 함. 전수 투입.** 결과 보고 고르면 사기. 시스템이 *자동 분류*하면 기능.

- 모든 Easy 문제 통째 투입 (고르지 않음).
- 시스템이 자동 분류 → **2 트랙**:
  1. **최적화 가능군** (행렬곱/reduction/stencil류): **ncu 메트릭 개선 곡선**(occupancy/BW 라운드별 상승). 최종 커널만 LeetGPU Pro percentile로 검증.
  2. **포화군** (sigmoid/grayscale류 BW포화 elementwise): "STOP, 최적화 불가" 판정.
- **핵심 전환**: 분류 주체가 사람이면 체리피킹, 시스템이면 지능. STOP 정확도 = "한계를 아는 시스템" = 정직의 증거.

### sigmoid 예제 = 포화군 실연
```cuda
Y[tid] = 1.0f / (1.0f + __expf(-X[tid]));  // 이미 BW 천장 근접
```
트레이스: `bw_pct≈0.85, compute_tput≈0.15, load_eff≈1.0`
→ 룰 매칭: `memory_saturated` → **"STOP, 손댈 것 없음."** (변형해도 안 빨라짐. 정직한 출력.)

## 5. 스코프 & 성공 기준

### MVP (1차)
- **문제**: LeetGPU Easy 전체 또는 무작위 N개(~15~20), 고르지 않음.
- **LLM**: 단일 API 모델 1개 (DeepSeek 또는 Codex, 초반 가벼운 테스트로 결정). 멀티-LLM/폴백사다리 **폐기.**
- **GPU 학습**: 시드룰 5~6개 + CUDA 멘탈모델 최소 = **3~5일, 읽기 전용.**
- **측정**: 루프 신호 = **Colab Pro A100 + ncu**(무제한). percentile 검증 = LeetGPU Pro(최종 커널만, 가끔).
- **포함**: 생성루프 + Correctness Gate + Trace Parser(ncu 파싱) + 시드 Hypothesis Engine + Rule Evolver(경량) + Ledger + 2트랙 곡선.
- **제외 (2차)**: RAG/패턴DB, 멀티-LLM, Medium/Hard, 사이트 크롤링, 원리노트.

### 성공 기준 (2축)
- **(a) 최적화 가능군**: 평균 **ncu 메트릭(occupancy/BW) 개선** 우상향 곡선 1장 (+ 가설 로그). 최종 커널 일부는 LeetGPU percentile로 교차검증.
- **(b) 포화군**: STOP 판정 정확도 (시스템이 "불가"라 한 게 실제로 맞았나).
- **합격선**: 가능군에서 *최소 3~5문제*가 라운드 진행에 따라 메트릭 상승 + 그 상승에 근거 로그 부착.
- **메타루프 증거**: 룰 신뢰도가 실제로 갱신됐다는 before/after 1장 (없으면 Rule Evolver는 장식).

### 기간 (대략)
- 4~6주 (MVP). GPU 학습 3~5일 선행 → 하네스 → 루프 → 곡선.

## 6. 미해결 / 리스크

| 축 | 내용 | 처리 |
|---|---|---|
| ~~Pro 트레이스 필요성~~ | ✅ **해소 ([[00-measurement-feasibility]], 판정 A).** 무료 LeetGPU는 통과/실패만 주지만, Colab Pro A100+ncu가 메트릭(dram_throughput 83%, warps_active 81%) 추출 성공. 신호원 확보. | 신호 = Colab ncu(무제한), percentile = LeetGPU Pro(가끔). 잔여: Pro 제출 제한 수치 / A100 크레딧 변동 |
| **라이선스** | AlphaGPU = CC BY-NC-ND. 포폴(비상업)이라 NC 회색지대. 단 challenge.py 직접 사용 시 ND 충돌 가능 | 우회안 C(자체 테스트 생성) 기본. AlphaGPU 직접 의존 최소화 |
| **메타루프 수렴 증거** | Rule Evolver가 "실제로 개선됐다"를 못 보이면 장식 | 성공기준 (b)에 before/after 명시 |
| **STOP 오판** | 포화군인데 사실 최적화 여지 있는 문제를 STOP 처리 = 가짜 정직 | STOP 판정을 수동 스팟체크(1~2개 직접 검증) |
| **GPU 멘탈모델 0 리스크** | 시드룰 근거를 못 쓰면 전체 신뢰 붕괴 | 3~5일 학습은 회피 불가. 착수 전제조건 |

## 7. 폐기된 것 (원 컨셉 대비)

- 멀티-LLM 폴백 3단 사다리 → 단일 LLM.
- RAG/패턴DB MVP 포함 → 2차로.
- "속도 상위권" 절대 목표 → "ncu 메트릭 *개선* 곡선 + percentile 검증점"으로 완화 (Task 0 반영, 상위권 자체 아님).
- 컴파일러/perf 직군 어필 → AX 메인 + 저수준 적용력 보조.
- 문제 선별 → 전수 + 자동 2트랙 분류.
- **옛 사양서 [[HANDOFF-SPEC]]** (2026-06-18, Hermes 핸드오프) → 이 설계가 대체. §0 상위권 목표·§2 멀티-LLM 폴백은 폐기, §9.5 저장소 구조·§9.6 동시편집 규칙은 구현 시 재참조용으로 유효.
