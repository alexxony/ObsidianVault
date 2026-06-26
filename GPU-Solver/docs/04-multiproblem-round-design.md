---
title: 다문제 라운드 설계 (A→B 차별점 실험)
created: 2026-06-26
tags: [gpu-solver, experiment, differentiator, evolver]
status: design
---

# 다문제 라운드 — 차별점(룰 진화) 실험 설계

> 🧭 프로젝트 급소. 3문제(llama/sigmoid/groupnorm) 서로 다른 신호로 evolver가
> 작동하는지 실증. **A(발화 관찰) → B(진화 이득)** 순차.
> 근거: [[03-git-mailbox-runner]] e2e PASS, [[02-prior-art-survey]] 차별점 정직화, MOC §현재위치.

## 전제 (확보됨)

3문제 e2e PASS, 서로 다른 신호 프로필 (= 측정 파이프가 워크로드 구별):

| | llama | sigmoid | groupnorm |
|---|---|---|---|
| 특성 | 컴퓨트바운드 | 메모리바운드 | reduction (중간) |
| tensorcore_active | **true** | false | false |
| bw_pct | 0.0003 | **0.668** | 0.287 |
| compute_tput | 0.628 | 0.696 | 0.640 |
| weight_pct | 1.0 | 1.0 | 1.0 |

## 코드가 드러낸 선결 (설계 전 진단)

evolver.py / rules.py / runner.py / harness.py 정독 결과 — 단순 "run_problem 3번
호출"로는 진화 안 됨. 2개 구조적 사실:

### 선결 1 — 룰 공유 (작은 변경)
- 룰 진화(success/fail 누적)는 **rules 객체가 문제 간 공유돼야** 일어남.
- `harness.run_loop(..., rules: list[Rule] | None = None)` — **이미 rules 주입 받음.**
- `runner.run_problem`이 그걸 **안 넘김** (파라미터 없음) → 호출마다 `seed_rules()` 새로 생성 → 진화 0.
- **수정**: `run_problem`에 `rules=None` 파라미터 추가 → `run_loop(... rules=rules)` 통과. 끝.

### 선결 2 — 코드 변형 (A는 불요, B는 필수)
- `FixedGenerator` = solve.py 고정 → 매 라운드 같은 signal → metric 정체 → 1~2라운드 수렴.
- 고정 코드면 `improved` 변동 없음 → promote/demote/retire/propose **전부 안 일어남**.
- **A**: 코드 변형 불요 (각 문제 1라운드, 발화 룰만 관찰).
- **B**: RealGenerator(LLM) 필수 (라운드마다 코드 변형 → improved 변동 → 진화 발생).

## A — 룰 발화 관찰 (concept layer, 오늘 가능)

### 목표
"측정 신호가 문제별로 **다른 룰**을 깨우나?" 입증/반증.
- 다 같은 룰 깨우면 → 진화해도 무의미 → B 방지.
- 다른 룰 깨우면 → 진화가 의미 있는 환경 → B 진행.

### 방법
1. **공유 rules 1개** = `seed_rules()` (외부 생성).
2. 3문제 각 `run_problem(problem, solve.py, ..., rules=공유, max_rounds=1)`.
3. 각 라운드 ledger에 `rule_idx`/`hypothesis_label` 기록됨 (ledger.py 기존 필드).
4. **관찰 표** = 문제 → 발화 라벨 → STOP 여부.

### 예상 (rules.py 조건 대입 — 검증 필요, 추측 아님 실측으로 확인)
| 문제 | signal | 예상 발화 룰 | 근거 |
|---|---|---|---|
| llama | tensorcore **on**, bw 0.0003 | `tensorcore_saturated`? (latency>50) 또는 None | tensorcore_active=true → fp32 룰 조건 탈락 |
| sigmoid | bw 0.668, tc off | `memory_bound_fusable` (bw>0.5) | bw>0.5 + tc off + weight≥5% |
| groupnorm | bw 0.287, tc off | None? (bw<0.5, tc off, 비융합) | 어느 조건도 안 맞을 수 있음 → propose 후보 신호 |

⚠️ **이 표는 예상.** 실제 발화는 A 라운드 ledger로 확정. groupnorm이 None이면
= 시드룰 빈 곳 = propose_candidate 진짜화 동기 (B의 재료).

### 산출
- `loop/run_multiproblem.py` — 3문제 순차 + 공유 rules + 발화 표 출력.
- ledger `multiproblem-ledger.jsonl` — 문제별 발화 기록 (공개 재료 후보).

### A 판정
- **다른 라벨 ≥2종 발화** → concept layer 실증, B 진행.
- **전부 같은 라벨/전부 None** → 신호 구별이 룰로 안 이어짐 → 룰 조건 재검토 (B 전에).

## B — 진화 이득 입증 (gain layer, 큰 작업 — A 후 판단)

> A 결과 보고 설계. 골격만.

- **RealGenerator(LLM)** = 가설 프롬프트 받아 solve.py 변형 (현 stub 대체).
- 다라운드 × 3문제 × 공유 rules → improved 변동 → promote/demote/retire/propose 실발생.
- **비교**: 진화 ON vs 정적(진화 OFF, seed 고정) → 같은 라운드 예산서 수렴 속도/최종 metric.
- gain = 진화 ON이 측정 이득 (한 문제 아닌 3문제 평균).
- ⚠️ MOC 교훈: 한 문제 데이터로 일반 우수성 주장 금지. 3문제 GT 필수.
- ⚠️ `propose_candidate` = 현 stub (`cond=lambda:False`). B서 진짜 조건 합성 필요할 수 있음 (A가 None 다발이면).

## 경계 (정직화)
- A = **concept layer** (신호→다른 룰). 단단하나 이득 아님.
- B = **gain layer** (진화가 정적보다 나음). 미입증, 큰 작업.
- A PASS ≠ 차별점 입증. A는 "B가 의미 있는 환경인지" 게이트일 뿐.

## 링크
- [[03-git-mailbox-runner]] — 인프라 (e2e PASS)
- [[02-prior-art-survey]] — 차별점 정직화 (concept/gain 분리)
- [[GPU-Solver-MOC]]
