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

## A 결과 (2026-06-26 실행) — PASS + 룰 결함 발견 ★

3문제 공유 rules 각 1라운드, 전부 passed. 발화 표:

| 문제 | 발화 룰 | idx | 정당? |
|---|---|---|---|
| llama | `tensorcore_saturated` [STOP] | 2 | ✅ tensorcore on + 느림 = 포화 정직 종료 |
| sigmoid | `fp32_no_tensorcore` | 1 | ❌ **오발화** — 메모리바운드(bw 0.668)인데 "matmul TF32로" |
| groupnorm | `fp32_no_tensorcore` | 1 | ❌ **오발화** — matmul 없는데 텐서코어 룰 |

판정: 2종 라벨 → 스크립트 ✅ PASS. **단 2/3 오발화 = A의 진짜 수확.**

### 결함 진단 (B의 직접 입력)
`fp32_no_tensorcore` 조건 (rules.py:57):
```python
cond = t.weight_pct >= WEIGHT_GATE and t.compute_tput > 0 and not t.tensorcore_active
```
- sigmoid/groupnorm = `tensorcore_active=false` + `compute_tput>0` → **조건 통과**.
- 근데 둘 다 메모리바운드 → "matmul TF32" 가설 부적절 (sigmoid엔 matmul 없음).
- **왜 가로챘나**: priority 1(텐서코어) < 2(융합). `match`가 priority 오름차순 →
  fp32 룰이 memory_bound_fusable보다 먼저 선점. 메모리 신호인데 텐서코어 룰이 이김.
- 진짜 발화해야: sigmoid(bw 0.668) → `memory_bound_fusable`, groupnorm(bw 0.287) → None(조건 미달).

### 의미 — 버그 아니라 차별점 재료
정적 시드룰이 **메모리바운드 워크로드를 오분류** = 다문제가 드러낸 결함.
CUDAMaster류(정적 룰)는 영원히 이 오발화 유지. **우리 evolver는 측정 피드백으로
고칠 수 있음** = 차별점의 실제 동작 시나리오 = **B의 타깃.**
단 A는 1라운드씩 → evolver 아직 안 고침(success/fail 미축적). 고침은 B(다라운드).

## B — 진화 이득 입증 (gain layer) — A 결함을 evolver가 고치는 시나리오

> A가 구체 타깃 줌: fp32_no_tensorcore의 메모리바운드 오발화. B = evolver가 이걸 고침.

### B 핵심 시나리오 (차별점 실증)
1. **다라운드 코드변형** (RealGenerator LLM, 현 FixedGenerator stub 대체):
   sigmoid/groupnorm에 fp32_no_tensorcore 가설("TF32") 적용 → **개선 없음**
   (메모리바운드엔 matmul 없음) → `improved=False` → evolver `demote`.
2. **반복 실패 누적**: 메모리바운드 문제서 fp32 룰 계속 fail → confidence 하락 →
   4회 실패 시 `retire_pass`가 **폐기** (evolver.py:50, RETIRE_AFTER_FAILS=3).
3. **정적 baseline 대비**: 진화 OFF(seed 고정)면 같은 오발화 영원 반복 →
   매 라운드 헛 가설. 진화 ON이면 오발화 룰 강등/폐기 → 올바른 룰로 수렴.
   **= gain (한 문제 아닌 3문제 평균 수렴 속도/헛라운드 수).**

### B 구현 항목
- [ ] **RealGenerator** — 가설 프롬프트 받아 solve.py 변형 (LLM API glue). 핵심·큰 작업.
- [ ] **진화 ON/OFF 비교 드라이버** — `run_multiproblem`에 `evolve_enabled` 플래그.
      OFF = seed 고정(rules 매 라운드 재생성, evolve 스킵). ON = 공유+evolve.
- [ ] **GT 정의** — "올바른 발화" 정답표 (sigmoid→memory_bound_fusable 등) →
      진화 후 발화가 GT에 수렴하는지 측정.
- [ ] (조건부) `propose_candidate` 진짜화 — groupnorm(None)처럼 시드룰 빈 곳 메우기.
      A서 groupnorm 본래 None이어야 = 빈 곳 존재 신호. 단 현재 fp32가 가로채 None 안 뜸.

### B 경계 (정직)
- ⚠️ MOC 교훈: 한 문제 데이터로 일반 우수성 주장 금지. 3문제 GT 필수.
- B = gain layer. A 결함 고침이 **정적 대비 측정 이득**이어야 차별점 입증.
- RealGenerator(LLM) = 비용 큼. 1차 공개 전 에이전트 수행(user 승인).

## 경계 (정직화)
- A = **concept layer** (신호→다른 룰). 단단하나 이득 아님.
- B = **gain layer** (진화가 정적보다 나음). 미입증, 큰 작업.
- A PASS ≠ 차별점 입증. A는 "B가 의미 있는 환경인지" 게이트일 뿐.

## 링크
- [[03-git-mailbox-runner]] — 인프라 (e2e PASS)
- [[02-prior-art-survey]] — 차별점 정직화 (concept/gain 분리)
- [[GPU-Solver-MOC]]
