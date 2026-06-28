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

## B 옵션1 결과 (2026-06-26 실행) — 메커니즘 실증 ★

LLM·GPU 0, fake responder(sigmoid 실측 신호 고정 주입)로 진화 ON/OFF 비교.
`loop/run_evolution_compare.py`. fp32_no_tensorcore 오발화가 측정 피드백으로
폐기되는지 = 차별점 메커니즘 검증.

| | OFF (정적, CUDAMaster류) | ON (진화) |
|---|---|---|
| 발화 변천 | fp32×4 → converged 종료 | fp32×4 → **retire** → memory_bound_fusable×2 |
| fp32 룰 | 영원히 발화 (폐기 불가) | demote×3 → **RETIRE** |
| 이벤트 | 없음 | promote, demote×3, retire, demote×2 |

**핵심: ON/OFF 갈림.** OFF는 오발화 영원 반복. ON은 측정 fail 피드백으로 fp32
폐기 → 올바른 메모리룰 전환. = CUDAMaster류 정적 룰에 없는 메타루프 실동작.

### 발견·수정 3결함 (전부 진짜 버그)
1. **retire 경계** `conf < 0.25` → `<=` (demote 3회 정확히 0.25서 한 끗 미달).
2. **converged 조기종료** — retire/propose 시 `no_improve` 리셋 (폐기 후 발화 변경
   관찰 가능하도록). harness.
3. **가짜 OFF** — `rules=None`은 정적 아니라 1객체가 진화함. `evolve_enabled`
   플래그 신설 → 진짜 정적 baseline (evolve 스킵). harness/runner.

self-check 5개(harness/evolver/runner/rules/ledger) 전부 PASS = 기존 계약 무파괴.

### ⚠️ 경계 (과대주장 금지)
- 이건 **mechanism layer** — demote/retire/전환이 실측 fail로 돈다.
- **gain layer 아님** — memory_bound_fusable도 fake 신호선 fail(success=0). 진짜
  개선(improved 변동)은 RealGenerator가 코드 변형해야 생김 = B 본체.
- **fake responder** — 신호 고정(값만 A 실측 sigmoid). 실 GPU 측정 아님.

### 다음 = B 본체 (RealGenerator)
- LLM이 가설 프롬프트 받아 solve.py 변형 → 진짜 improved 변동 → 정적 대비 수렴
  속도/헛라운드 수 = gain layer. 3문제 GT 필수. 비용 큼.

## B 본체 진행 (2026-06-26) — RealGenerator + 대행 GPU 닫힘 ★

### 결정: API vs 대행
- generate(LLM)=API 호출=GPU 불요 → 로컬 실행, 우편함 프로토콜 무변(code 필드).
- **API 키 결정**: user = Claude 구독만, ANTHROPIC_API_KEY 없음(별도 과금 지갑).
  → 제품 아키텍처 = RealGenerator(API 무인), PoC 입증 = CallbackGenerator(대행, 키 0).
- A100 풀활용하려면 로컬 LLM(Qwen) 필요하나 설정 큼 → **대행으로 gain 먼저** 선택.
  (A100 저활용 감수 — 작은 커널 PoC라 A100 원래 거의 안 씀. 본질은 룰 진화.)

### 구현 (커밋 936f39a, 9a1ca39, 9ef32a0)
- `generator.py`: **RealGenerator**(Claude API, call_fn 주입) + **CallbackGenerator**(대행).
  둘 다 glue.Generator 계약. 시스템 프롬프트가 어댑터 5심볼·reference 보존 강제.
- `runner.run_problem(generator=None)`: None이면 FixedGenerator(기존 무파괴), 주입 시 변형 라운드.
- `run_gain_round.py`: 대행 1스텝 드라이버 (R0원본→R1변형).
- `git_sync` push 5회 재시도 — watch와 로컬 동시 push 레이스(non-fast-forward) 수정.

### gain 1스텝 실측 (sigmoid, 진짜 A100) ★
대행(에이전트)이 발화 룰 `fp32_no_tensorcore` 가설 보고 TF32 변형 → Colab gate+ncu:

| | R0 (원본) | R1 (TF32 변형) |
|---|---|---|
| bw_pct | 0.6705 | 0.6704 |
| compute_tput | 0.6964 | 0.6958 |
| improved | True | **False** |

진화 이벤트 `[promote, demote]`, fp32 룰 success=1 fail=1.

**핵심 입증 (mechanism, 진짜 GPU)**:
1. 대행+GPU 닫힘 작동 — LLM 자리에 에이전트 대행, 코드변형 라운드 완결.
2. **가설 헛방 실측** — TF32 켜도 bw 0.6705→0.6704(거의 동일). sigmoid엔 matmul 없어
   TF32 무효 = **추측 아니라 A100 측정**이 확인. (이전 evolver 실험은 fake 신호였음.)
3. evolve가 진짜 GPU improved=False로 demote — 메커니즘이 실측 데이터로 작동.

### ⚠️ 경계 (과대주장 금지)
- 이건 **mechanism 실데이터 1점** (대행+GPU 닫힘 + improved 실변동). gain layer 아님.
- gain(정적 대비 측정 이득)은 **3문제 × N라운드** 필요 — 1라운드는 demote 1회뿐.

## ✅ gain 신호 첫 실증 (2026-06-27, 진짜 A100)

`run_gain_compare.py` — 같은 variant 큐를 진화 **ON/OFF 두 트랙**에 주입(공정 비교).
sigmoid 8라운드 설정, flat variants(seed 동일 코드 = 측정 노이즈만, "나쁜 코드 조작" 0).

| | OFF (정적, CUDAMaster류) | ON (룰 진화) |
|---|---|---|
| 발화 룰 | `fp32_no_tensorcore` ×6 **영원** | fp32 ×4 → **retire** → `memory_bound_fusable` ×3 |
| retire | 0 | **1** |
| 수렴 | converged(6R) | converged(7R) |

**메타루프 실작동 (진짜 GPU 데이터):**
1. sigmoid(메모리바운드)에 틀린 컴퓨트 룰 `fp32_no_tensorcore` 발화.
2. R1~R4 실측 → 텐서코어 가설로 개선 안 됨 → fail 누적 (fail≥3 & conf≤0.25).
3. ON: **fp32 룰 retire** → 다음 우선순위 맞는 룰 `memory_bound_fusable`(메모리 융합)로 자동 전환.
4. OFF: 똑같이 실패해도 fp32 영원 발화 = 정적 룰표의 한계.

= **"틀린 정적 룰이 측정으로 폐기되고 맞는 룰로 갈아탐"** — 차별점 메커니즘이 실 A100서 닫힘.

### ⚠️ 경계 (여전히 정직)
- 이건 **gain 신호**(틀린 룰 폐기 = 진화가 정적과 다른 결정) 실증. **성능 gain 아님.**
- metric 곡선은 ON/OFF 거의 동일(~0.67) — 같은 코드 큐라 당연. "진화하면 커널 더 빠름"은
  **여전히 미입증** (LLM이 진짜 코드 변형해야 = RealGenerator/API 단계).
- 즉 입증된 것 = **"진화가 틀린 룰을 측정으로 버린다"**. 미입증 = **"그 결과 최종 성능↑"**.
- 1문제(sigmoid)만 — 다문제 일반화도 남음.

## 🧪 성능 gain 1차 시도 (2026-06-27, sigmoid --latency) — null 결과

`run_gain_compare.py`에 **latency metric mode** 추가(`--latency`): metric=`-latency_us`
(낮을수록 좋음, best=최저). harness `_metric(sig, mode)` + best 초기 -inf 분기, occupancy 기본 보존.
sigmoid BLOCK 변형 variant(256~8192) 12R 실측:

| | OFF | ON |
|---|---|---|
| best latency | 765184 us | 769952 us |
| 곡선 | 765k~774k (평평) | 769k~781k (평평) |

**성능 gain ❌ (예상 적중)**: ON best가 OFF보다 오히려 느림. BLOCK 변형이 latency 안 바꿈(±1% 노이즈).
- 원인 = **sigmoid 메모리바운드 = 최적화 여지 없음.** BW 이미 포화 → BLOCK 무관.
- 룰 진화 신호(retire 1, fp32→memory_bound_fusable 전환)는 **재현**되나 더 빠른 커널로 안 이어짐.
- 결론: **루프 결함 아니라 문제 선택 문제.** 여지 없는 워크로드선 진화해도 성능 gain 불가.

## ❌ 성능 gain 2차 시도 (2026-06-28, groupnorm + llama) — 3문제 전부 null 확정

여지 큰 문제(groupnorm reduction, llama matmul)로 재시도 → **둘 다 null. 원인 = 측정 환경.**

### groupnorm — 3알고리즘 전부 동일 latency (DRAM BW 천장)
baseline 진단: `fp32_no_tensorcore` **오발화**(matmul 없는데), 진짜 신호 = `load_eff=0.0`(uncoalesced).
PROFILE_SIZE=(16,128,64,64,32)=536MB fp32. **진짜 다른 알고리즘 커널 3종** 작성·실측:

| variant | 알고리즘 | latency | 비고 |
|---|---|---|---|
| baseline | 단일블록 2-pass (X 3로드) | 36288 us | occ 0.51 |
| R_coalesced(welford) | 1-pass 통계 (X 2로드) | 36032 us | 로드 줄여도 무변 |
| R_coalesced(분할병렬) | 3커널 atomic, grid 512→4096 (8배) | 35872 us | occ 0.51 불변 |

**전부 ±1% = 노이즈.** 알고리즘·grid·BLOCK 무관 latency ~36ms 고정.
= **이 크기 fp32는 DRAM BW 천장.** reduction이어도 여지 없음 (큰 텐서 read+write 1회 = 불가피).
- 분할병렬 버그 교훈: `mask = idx<group_elems`만이면 BLOCK>tile 시 타일 중복 합산.
  **`mask = (idx<group_elems) & (idx<tile_end)`** 필수. NTILE=1 PASS·NTILE=8 FAIL로 격리.

### llama — TF32 효과 무효 (환경이 PoC와 다름)
seed=TF32 OFF("highest") vs variant=TF32 ON("high"), Event latency 실측:

| | TF32 OFF | TF32 ON | speedup |
|---|---|---|---|
| latency | 24320 us | 24352 us | **1.00×** |
| max_err | 2.4e-7 | 2.9e-4 | (TF32 실적용 확인) |

**성능 gain ❌.** TF32는 실제 적용됨(err 변동)이나 latency 무효.
- **원인 = 측정 환경 ≠ PoC.** llama 24ms = PoC([[01-hard-loop-poc]] R5) **1.4ms의 17배.**
  현 환경선 matmul 비병목 (SDPA flash 지배 추정, ncu 비중 미확인). PoC R5(1.71×)는 다른 환경.

### 종합 결론 — 성능 gain 추격 중단
| 문제 | 특성 | 결과 |
|---|---|---|
| sigmoid | elementwise | null (메모리바운드, 1차) |
| groupnorm | reduction | null (DRAM BW 천장, 3알고리즘) |
| llama | matmul | null (TF32 무효, 환경≠PoC) |

= **루프 결함 아님. 이 측정 환경(git-mailbox Colab)서 어떤 문제도 최적화 여지 안 보임.**
차별점 = **메커니즘(룰진화·오발화 retire)으로 확정.** 성능 gain = future work.

### 성능 gain 인프라 (환경 갖춰지면 재사용)
- `run_gain_hypcond.py` — **가설-조건부 콜백 드라이버**. 콜백이 발화 룰 라벨 보고 코드 선택
  → 진화 차이가 코드 차이로 이어져 성능 gain 갈림 측정 가능 (run_gain_compare 큐 한계 극복).
- variants: `groupnorm/variants/R_tf32.py`·`R_coalesced.py`, `llama/variants/R_seed_tf32off.py`·`R_tf32on.py`.

## ✅ 성능 gain 3차 — matmul 6.4× 첫 돌파 + 측정 버그 정정 (2026-06-29)

**2차 "3문제 전부 null = 환경 한계" 결론이 측정 버그로 판명 → 반전.**

- **llama ncu 비중** (`executor.ncu_breakdown` 신규): attention(flash) 54% > matmul 23%, flash 단일 48% 지배.
  → llama TF32 무효는 "환경 한계" 아니라 **matmul 비병목**(attention 지배)이라서. 데이터 확정.
- **순수 matmul**(`problems/matmul/`, 4096² fp32) TF32 OFF/ON: 처음 `_profile_ncu` latency 95ms **동률**="환경 TF32 못 냄" 오결론.
- **🐛 측정 버그 정정** — `_profile_ncu --launch-count 1`이 **커널 1개만** 측정(matmul 아닌 엉뚱한 커널). 전체 duration 합산으로 고침
  (커밋 `62235bb`). 재측정: **OFF=`ampere_sgemm` 9.4ms vs ON=`cutlass_tensorop` 1.5ms = 6.4× 실재.** 커널명·FLOP 검산 확인.
- **✅ 진화 gain 라운드** (`run_gain_hypcond matmul`): 루프가 R0(fp32 9.6ms)→`fp32_no_tensorcore` 발화→R1(TF32 1.5ms).
  = **"측정→가설→재작성→더 빠른 커널" 첫 실증 (gain layer 돌파).** R1 후 `tensorcore_saturated`→stop = 정상 종료 판정.
  단 진화 ON≈OFF (matmul 첫 발화가 맞는 룰 → retire 불요).
- **두 축 동시 시도→구조적 불가** — 비합착 Triton matmul(`problems/matmul_tri/`) probe: **tc=True**(tl.dot TF32 자동)+load_eff=0.0
  → 또 첫 발화가 맞는 룰(uncoalesced). 시드 룰이 워크로드에 잘 맞아 retire 안 생김(역설). 틀린룰 자연발화=메모리바운드(gain 천장)
  vs 맞는룰=compute(retire 없음) = **상호배타.** 연출 없이 한 문제서 둘 다 = 불가 확정.

**= 차별점 두 축 각각 단일문제 실증: gain=matmul 6.4×, 진화 retire=sigmoid. 동시·일반화 future work.**

## 🔖 다음 세션 착수 (재개 시 선택)
차별점 두 축 각각 입증 완료. 재개 선택지 — [[PROGRESS]] §다음 할 일:
1. **(포트폴리오 마무리·권장)** 노트 한/영·README 다 gain 반영됨. public 전환 가능(민감정보 clean).
2. **(다문제 gain 일반화)** matmul 외 compute-bound 문제서도 gain 일관 확인 = gain 축 일반화.
3. **(두 축 동시 재설계)** rule-fire 시점 진짜 모호한 워크로드 찾기 — 상호배타 우회 필요.
- ⚠️ mailbox 청소 = `cmd/* result/* done/*` 전부 (done 마커는 확장자 없음).
- ⚠️ ncu=True인데 llama 24ms = Event fallback 동작 추정 (executor가 ncu None→Event). 확인 필요.
- 제품 무인화 = RealGenerator + ANTHROPIC_API_KEY(user 발급 시).

## 경계 (정직화)
- A = **concept layer** (신호→다른 룰). 단단하나 이득 아님.
- B = **gain layer** (진화가 정적보다 나음). 미입증, 큰 작업.
- A PASS ≠ 차별점 입증. A는 "B가 의미 있는 환경인지" 게이트일 뿐.

## 링크
- [[03-git-mailbox-runner]] — 인프라 (e2e PASS)
- [[02-prior-art-survey]] — 차별점 정직화 (concept/gain 분리)
- [[GPU-Solver-MOC]]
