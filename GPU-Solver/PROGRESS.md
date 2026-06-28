---
title: GPU-Solver 진행 현황 (재개 진입점)
created: 2026-06-26
updated: 2026-06-27
tags: [gpu-solver, progress, resume]
---

# GPU-Solver — 진행 현황 (재개용 단일 진입점)

> 🧭 세션 재개 시 이 문서 먼저 읽기. 최신 상태 + 다음 할 일 + 재개 절차.
> 상세는 [[04-multiproblem-round-design]], 허브는 [[GPU-Solver-MOC]].

## 한 줄 요약

GPU 커널 최적화 루프. 차별점 = **룰표가 측정 피드백으로 진화** (CUDAMaster 등 선행은 정적).
현재: **메커니즘 전부 입증(룰진화·오발화retire·진짜 A100 gain 신호). 성능 gain은 3문제 연속 null —
측정 환경(git-mailbox Colab)이 PoC(1.4ms)와 달라(llama 24ms=17배) 어떤 문제도 최적화 여지 안 보임.**
차별점 확정 = **메커니즘 OK, 최종 성능 gain은 환경 한계로 future work.**

## ✅ 입증된 것 (전부 push·검증)

| 단계 | 상태 | 증거 |
|---|---|---|
| 인프라 (git-우편함) | ✅ | e2e PASS, 로컬↔Colab(A100) 자동 왕복 |
| 측정 (ncu `--launch-count 1`) | ✅ | 6배↓, 룰 신호 8개 유지 |
| 3문제 신호 프로필 | ✅ | llama(컴퓨트)·sigmoid(메모리)·groupnorm(reduction) 서로 다름 |
| A 발화 관찰 | ✅ | 신호가 다른 룰 깨움 + **fp32 오발화 발견** |
| B 옵션1 (진화 메커니즘) | ✅ | ON=retire→전환, OFF=영원 오발화 (fake 신호) |
| B 대행 GPU 닫힘 | ✅ | **진짜 A100** — TF32 헛방 실측 → fp32 룰 demote |
| **gain 신호 (진화 ON/OFF 비교)** | ✅ | **진짜 A100 14R** — sigmoid서 ON=fp32 retire→memory_bound_fusable 전환, OFF=fp32 ×6 영원 (커밋 `run_gain_compare.py`) |

## ⬜ 미입증 (= 차별점 급소)

- **성능 gain** — "진화→더 빠른 커널". **3문제 전부 null (실측 확정):**
  | 문제 | 시도 | 결과 |
  |---|---|---|
  | sigmoid | `--latency` 12R, BLOCK 변형 | null — 메모리바운드 ~770ms ±1% |
  | groupnorm | 단일블록·welford 1패스·분할병렬 3알고리즘 | null — 전부 ~36ms ±1%, DRAM BW 천장 |
  | llama | TF32 OFF vs ON (Event latency) | null — **OFF=24320us, ON=24352us (1.00×)** |
  - **원인 = 측정 환경 ≠ PoC.** llama 24ms = PoC 1.4ms의 **17배.** TF32는 실제 적용됨
    (err 2e-7→2.9e-4 변동 = 저정밀 확인)나 latency 무효 → **현 환경선 matmul 비병목.**
  - ✅ **(2026-06-29) ncu 비중 실측으로 확정** (`executor.ncu_breakdown` 신규, 전체 forward 23커널 그룹 집계):
    **attention(fmha_cutlassF_f32) 54% > matmul(cutlass gemm) 23% > elementwise 23%.** flash attention 단일 커널이
    전체의 ~48%(547us/1132us) 지배. = **"SDPA flash 지배" 추정 → 데이터 확정.** matmul 비병목이라 TF32 OFF/ON 무효가 당연.
    추가: fmha가 **fp32 폴백**(MemEffAttention) = attention 자체가 느린 주범. 진짜 여지 = matmul TF32 아니라 attention bf16 flash,
    단 challenge fp32 정확도 요구 시 불가 → **이 환경/문제선 최적화 여지 없음 확정.** PoC R5(1.71×)는 다른 환경.
  - = **루프 결함 아님. 어떤 문제도 이 측정 환경서 최적화 여지 안 보임** (문제·환경 선택 문제).
- **다문제 일반화** — gain 신호(오발화retire)는 sigmoid 1문제만.
- 차별점 현 상태 = **"개념·메커니즘·룰진화 이득 OK, 최종 성능 gain은 환경 한계로 future work"** (정직 유지).

## 🧪 성능 gain 인프라 (재시도용 — 환경 갖춰지면)

- `loop/run_gain_hypcond.py` = **가설-조건부 콜백 드라이버** (run_gain_compare 큐 한계 극복).
  콜백이 발화 룰 라벨 보고 코드 선택 → 진화 차이가 코드 차이로 이어짐 → 성능 gain 갈림 측정 가능.
- groupnorm variants: `R_tf32.py`(fp32 가설 충실, null), `R_coalesced.py`(분할병렬, correctness PASS·latency 무효).
  - ⚠️ 분할병렬 버그 교훈: `mask = idx < group_elems`만이면 BLOCK>tile 시 타일 중복 합산.
    **`mask = (idx<group_elems) & (idx<tile_end)`** 필수. NTILE=1 PASS·NTILE=8 FAIL로 격리.
- llama variants: `R_seed_tf32off.py`(느림 base), `R_tf32on.py`(빠름)... 단 환경서 동률.
- baseline 신호 진단: `scratchpad/profile_baseline.py`, headroom: `scratchpad/check_llama_headroom.py`.

## 🔖 다음 할 일 (성능 gain 추격 중단 — 환경 한계 확정)

성능 gain은 3문제 연속 null + 원인이 측정 환경(≠PoC)으로 확정 → **추격 중단, 정직 기록.**
차별점 = **메커니즘(룰진화·오발화retire·측정 닫힘)** 으로 확정. 성능 gain = future work.

**재개 시 선택지 (우선순위):**
1. **(포트폴리오 마무리)** 차별점 = "측정 피드백으로 진화하는 룰표" — 메커니즘 입증 완료로
   서술. 성능 gain은 "환경 갖춰지면 인프라 그대로 측정 가능"(run_gain_hypcond.py)으로 정직히 표기.
2. **(원인 확정 원하면)** llama ncu 비중 분석 1R (~10분) — matmul %가 작으면 TF32 무효 = SDPA flash
   지배 확정. `run_gain_round.py llama <variant>` 또는 직접 ncu 요청.
3. **(환경 재현 원하면)** PoC 1.4ms 환경(SSH Colab+do_bench) 복원 시도 — 현 24ms와 17배 차 원인.
   단 git-mailbox 안정성↑이라 권장 안 함.
- ⚠️ mailbox 청소 = `cmd/* result/* done/*` 전부 (done 마커 확장자 없음).
- harness latency mode = `--latency` 플래그 (metric=-latency_us, best=최저). occupancy 기본 보존.
- ⚠️ ncu=True인데 llama 24ms = Event fallback 동작 추정 (ncu 권한? executor가 None→Event). 확인 필요.

## 🛠️ 재개 절차

### 코드 위치
- 로컬: `~/workspace/gpu_solver_test/` (= `alexxony/gpu-solver-loop` **master**)
  - `loop/` = 인프라 (executor/watch/runner/mailbox/signals/rules/evolver/ledger/generator).
  - `loop/run_gain_round.py` = 대행 gain 1스텝. `loop/run_multiproblem.py` = A 발화관찰.
  - `loop/run_evolution_compare.py` = 진화 ON/OFF (fake). `loop/run_e2e.py` = 단일 e2e.
  - `problems/{llama,sigmoid,groupnorm}/solve.py`.
- 우편함: `~/workspace/gpu-mailbox` (= `alexxony/gpu-mailbox` **main**). loop=master — 주의.
- 로컬 python = `python3` (torch 없음 — gate/ncu는 Colab만).

### Colab 재기동 (닫았으면)
1. colab.research.google.com → 노트북 열기 (GitHub 탭 → `alexxony/gpu-solver-loop` → `colab_mailbox.ipynb`).
2. 셀1(nvidia-smi A100 확인) → 셀2(clone, PAT=Colab Secrets `GPU_MAILBOX_TOKEN`) → watch 셀.
3. watch 셀 = `max_iters=None` (무한 — 한 번 띄우면 계속). polling 로그 뜨면 active.
4. loop 코드 안 바꿨으면 재clone만, **런타임 재시작 불요**.

### gain 라운드 실행 (watch 살아있으면)
- 1스텝: `python3 run_gain_round.py <problem> <variant_solve.py>`.
- 대행 흐름: 드라이버가 발화 가설 출력 → 에이전트가 변형 solve.py 만듦(scratchpad) → 콜백 주입.

## ⚠️ 운용 교훈 (재발 방지)
- **git push divergent**: watch와 로컬 동시 push 레이스 → `git_sync` 5회 재시도 박음(커밋 `9ef32a0`). push 후 `git rev-parse HEAD origin/<br>` 일치 확인 필수.
- **Colab sys.modules 캐시**: loop 코드(executor/watch/signals) 바꾸면 **런타임 재시작** 필수 (재clone만으론 구버전). 단 **문제만**(새 solve.py/variant=REQ 전송) 바꾸면 재시작 불요. → **마찰 영구해결 길 = watch에 `cmd['raw_script']` 분기(임의 파이썬 subprocess 실행). 넣으면 새 측정 추가 시 executor 안 고쳐 재시작 영원 불요.** 미구현(별도 세션). breakdown은 executor 분기로 했으니 1회 재시작 필요.
- **노트북 ≠ 실험 셀**: `colab_mailbox.ipynb` = watch 전용. 측정/실험 셀 넣지 않음.
- **mailbox 정리**: gain 라운드 후 cmd/result/done clean + commit.

## 비용/환경 메모
- generate(LLM) = API 호출 = GPU 불요 → 로컬. 제품 무인 = RealGenerator + `ANTHROPIC_API_KEY`(user 미발급).
- PoC 입증 = CallbackGenerator(대행, 키 0). user = Claude 구독만(API 지갑 별도).
- A100 저활용 = 작은 커널 PoC 본질 (낭비 아님). 풀활용 원하면 로컬 LLM(Qwen) — 설정 큼, 보류.

## 링크
- [[05-portfolio-differentiator]] — 포트폴리오 차별점 서술 (방법론 톤, 청중별 포지셔닝)
- [[05-portfolio-differentiator-EN]] — 위 영문판 (깃헙 README·해외 채용용)
- [[GPU-Solver-MOC]] — 문서 허브
- [[04-multiproblem-round-design]] — A/B 실험 상세
- [[03-git-mailbox-runner]] — 인프라 설계 + e2e
- [[02-prior-art-survey]] — 차별점 검증
- [[2026-06-22-agentic-gpu-optimizer-design]] — 설계 스펙
