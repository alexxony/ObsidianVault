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
현재: **진짜 A100서 gain 신호 입증 — 진화 ON이 틀린 룰 폐기→맞는 룰 전환, OFF는 영원 오발화.**
남은 핵심 = 성능 gain (진화→더 빠른 커널, = LLM 진짜 코드변형 필요).

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

- **성능 gain** — "진화→더 빠른 커널" (metric 우상향 + ON 우위). 지금은 flat variant(코드 불변)라
  metric 곡선 ON/OFF 동일 = 룰 폐기만 입증, 성능 향상 아님. **LLM 진짜 코드변형 필요.**
- **다문제 일반화** — gain 신호는 sigmoid 1문제만. groupnorm·llama도 ON/OFF 비교 남음.
- 차별점 현 상태 = **"개념·메커니즘·룰진화 이득 OK, 최종 성능 gain 미입증"** (정직 유지).

## 🔖 다음 할 일 = 성능 gain layer

**진짜 코드변형으로 곡선 우상향 + ON 우위 실증:**
1. flat variant 말고 **실제 더 나은 solve.py** 생성 (RealGenerator+API 또는 대행).
2. `run_gain_compare.py <problem> <variants_dir> <max_rounds>` — ON/OFF 자동 비교 (준비됨).
3. groupnorm·llama로 다문제 일반화.
- ⚠️ ncu 왕복 시간 큼 (llama=multi-kernel 10~12분/R). **집중 세션** 권장.
- ⚠️ mailbox 청소 시 `done/*` 전부 (확장자 없음 — json 패턴만으론 안 지워짐).

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
- **Colab sys.modules 캐시**: loop 코드 바꾸면 **런타임 재시작** 필수 (재clone만으론 구버전).
- **노트북 ≠ 실험 셀**: `colab_mailbox.ipynb` = watch 전용. 측정/실험 셀 넣지 않음.
- **mailbox 정리**: gain 라운드 후 cmd/result/done clean + commit.

## 비용/환경 메모
- generate(LLM) = API 호출 = GPU 불요 → 로컬. 제품 무인 = RealGenerator + `ANTHROPIC_API_KEY`(user 미발급).
- PoC 입증 = CallbackGenerator(대행, 키 0). user = Claude 구독만(API 지갑 별도).
- A100 저활용 = 작은 커널 PoC 본질 (낭비 아님). 풀활용 원하면 로컬 LLM(Qwen) — 설정 큼, 보류.

## 링크
- [[GPU-Solver-MOC]] — 문서 허브
- [[04-multiproblem-round-design]] — A/B 실험 상세
- [[03-git-mailbox-runner]] — 인프라 설계 + e2e
- [[02-prior-art-survey]] — 차별점 검증
- [[2026-06-22-agentic-gpu-optimizer-design]] — 설계 스펙
