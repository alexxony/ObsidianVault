---
title: GPU-Solver 진행 현황 (재개 진입점)
created: 2026-06-26
updated: 2026-06-26
tags: [gpu-solver, progress, resume]
---

# GPU-Solver — 진행 현황 (재개용 단일 진입점)

> 🧭 세션 재개 시 이 문서 먼저 읽기. 최신 상태 + 다음 할 일 + 재개 절차.
> 상세는 [[04-multiproblem-round-design]], 허브는 [[GPU-Solver-MOC]].

## 한 줄 요약

GPU 커널 최적화 루프. 차별점 = **룰표가 측정 피드백으로 진화** (CUDAMaster 등 선행은 정적).
현재: **대행으로 진짜 A100 진화 루프 닫힘 증명 완료.** 남은 핵심 = gain layer (진화가 정적보다 측정 이득) 입증.

## ✅ 입증된 것 (전부 push·검증)

| 단계 | 상태 | 증거 |
|---|---|---|
| 인프라 (git-우편함) | ✅ | e2e PASS, 로컬↔Colab(A100) 자동 왕복 |
| 측정 (ncu `--launch-count 1`) | ✅ | 6배↓, 룰 신호 8개 유지 |
| 3문제 신호 프로필 | ✅ | llama(컴퓨트)·sigmoid(메모리)·groupnorm(reduction) 서로 다름 |
| A 발화 관찰 | ✅ | 신호가 다른 룰 깨움 + **fp32 오발화 발견** |
| B 옵션1 (진화 메커니즘) | ✅ | ON=retire→전환, OFF=영원 오발화 (fake 신호) |
| **B 대행 GPU 닫힘** | ✅ | **진짜 A100** — TF32 헛방 실측 → fp32 룰 demote |

## ⬜ 미입증 (= 차별점 급소)

- **gain layer** — "진화가 정적 대비 측정 이득(헛라운드↓/수렴↑)". 1라운드론 못 봄.
- 차별점 현 상태 = **"개념·메커니즘 OK, 이득 미입증"** (정직한 포지션 유지).

## 🔖 다음 할 일 = gain layer 본격

**3문제 × 5~6라운드 대행 gain 루프** (= 18라운드):
1. sigmoid·groupnorm(메모리) + llama(컴퓨트) 각 5~6라운드.
2. 각 라운드 = 에이전트가 발화 룰 가설 보고 solve.py 변형 대행 → Colab gate+ncu → evolve.
3. 진화 ON vs OFF 비교 → 헛라운드 수·수렴 속도 → **gain 입증/반증**.
- ⚠️ 18라운드 = 대행 18왕복 + ncu 18회 = 시간 큼. **집중 세션** 권장.
- 코드 준비됨: `run_gain_round.py`(1스텝) → 다라운드 루프 드라이버로 확장.

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
