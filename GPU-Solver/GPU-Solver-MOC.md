---
title: GPU-Solver — Map of Content (MOC)
status: active
created: 2026-06-22
tags: [gpu-solver, moc, index, portfolio]
---

# GPU-Solver — Map of Content

> 🧭 이 프로젝트 문서 허브. 모든 노트의 진입점 + 그래프 중심.
> **한 줄 정의**: GPU 커널 최적화 루프. 분류→재작성 골격은 선행(CUDAMaster)이 풀었고 — 내 기여는 **룰표가 측정 피드백으로 진화한다는 것** (포트폴리오 / AX 메인). 근거: [[2026-06-22-agentic-gpu-optimizer-design]] §1.

> 🔖 **재개는 [[PROGRESS]] 먼저** — 최신 상태 + 다음 할 일 + 재개 절차 1장.

## 읽는 순서 (신규 진입자용)

1. [[agentic-gpu-solver-concept]] — 컨셉 메모 (브레인스토밍 출발점)
2. [[2026-06-22-agentic-gpu-optimizer-design]] — **설계 스펙 (현재 단일 진실 출처)**
3. [[2026-06-22-agentic-gpu-optimizer]] — 구현 계획 (TDD 11태스크)
4. [[00-measurement-feasibility]] — Task 0 검증 결과 (측정 가용성)

## 문서 목록

| 문서 | 역할 | 상태 |
|---|---|---|
| [[agentic-gpu-solver-concept]] | 컨셉/브레인스토밍 메모 | 🟡 brainstorming |
| [[2026-06-22-agentic-gpu-optimizer-design]] | 설계 스펙 (포지셔닝·아키텍처·가설엔진) | 🔵 proposal |
| [[2026-06-22-agentic-gpu-optimizer]] | 구현 계획 (Task 0~10) | 🔵 proposal |
| [[00-measurement-feasibility]] | Task 0 측정 검증 (판정 A) | 🟢 done |
| [[01-hard-loop-poc]] | Hard 최적화 루프 PoC (R0→R2' + §R4~R7 새 스택: flash 4D·TF32로 누적 3.86×, 반증/기각 다수=룰 진화) | 🟢 done |
| [[02-prior-art-survey]] | 사전 탐사 — 선행연구 4종 비교, 차별점 검증 | 🟢 done |
| [[03-git-mailbox-runner]] | Git-우편함 Runner 설계 (터널 제거, cmd/result 비동기) + 의사결정 여정 | 🟢 e2e PASS / 3문제 신호 확보 |
| [[04-multiproblem-round-design]] | 다문제 라운드 (A 발화관찰→B 진화) — 차별점 실험 | 🟢 A PASS + gain 신호 실증(진화 ON=틀린룰 retire→전환) / 성능 gain은 sigmoid null(여지 큰 문제 필요) |
| `~/workspace/gpu_solver_test/loop/` | 자동 루프 골격 (코드, 별도 git repo) | 🟢 골격 done / 글루 stub |
| [[HANDOFF-SPEC]] | 옛 Hermes 사양 | ⚫ deprecated |

## 핵심 결정 추적 (설계 진화)

- **목표 변경**: "런타임 속도 상위권" → **ncu 메트릭 개선 곡선 + percentile 검증점** (Task 0 반영). 근거: [[2026-06-22-agentic-gpu-optimizer-design]] §0, §7.
- **LLM 단순화**: 멀티-LLM 3단 폴백 사다리 → **단일 LLM**. ([[HANDOFF-SPEC]] §2에서 폐기됨)
- **직군 재조준**: 컴파일러/perf 어필 → **AX(에이전트/LLM) 메인 + 저수준 적용력 보조.**
- **측정 아키텍처**: 루프 신호 = Colab Pro A100 + ncu(무제한) / percentile 검증 = LeetGPU Pro(가끔). 근거: [[00-measurement-feasibility]].

## 현재 위치 / 다음

- ✅ Task 0 (측정 가용성) **통과** — 신호원 확보.
- ✅ [[01-hard-loop-poc]] **수동 루프 R0→R2'** — R1 flash 2.03× 적중, R2/R2' GQA 반증 2건(R1 챔피언 유지). ncu로 진짜 병목(elementwise 메모리바운드) 확정. 루프가 측정으로 굴러감 증명.
- ✅ [[02-prior-art-survey]] **차별점 검증 (2라운드) — 절반 죽고 절반 살았다.** ❌ "결정론 룰 라벨→LLM 재작성→측정검증" = CUDAMaster(arXiv 2603.07169)가 이미 함(신규 아님). ✅ **룰DB 진화 메타루프 = 6개 선행 전부 정적, 우리만 = 차별점 후보.** ⚠️ 더 날카로운 후보 = roofline 라벨 coarseness 극복.
- ⚠️ **차별점 정직화 (2026-06-24) — "개념 OK, 이득 미입증"으로 후퇴** ([[2026-06-22-agentic-gpu-optimizer-design]] §1 두 층 분리). 진화 우수성 1차 검증(`verify_evolution.py`, [[02-prior-art-survey]] 미해결1) = **같은 문제 N회전선 임계 거의 불변 → 정적 대비 이득 無.** coarseness도 **(A) 개념층**(roofline에 비중·텐서코어 축 없음=문제 독립, 자명, R3'/R5/R6 증거)은 단단하나 **(B) 이득층**(우리 세분이 더 낫다)은 이 한 문제서만 관찰=미입증. **핵심 교훈: 한 문제 데이터로 일반 우수성 주장 금지.** PoC 핵심 과제 = "추가 축(비중·텐서코어·진화)이 **다문제서** 측정 이득" 입증 (미완).
- ✅ **자동 루프 골격 7파일 — GPU 없이 로컬 self-check PASS.** 코드 = `~/workspace/gpu_solver_test/loop/` (독립 git repo). 순수 로직(★Trace Parser·Hypothesis Engine·Rule Evolver·Ledger) 완결. 진화 = **신뢰도 ±1 + 폐기**까지 실동작(틀린 임계값 0.5→0.0 강등→폐기). 글루(Generator/Gate/Profiler) = stub, Colab 대기.
- ⚠️ **시드룰 정합 + evolver 갭 발견 (2026-06-24)** — 시드룰을 PoC R3~R7 trajectory 룰(비중 게이트·TF32·텐서코어·융합)로 교체(커밋 `ca1d56d`), signals에 weight_pct·tensorcore_active 추가. 4파일 self-check PASS. **단 차별점 핵심(새 룰 학습)은 미구현**: `evolver.propose_candidate`가 룰 조건 합성 못 하고 stub(`lambda:False`). 진짜화 시도 전 우수성 검증→미입증(위 항목)→**보류.**
- ✅ **Colab 터널 + 새 스택 baseline 확정** ([[01-hard-loop-poc]] §R4) — SSH 터널(cloudflared) + `colabrun` 래퍼로 A100-40GB 실측. **flash 4D 재발견**: 이전 세션 "flash 죽음→naive 챔피언" = 3D 호출 버그 오결론, 재반증. `solve()`=flash4d 전환(2.25×, 커밋 `0271aad`).
- ✅ **수동 루프 6라운드 완주 → 누적 3.86×** ([[01-hard-loop-poc]] §R4~R7). 챔피언 = flash4d+TF32 0.857ms (naive 3.244 대비). **큰 이득 2개**: 4D flash(2.26×)·TF32(R5, matmul 천장 52.7% 직격 →3.86×). **반증/기각 4개 = 전부 룰 정밀화**: R3 torch.compile 회귀 / R3' Triton 융합 약적중(커널-39%지만 비중 2.5%<5%) / bf16 동률(TF32 이미 텐서코어) / fused_ffn 승격 측정기각(게이트통과≠승격) / R7 elementwise **착수 전 룰 선험 기각**(R6 학습을 다음 라운드에 적용=헛 라운드 절약). **= 룰DB 진화(차별점)의 실물 증거 다수.** ncu 천장 재편 추적: matmul 52.7%→20.3%, flash_attn 28.7%→48.6%(신1위).
- ✅ **Git-우편함 Runner 양쪽 골격 done (2026-06-24)** ([[03-git-mailbox-runner]]) — 터널 운용 3고통(끊김 로그 노이즈·세션 URL 복붙·배포불가) 진단 → **B+ 결정**: 자동 루프 채널 = git 우편함(비동기 cmd/result JSON), 수동 탐색 = ssh 병존. **로컬측** `mailbox.py` `MailboxProfiler`(glue.Profiler 구현, 커밋 loop `5348fd6`). **Colab측** `watch.py` 골격(폴링/멱등/실패격리, 커밋 `1889a17`) — `execute_request`만 stub(GPU 컴파일/gate/ncu, Colab서 채움). 양쪽 다 sync_fn/execute 주입으로 **GPU·git·네트워크 0 self-check PASS**. 핵심: "연결" 부재 → "끊김" 개념 소멸. 즉답손실≈0. 배포 = repo fork 1개, PAT repo-scope만, Drive 불요. **부수: selfcheck 회귀 수리**(bad_sig stale → differentiator_e2e 부활). **의미: 이 인프라가 "다문제 GPU 다회"를 비동기·무인으로 현실화 = 차별점 실험의 전제조건 해소.** 단 인프라 ≠ 실험 — 오늘 GPU 실측 0회.
- ✅ **gain 신호 실증 (2026-06-27)** ([[04-multiproblem-round-design]]) — `run_gain_compare.py`로 진화 ON/OFF 두 트랙 공정 비교(같은 variant 큐). 진짜 A100 sigmoid 14R: **ON=틀린 fp32_no_tensorcore 룰 4실패후 retire→memory_bound_fusable 자동 전환, OFF=fp32 ×6 영원 오발화.** = "틀린 정적 룰이 측정으로 폐기되고 맞는 룰로 갈아탐" 차별점 메커니즘이 실 GPU로 닫힘. 인프라 부수: 노트북 Colab Secrets 인증 전환(평문 PAT 제거).
- 🧪 **성능 gain 1차 시도 → null (2026-06-27)** — harness latency mode(`--latency`, metric=-latency_us) 추가. sigmoid 12R: **성능 gain 없음** (ON best 769952us > OFF 765184us). 원인 = 메모리바운드라 BLOCK 변형해도 latency ±1% 노이즈 = 최적화 여지 없음. **루프 결함 아니라 문제 선택 문제** (룰 진화는 재현되나 더 빠른 커널로 안 이어짐).
- ⏭️ 다음 — **성능 gain은 여지 큰 문제로**:
  - **(핵심) groupnorm/llama + 진짜 다른 알고리즘 커널** — sigmoid(메모리바운드)는 여지 없어 부적합. reduction(groupnorm)·matmul(llama)은 알고리즘 선택이 latency 좌우. variant = BLOCK 1줄 치환 아니라 진짜 다른 커널(대행, cherry-pick 안 되게 발화 가설 충실). `run_gain_compare.py groupnorm <dir> 6 --latency`. groupnorm 먼저(ncu 빠름), llama는 multi-kernel 10~12분/R.
  - (보조) 성능 gain 입증되면 RealGenerator(API)로 무인화 — 대행을 LLM으로 드롭인 교체.
  - (선택) R8 flash_attn 48.6% 추가 최적화 — 알고리즘 곁가지, 차별점 무관.

## 폐기/역사

- [[HANDOFF-SPEC]] — 옛 Hermes 핸드오프 사양. 설계 스펙이 대체. 단 §9.5 저장소 구조 / §9.6 동시편집 규칙은 구현 착수 시 재참조용으로 유효.
