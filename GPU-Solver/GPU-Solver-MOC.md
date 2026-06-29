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
| [[04-multiproblem-round-design]] | 다문제 라운드 (A 발화관찰→B 진화) — 차별점 실험 | 🟢 메커니즘 확정(진화 ON=틀린룰 retire→전환, 실 A100) / 성능 gain 3문제 전부 null=환경 한계, 추격 중단 |
| [[05-portfolio-differentiator]] | **포트폴리오 차별점 서술** (방법론 톤, 청중별 포지셔닝) | 🟢 active |
| [[05-portfolio-differentiator-EN]] | 위 노트 영문판 (깃헙 README·해외 채용용) | 🟢 active |
| [[06-cudaforge-comparison-design]] | CudaForge 성능 비교 실험 설계 (판정 주체만 통제변수, H1성능/H2품질 분리) | 🔵 design |
| [[07-chip-lang-context-design]] | 칩×문법 컨텍스트 설계 — 환경을 룰 1급 입력으로 (칩=1급/문법=신호흡수, 사용자는 무대만·룰은 측정진화) | 🔵 design (미구현) |
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
- ❌ **성능 gain 3문제 전부 null → 추격 중단 (2026-06-28)** — 환경 한계로 확정 ([[PROGRESS]] 미입증 표). sigmoid(메모리바운드 ±1%), groupnorm(단일·welford·**분할병렬 3알고리즘** 전부 ~36ms ±1%, DRAM BW 천장), llama(**TF32 OFF=24320us vs ON=24352us, 1.00×** — TF32 실적용됐으나 err 2e-7→2.9e-4만 변동, latency 무효). **원인 = 측정 환경(git-mailbox Colab) ≠ PoC**: llama 24ms = PoC 1.4ms의 **17배** → 현 환경선 matmul 비병목(SDPA flash 지배 추정). **루프 결함 아님 — 어떤 문제도 이 환경서 최적화 여지 안 보임.** 인프라 부수: `run_gain_hypcond.py`(가설-조건부 콜백, 큐 한계 극복), groupnorm/llama variants 커밋(`1424053`). 분할병렬 버그 교훈: mask에 `idx<tile_end` 필수.
- ✅ **차별점 확정 = 메커니즘 (2026-06-28)** — 룰표가 측정 피드백으로 진화(오발화 retire→정룰 전환)는 진짜 A100으로 닫힘. 성능 gain(진화→더 빠른 커널)은 **환경 한계로 future work**. 포트폴리오 서술 = "측정 피드백 진화 메타루프" 메커니즘 입증 + 성능 gain은 인프라 준비됨(환경 갖춰지면 측정 가능).
- ✅ **gain layer 첫 돌파 (2026-06-29)** — "환경 한계"는 측정 버그였음(`_profile_ncu --launch-count 1` = 커널 1개만 측정). 고침(전체 duration 합산) 후 matmul 재측정: **TF32 7.2× 실재**(OFF `ampere_sgemm` 9.4ms vs ON `cutlass_tensorop` 1.5ms). 진화 라운드: 루프가 R0(fp32 9.6ms)→`fp32_no_tensorcore` 발화→R1(TF32 1.5ms) = **6.4× gain 실측 = "측정→가설→재작성→더 빠른 커널" 첫 실증.** 단 진화 ON>OFF 차이는 없음(첫 발화가 맞는 룰→retire 불요). **차별점 두 축 분리 확정: gain 축=matmul 6.4×, 진화 축=sigmoid 오발화 retire. 일반화는 future work.** 부수: `executor.ncu_breakdown`·`_profile_ncu` 수정·`problems/matmul/`.
- ⏭️ **두 축 동시(한 문제서 retire+gain) = 구조적 future work (2026-06-29)** — 비합착 Triton matmul(`problems/matmul_tri/`)로 시도했으나 `tl.dot`이 TF32 텐서코어 자동(tc=True) → fp32룰 발화 막힘, uncoalesced 1순위 자연 발화 = 또 첫 발화가 맞는 룰. 근본 = **시드 룰이 워크로드에 잘 맞아 retire 안 생김**(역설). 틀린 룰 자연 발화=메모리바운드(gain 천장) vs 맞는 룰=compute(retire 없음) = 상호배타. 연출 없이 한 문제서 둘 다 불가 확정. = 차별점 두 축은 각각 단일문제 입증으로 확정, 동시·일반화는 정직히 future work.
- ✅ **포트폴리오 차별점 노트 작성 (2026-06-28)** — [[05-portfolio-differentiator]]. 방법론 톤(가설→ablation→측정 검증→claim 레이어 분리)으로 청중별 포지셔닝 확정. 핵심: 어필 축 = 하네스 정교함 아니라 **통제 실험 설계 + negative result 정직 측정.** 메커니즘/이득 레이어 경계를 넘지 않는 서술 = 초보여도 격하 방어.
- ✅ **다신호 확장 + CudaForge 비교 설계 (2026-06-29)** — signals.py에 nsys(`launch_gap_pct`)·torch profiler(`op_name/op_weight/op_shape`) 파서 추가(기존 ncu 보존, 3소스 합성). rules.py에 새 룰 2개(`launch_overhead`=nsys, `attention_dominant`=torch) — 워크로드 가드 포함. self-check 전부 PASS. **왜 여태 ncu만? = 룰 신호가 전부 커널 내부 메트릭이라 nsys 자리 안 만듦(spec-구현 드리프트, 정정).** [[06-cudaforge-comparison-design]] 작성 — 판정 주체(우리 룰 vs CudaForge Judge LLM)만 통제변수, H1(성능, 불확실)/H2(품질·결정론·재현, 차별점 본질) 분리. ⚠️ executor 측정 배선·judge arm 미구현(설계만).
- ⏭️ 다음 (재개 시 선택) — [[PROGRESS]] §다음 할 일: (1) 06 실험 실행(executor에 nsys/torch 측정 배선→Colab 재시작 1회 + judge arm 재구현), (2) 포트폴리오 마무리, (3) 다문제 gain 일반화.
  - (선택) R8 flash_attn 48.6% 추가 최적화 — 알고리즘 곁가지, 차별점 무관.

## 폐기/역사

- [[HANDOFF-SPEC]] — 옛 Hermes 핸드오프 사양. 설계 스펙이 대체. 단 §9.5 저장소 구조 / §9.6 동시편집 규칙은 구현 착수 시 재참조용으로 유효.
