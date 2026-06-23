---
title: GPU-Solver — Map of Content (MOC)
status: active
created: 2026-06-22
tags: [gpu-solver, moc, index, portfolio]
---

# GPU-Solver — Map of Content

> 🧭 이 프로젝트 문서 허브. 모든 노트의 진입점 + 그래프 중심.
> **한 줄 정의**: GPU 커널 최적화 루프. 분류→재작성 골격은 선행(CUDAMaster)이 풀었고 — 내 기여는 **룰표가 측정 피드백으로 진화한다는 것** (포트폴리오 / AX 메인). 근거: [[2026-06-22-agentic-gpu-optimizer-design]] §1.

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
| [[01-hard-loop-poc]] | Hard 문제 최적화 루프 PoC (수동 R0→R2' + §R4 새 스택 flash 4D 재발견, 반증 3건) | 🟢 done |
| [[02-prior-art-survey]] | 사전 탐사 — 선행연구 4종 비교, 차별점 검증 | 🟢 done |
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
- ✅ [[02-prior-art-survey]] **차별점 검증 (2라운드) — 절반 죽고 절반 살았다.** ❌ "결정론 룰 라벨→LLM 재작성→측정검증" = CUDAMaster(arXiv 2603.07169)가 이미 함(신규 아님). ✅ **룰DB 진화 메타루프 = 6개 선행 전부 정적, 우리만 = 유일 차별점.** ⚠️ 더 날카로운 후보 = roofline 라벨 coarseness 극복.
- ✅ **자동 루프 골격 7파일 — GPU 없이 로컬 self-check PASS.** 코드 = `~/workspace/gpu_solver_test/loop/` (독립 git repo). 순수 로직(★Trace Parser·Hypothesis Engine·Rule Evolver·Ledger) 완결. **차별점 E2E 실증**: 틀린 정적 임계값이 측정 피드백으로 신뢰도 0.5→0.0 강등→폐기→다음 후보 전환. 정적(CUDAMaster류)은 불변 = 격차 증명(성공기준 b 충족). 글루(Generator/Gate/Profiler) = stub, Colab 대기.
- ✅ **Colab 터널 + 새 스택 baseline 확정** ([[01-hard-loop-poc]] §R4) — SSH 터널(cloudflared) + `colabrun` 래퍼로 A100-40GB 실측. **flash 4D 재발견**: 이전 세션 "flash 죽음→naive 챔피언" = 3D 호출 버그 오결론, 재반증. `solve()`=flash4d 전환(2.25×, 커밋 `0271aad`). R3 torch.compile FFN 융합 = 세번째 반증.
- ⏭️ 다음: (a) **glue.py stub→Real** (RealGenerator=LLM API, RealGate=challenge.py reference atol, RealProfiler=nsys/ncu) → 첫 자동 라운드. (b) **R3' 직접 Triton FFN 융합** (torch.compile 폐기).

## 폐기/역사

- [[HANDOFF-SPEC]] — 옛 Hermes 핸드오프 사양. 설계 스펙이 대체. 단 §9.5 저장소 구조 / §9.6 동시편집 규칙은 구현 착수 시 재참조용으로 유효.
