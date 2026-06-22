---
title: GPU-Solver — Map of Content (MOC)
status: active
created: 2026-06-22
tags: [gpu-solver, moc, index, portfolio]
---

# GPU-Solver — Map of Content

> 🧭 이 프로젝트 문서 허브. 모든 노트의 진입점 + 그래프 중심.
> **한 줄 정의**: 트레이스를 해석하는 머리는 내가 짜고, 코드 재작성만 LLM에 시키는 **근거 기반 GPU 커널 최적화 루프** (포트폴리오 / AX 메인).

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
| [[01-hard-loop-poc]] | Hard 문제 최적화 루프 PoC (수동 R0→R2', ncu 병목 확정 + 반증 2건) | 🟢 done |
| [[02-prior-art-survey]] | 사전 탐사 — 선행연구 4종 비교, 차별점 검증 | 🟢 done |
| [[HANDOFF-SPEC]] | 옛 Hermes 사양 | ⚫ deprecated |

## 핵심 결정 추적 (설계 진화)

- **목표 변경**: "런타임 속도 상위권" → **ncu 메트릭 개선 곡선 + percentile 검증점** (Task 0 반영). 근거: [[2026-06-22-agentic-gpu-optimizer-design]] §0, §7.
- **LLM 단순화**: 멀티-LLM 3단 폴백 사다리 → **단일 LLM**. ([[HANDOFF-SPEC]] §2에서 폐기됨)
- **직군 재조준**: 컴파일러/perf 어필 → **AX(에이전트/LLM) 메인 + 저수준 적용력 보조.**
- **측정 아키텍처**: 루프 신호 = Colab Pro A100 + ncu(무제한) / percentile 검증 = LeetGPU Pro(가끔). 근거: [[00-measurement-feasibility]].

## 현재 위치 / 다음

- ✅ Task 0 (측정 가용성) **통과** — 신호원 확보.
- ✅ [[01-hard-loop-poc]] **수동 루프 R0→R2'** — R1 flash 2.03× 적중, R2/R2' GQA 반증 2건(R1 챔피언 유지). ncu로 진짜 병목(elementwise 메모리바운드) 확정. 루프가 측정으로 굴러감 증명.
- ✅ [[02-prior-art-survey]] **차별점 검증** — 선행 4종(Sakana/CUDA-Agent/CudaForge/robust-kbench) 전부 병목 해석을 LLM에 맡김. 우리 "결정론적 룰DB + 진화 메타루프" = 미충족 빈틈. "재조합 함정" 아님.
- ⏭️ 다음: (a) NVIDIA DSL+SOL roofline 분류 추가 조사(인접 선행) → (b) Runner(SSH) + Trace Parser 구현 → (c) solve.py 단일소스 → R3.

## 폐기/역사

- [[HANDOFF-SPEC]] — 옛 Hermes 핸드오프 사양. 설계 스펙이 대체. 단 §9.5 저장소 구조 / §9.6 동시편집 규칙은 구현 착수 시 재참조용으로 유효.
