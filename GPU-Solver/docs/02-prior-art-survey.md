---
title: 사전 탐사 — agentic CUDA 커널 최적화 선행연구
status: done
created: 2026-06-23
tags: [gpu-solver, prior-art, survey, research, agentic, cuda, kernel-optimization]
---

# 사전 탐사 — agentic CUDA 커널 최적화 선행연구

> 🗂️ **인덱스: [[GPU-Solver-MOC]]**
> [[2026-06-22-agentic-gpu-optimizer-design]]의 차별점이 살아있는지 검증. deep-research 하니스(5각 검색·22소스·105 claim 추출·25 검증·22 확정·3 기각).

## 결론 한 줄

**우리 차별점 = 살아있다. "그냥 재조합" 함정 아님.** 핵심("트레이스 해석 룰은 사람이 작성, LLM은 코드 재작성만")은 조사한 4개 시스템 어디에도 완전 구현 안 됨 — 셋 다 병목 해석을 LLM에 맡김. 단 결정론적 roofline 분류(NVIDIA DSL+SOL) = 미조사 인접 선행, 추가 조사 필요.

## 조사 대상 4종 비교

| 시스템 | 루프 구조 | 병목 해석 주체 | 사람 룰 레이어 | 프로파일러 |
|---|---|---|---|---|
| **Sakana AI CUDA Engineer** | torch→CUDA 번역 → 진화적 메타생성 | **LLM**이 프로파일러 출력 자연어 요약 | ❌ 없음 | torch/NCU/Clang-tidy, LLM이 요약 |
| **CUDA-Agent** (ByteDance/Tsinghua) | impl→compile→verify→profile (대규모 RL) | **단일 LLM**이 멀티턴 자가진단 | △ SKILL.md(정적 프롬프트), 파싱·진화 X | RL 리워드 신호로 배선 |
| **CudaForge** (가장 유사) | Coder+Judge 2에이전트 (training-free) | **Judge LLM**이 NCU raw 읽고 판정 | ❌ 결정론 매핑 없음 (24-metric 선택만 사람) | NCU, Judge가 reasoning으로 해석 |
| **robust-kbench** | (최적화 X) 벤치마크·검증 스위트 | — | — | NCU 선택적, 미소비 |

### 핵심 대조: CudaForge (우리와 정반대)

가장 가까운 선행이지만 **정반대 설계.** Judge 프롬프트(verbatim): *"You are a senior CUDA performance engineer ... identify exactly one highest-impact speed bottleneck by 3-4 most important metrics ... Be surgical and metrics-driven."* 병목 클래스를 *"occupancy, memory coalescing, smem bank conflicts, register pressure, ... etc."* — **'etc.'로 열린 집합, LLM reasoning이 선택.** 우리는 **결정론적 룰DB가 라벨링.** → 완벽한 대조군. "왜 룰 분리가 LLM-판정보다 나은가"가 우리 논점.

## 4개 질문 답

### (1) 이미 풀린 것 → 재발명 금지, 차용

- **리워드 해킹 방지 측정** — robust-kbench task-filtering 6체크(output range / std-dev / axes variation / init impact / input impact / LLM-judge inefficiency). KernelBench류가 가짜 50~120× speedup 내는 문제(오염 task 제외 시 평균 3.13×→1.49× 붕괴)를 막음. → 우리 **Correctness Gate에 차용.**
- **추론 워크로드 wall-clock 평가** — ISO-Bench(vLLM/SGLang 실제 워크로드). 평가 하니스 처음부터 만들 필요 없음.

### (2) 차별 빈틈 → 우리 존재 이유

- **결정론적 트레이스→병목 라벨 룰DB.** 4개 어디에도 없음. 셋 다 LLM이 해석.
- **룰이 측정 피드백으로 진화하는 메타루프(Rule Evolver).** CUDA-Agent SKILL.md조차 정적 — 진화 안 함. Sakana Innovation Archive(RAG)·CUDA-Agent RL은 '학습'이지 '사람 작성 룰의 진화' 아님.

### (3) 차용 컴포넌트

| 출처 | 차용 대상 | 우리 어디에 |
|---|---|---|
| robust-kbench task-filtering | 리워드해킹 방지 6체크 | Correctness Gate 강화 |
| ISO-Bench | 추론 워크로드 평가 | 평가 지표(추론 타깃 시) |
| CUDA-Agent SKILL.md | symptom→cause→solution 진단 테이블 | 시드 룰DB **초안 참고** (그대로 X — 우리는 파싱+진화 추가) |

### (4) 함정인가? → 아니

- 포폴: "남들은 LLM이 병목 판정=블랙박스. **나는 트레이스 해석을 결정론 룰로 외부화 + 그 룰을 측정 피드백으로 진화.** 감사 가능·내 코드." (= design §3.2-3.3 분리 논점)
- 성능 SOTA 경쟁 아님. **해석가능·감사가능 최적화**가 기여 (성능 랭킹 과약속 금지).

## 정직한 리스크 2개

1. **"룰이 LLM 판정보다 정말 나은가?"** 증명 부담 우리 것. CudaForge·CUDA-Agent는 SOTA speedup 냄. 우리 룰DB가 그만큼 좋은 라벨 못 내면 차별점→약점. 비교 실험 출처 없음 = **우리 PoC로 입증할 가설.**
2. **우리 PoC(R1/R2)는 룰 5개도 아직 안 만듦.** 빈틈은 확인됐으나 메울 능력은 미증명. [[01-hard-loop-poc]] 다음 단계가 그 증명.

## 미해결 (open questions) — 추가 조사 필요

1. **★ NVIDIA DSL+SOL (arXiv 2603.29010)** — roofline 기반 **결정론적** 병목 분류(Otsu's method로 Compute Bound / Memory-Latency Bound / Memory-BW Bound). **우리 룰DB와 가장 가까운 선행일 수 있음** — 차별점 핵심이 여기서 갈림. SOL-ExecBench도 같이 미조사(이번 보고서 budget drop).
2. **KernelAgent** (PyTorch blog, "hardware-guided multi-agent orchestration) — 또 하나의 인접 사례, 미조사.
3. **룰 '진화' 선행 정말 없나?** Sakana RAG·CUDA-Agent RL과 '사람 룰 진화'의 구분이 충분히 방어 가능한지 추가 문헌.
4. **nsys 아무도 안 씀** (ncu/torch/clang만). nsys 기반 자동 룰 해석 = 완전 미개척이나, "왜 아무도 안 쓰는지"(부적합 가능성) feasibility 검증 필요.
5. robust-kbench·ISO-Bench를 우리 메타루프에 실제 plug-in 가능? 라이선스/API/통합비용 미확인.

## 방법 메모

- 기각된 3 claim = "완전 LLM-driven"이라는 **과잉 일반화** 주장이 1-2 표로 죽음(컷). 핵심 결론(룰DB 부재)은 22 확정 claim으로 살아있음.
- '룰DB 부재'는 부분적으로 argument-from-absence(논문이 "우리는 룰DB 안 씀" 명시 X)이나, 각 시스템 전체 스캐폴딩 열거에 결정론적 트레이스 해석 레이어 없음 = 합리적 추론.

## 1차 출처

- Sakana AI CUDA Engineer: https://sakana.ai/ai-cuda-engineer/ · https://arxiv.org/abs/2509.14279 · https://github.com/SakanaAI/robust-kbench
- CUDA-Agent: https://arxiv.org/abs/2602.24286 · https://github.com/BytedTsinghua-SIA/CUDA-Agent
- CudaForge: https://arxiv.org/html/2511.01884v1 · https://github.com/OptimAI-Lab/CudaForge
- ISO-Bench: https://github.com/Lossfunk/ISO-Bench
- NVIDIA SOL-ExecBench: https://github.com/nvidia/sol-execbench · DSL+SOL: https://arxiv.org/html/2603.29010v1
- KernelAgent: https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/
