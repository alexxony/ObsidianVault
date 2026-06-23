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

**차별점 절반 죽고 절반 살았다.** (2026-06-23 roofline 2차 조사로 1차 결론 일부 수정.)
- ❌ **"결정론 룰 라벨 → LLM 재작성만 → 측정검증" 파이프라인 = 신규 아님.** [[#roofline 2차 조사]]에서 **CUDAMaster**(arXiv 2603.07169)가 정확히 이걸 함. KernelAgent도 하이브리드로 함. 1차 결론("셋 다 LLM에 맡김")은 **CUDAMaster를 못 찾아 과대평가**였음.
- ✅ **"룰DB가 결과 피드백으로 진화하는 메타루프" = 조사한 6개 어디에도 없음.** 전부 정적/일회캘리브 임계값. **이 하나가 유일 차별점.**
- ⚠️ 차별점이 진화 메타루프 하나에 전부 걸림 → 작동 못 하면 CUDAMaster 열등복제. 리스크 큼.

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

## roofline 2차 조사 (2026-06-23)

> open question #1·#2 해소. deep-research(6각·16소스·80 claim·20확정·5기각). **결정론 roofline 분류 = 우리 룰DB와 같은가?** 답: 대체로 같음 + CUDAMaster가 LLM 재작성 연결까지 함.

### ★ CUDAMaster (arXiv 2603.07169) — 진짜 가장 가까운 선행 (1차 조사 누락)

우리 파이프라인과 **구조적으로 거의 동일.** 정직하게 인정.

- ncu 메트릭 **결정론적 임계값 룰표**(Table/Appendix A.2): Compute Bound = `Compute.Th>30%`; Memory Latency Bound = `Compute·DRAM·Memory.Th 셋 다 <30%`; Memory Bandwidth Bound = `Compute.Th<30% & (DRAM.Th>30% or Memory.Th>30%)`.
- 임계값 30% = **Otsu's method**(클래스간 분산 최대화)로 베이스라인 분포에서 **일회 오프라인 캘리브 후 고정.**
- 그 라벨 → **멀티에이전트 LLM 재작성**(PlannerAgent→CoderAgent) → ExecuteAndFilter 재프로파일 + CheckValid 매 라운드 검증.
- = 우리 "결정론 라벨 → LLM 재작성만 → 측정검증" **그대로.** 룰 진화만 빠짐.

**정정:** 내 2차 조사 전제가 틀렸음 — Otsu's method는 DSL+SOL(2603.29010)이 아니라 **CUDAMaster(2603.07169)**가 씀. DSL+SOL은 고전 roofline ridge-point 컷(Otsu 아님) + SOL 리포트를 **LLM이** 생성.

### 결정론 roofline 분류는 이미 흔하다 (= 부분 재발명)

| 출처 | 분류 방식 | 코드 변형 연결 |
|---|---|---|
| **CUDAMaster** | Otsu 일회캘리브 30% 임계값표 (측정 ncu 트레이스) | ✅ LLM 재작성 |
| **KernelAgent** (PyTorch) | roofline SOL(max(compute,memory)) **+ LLM 근거추론** 하이브리드. 3-level RAG 최적화DB(병목별) | ✅ LLM 재작성. 단 RAG 정적 코사인 |
| **NVlabs/SOLAR** | `compute_cycles >= mem_cycles` (분석적, 측정 아님) | ❌ 분석 리포트 |
| **Nsight Compute 내장** | ridge-point가 차트를 Memory/Compute Bound 영역 분할, achieved AI가 떨어진 영역 = 라벨. **사용자 확장 .section + Python 룰파일**(NvRules API) | ❌ 리포트 |
| **DSL+SOL** (2603.29010) | `tSOL=max(Tcompute,Tmem)`, AI vs ridge-point. SOL 리포트 **LLM 생성** | 에이전트 steering |

→ **Nsight Compute 자체가 사용자작성 룰파일로 병목 finding 방출** = 우리 룰DB와 기능 동일. **우리 기본 분류는 부분 재발명.** 차용해야지 재구현 X.

### 3개 질문 답 (갱신)

1. **NVIDIA roofline = 우리 룰DB와 같나?** 같음(결정론 분류). "정적 룰→LLM 재작성"도 CUDAMaster/KernelAgent가 이미 함. **진화 메타루프만 새것.**
2. **재발명인가?** 기본 분류(compute/memory bound)는 **응, 부분 재발명.** Nsight 내장 + CUDAMaster 차용.
3. **차용 가능?** ✅ NVIDIA roofline 공식(`tSOL=max(Tcompute,Tmem)`, ridge=MAC_per_cycle/DRAM_BW, AI=MACs/bytes) + CUDAMaster 30% 임계값표 = **우리 시드 룰 근거로 직접 차용.** SOLAR/Nsight 공식 단순·공개.

### 진짜 차별점 (살아남은 것) + 새 방어선 후보

- ✅ **룰DB 진화 메타루프.** 6개 전부 정적/일회캘리브(CUDAMaster Otsu 1회·고정, SOLAR/Nsight/DSL+SOL 정적 공식, KernelAgent RAG 정적 코사인). **결과 피드백으로 룰이 갱신 = 우리만.**
- ⚠️ **더 날카로운 방어선 후보** (open#4): 진화 자체보다 **roofline 라벨의 coarseness 극복**일 수 있음. Nsight 기하학 라벨은 latency/occupancy/L2-thrash를 'memory bound'로 오판(VI-HPS/NERSC 경고). 우리가 측정 트레이스로 **진짜 근본원인까지** 세분하면 = 진화보다 강한 차별. 단 미검증.

## 미해결 (open questions) — 추가 조사/검증 필요

1. **진화 룰이 정적 룰보다 정말 나은 라벨/결과를 내나?** = 차별점이자 미검증 가설. 어떤 실험이 "진화 룰 > CUDAMaster 고정 임계값"을 증명? → **우리 PoC의 핵심 입증 과제.**
2. **CUDAMaster도 측정 ncu 트레이스를 라벨함** → "우리는 측정 트레이스 라벨" 구분은 SOLAR/Nsight-분석경로 대비만 차별, CUDAMaster 대비는 아님. 측정 vs 분석은 차별점 못 됨.
3. **결정론 룰엔진 vs LLM 생성 리포트** — DSL+SOL은 SOL 리포트를 LLM 생성(NVIDIA가 실제 출시한 경로), CUDAMaster는 결정론. 우리는 결정론 채택 → 환각/오류 줄이는 게 실측 이득인지 입증 필요.
4. **★ coarseness 극복이 진짜 방어선?** 기하/임계값 roofline 라벨이 latency/occupancy/L2 stall을 오판하는 한계를, 진화 룰DB가 근본원인까지 세분 가능한가. **진화 per se보다 이게 방어 가능한 진짜 novelty일 수 있음** — 다음 설계 핵심.
5. nsys 아무도 안 씀(ncu/torch/clang만) — 자동 룰 해석 부적합 가능성 feasibility.
6. robust-kbench·ISO-Bench·SOLAR/Nsight 공식 plug-in 라이선스/통합비용 미확인.

## 방법 메모

- 기각된 3 claim = "완전 LLM-driven"이라는 **과잉 일반화** 주장이 1-2 표로 죽음(컷). 핵심 결론(룰DB 부재)은 22 확정 claim으로 살아있음.
- '룰DB 부재'는 부분적으로 argument-from-absence(논문이 "우리는 룰DB 안 씀" 명시 X)이나, 각 시스템 전체 스캐폴딩 열거에 결정론적 트레이스 해석 레이어 없음 = 합리적 추론.

## 1차 출처

- Sakana AI CUDA Engineer: https://sakana.ai/ai-cuda-engineer/ · https://arxiv.org/abs/2509.14279 · https://github.com/SakanaAI/robust-kbench
- CUDA-Agent: https://arxiv.org/abs/2602.24286 · https://github.com/BytedTsinghua-SIA/CUDA-Agent
- CudaForge: https://arxiv.org/html/2511.01884v1 · https://github.com/OptimAI-Lab/CudaForge
- ISO-Bench: https://github.com/Lossfunk/ISO-Bench
- **★ CUDAMaster** (가장 가까운 선행): https://arxiv.org/html/2603.07169v1
- NVIDIA SOL-ExecBench: https://github.com/NVIDIA/SOL-ExecBench · SOLAR: https://github.com/NVlabs/SOLAR · DSL+SOL: https://arxiv.org/html/2603.29010v1
- Nsight Compute Sections & Rules: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- KernelAgent: https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/ · repo: https://github.com/meta-pytorch/KernelAgent
