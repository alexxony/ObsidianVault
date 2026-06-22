---
title: Hard 문제 최적화 루프 PoC (수동 1회전)
status: done
created: 2026-06-22
tags: [gpu-solver, poc, loop, hard, optimization, leetgpu, colab, ncu, profiling]
---

# Hard 문제 최적화 루프 PoC — 수동 1회전

> 🗂️ **인덱스: [[GPU-Solver-MOC]]**
> [[2026-06-22-agentic-gpu-optimizer-design]] §1 가설엔진 / §3 루프를 시스템 없이 **손으로 1회전** 실연.
> 측정원 = Colab Pro A100 + ncu ([[00-measurement-feasibility]] 판정 A).

## 결론 한 줄

**컨셉이 실제로 돈다.** Hard 문제(Llama decoder block) 정확성 통과 + 근거 기반 최적화 1라운드로 **2.03× 가속** 달성. 가설 적중.

## 대상 문제

- LeetGPU Hard: Llama-style transformer decoder block (pre-norm, GQA 8q/2kv, RoPE, SwiGLU).
- 칩/프레임워크: NVIDIA A100-80GB, Triton 허용 환경 (torch+triton, numpy 금지).
- seq_len ≤ 4096, 성능 측정 seq=2048. d_model=512, ffn_hidden=1408.
- 시그니처: `solve(x, output, weights, cos, sin, seq_len)`. 가중치 = 단일 packed buffer (2,819,072 floats).

## 정확성 게이트 (PASS)

- 독립 참조 구현(다른 코드 경로: GQA를 `expand`+명시 루프, solve는 `repeat_interleave`+batched matmul)과 `allclose` 교차검증.
- seq ∈ {1, 4, 128, 2048} 전부 PASS (atol/rtol 1e-3).
- **통과 = 입장권.** 누구나 재시도로 됨. 차별점 아님.

## 최적화 루프 (수동 1회전) — 차별점

루프 = `측정 → 룰 가설 → 변형 → 재측정 → 판정`. 시스템 미존재 단계라 사람이 룰 대역.

| 라운드 | 변형 | latency (seq=2048) | 배속 | 정확성 | 가설 적중 |
|---|---|---|---|---|---|
| R0 | naive (score 행렬 (8,2048,2048) materialise) | 3.495 ms | 1.00× | PASS | (기준선) |
| R1 | flash attn (`F.scaled_dot_product_attention`, is_causal) | 1.723 ms | 2.03× | PASS | ✅ |
| R2 | GQA `enable_gqa=True` (kv broadcast, repeat_interleave 삭제) | 3.616 ms | 0.48× | PASS | ❌ **반증** |
| R2' | GQA `expand` view (repeat_interleave 대체, flash 유지) | 1.812 ms | 1.93× | PASS | ❌ **반증** |

**R1 가설(룰 발화)**: "seq 큼 + score 행렬 materialise = attention 메모리 병목 → flash로 융합."
- 변형 범위 = attention 6줄만. RMSNorm/RoPE/GQA/FFN 동일 (최소 타깃).
- 안전장치: 속도 보기 전 정확성 먼저 재확인 (PASS 유지). 가짜 개선 방지.
- 판정: 줄었나(✅) + 정확성 유지(✅) → **룰 신뢰도 +1.**

수동 "더 빨리 해줘"와의 차이 = **가설이 근거 있음** (트레이스/구조 → 병목 → 타깃 변형). = [[2026-06-22-agentic-gpu-optimizer-design]] §3 가설엔진의 실물.

## R2 — ncu 병목 확정 후 2회 변형 (둘 다 반증)

R1까진 병목이 눈에 뻔함(score 행렬). R2부터 **ncu per-kernel 트레이스로 진짜 병목 확정 후 가설.** 추측 변형 금지.

### ncu 병목 확정 (R1 코드, A100, nvtx attn/ffn 구간)

`--csv --page raw`로 커널별 `gpu__time_duration` 추출. 1회 forward 기준:

| op | 커널 | 시간/launch | SM% | DRAM% | 분류 |
|---|---|---|---|---|---|
| FFN `silu(gate)*up` mul | BinaryFunctor mul | **75.7 μs** | 4% | — | **메모리바운드** |
| FFN silu | silu_kernel | 50.5 μs | 33% | — | 메모리바운드 |
| attn RoPE/residual mul·add | elementwise mul (×다수) | 18~37 μs ea | 24~37% | 30~52% | 메모리바운드 |
| RMSNorm reduce | reduce_kernel | 27 μs (×2) | 10% | — | 메모리바운드 |
| GQA repeat_interleave | CatArrayBatchedCopy | 30 μs (×2) | 17% | — | 메모리바운드 |
| QKV/O/FFN matmul | ampere_sgemm | 2~7 μs ea | 68~85% | — | 연산(공짜) |
| flash attn | fmha_cutlassF | **1.74 μs** | 5% | 48% | (R1 성과) |

**핵심: 천장 = matmul 아님. elementwise/메모리바운드 커널이다.** sgemm 전부 ~30μs, sdpa 1.7μs = 연산 거의 공짜. 시간 먹는 놈 = elementwise mul/add/silu (SM 3~33%, DRAM 30~52% = HBM 왕복 병목).

### R2 가설 (룰 발화, 근거=ncu)

> "elementwise 커널 각자 HBM 왕복 = 메모리바운드. 데이터 큼·연산 작음 → **융합으로 왕복 제거.**"
> 1차 타깃 = GQA repeat_interleave(CatArray 30μs×2) 제거 (작은·안전 변형부터).

### R2/R2' 변형 + 반증

- **R2 (`enable_gqa=True`)**: kv 안 늘리고 sdpa 내부 broadcast → repeat_interleave 삭제. 결과 **3.616 ms (0.48×, 반증).** 원인: enable_gqa가 **flash 백엔드 끔** → math 경로 fallback. CatArray 30μs 아끼고 flash 1.7μs→~1900μs 잃음. 순손실.
- **R2' (`expand` view)**: repeat_interleave를 stride-0 expand로 대체, enable_gqa 빼서 flash 복귀. 결과 **1.812 ms (1.93×, 반증).** R1(1.723)보다 +5%. 원인: `expand→reshape`가 stride-0라 reshape이 결국 contiguous 복사 유발 → CatArray 안 사라짐 + 노이즈.

**판정: GQA repeat_interleave 제거 = 이 케이스서 이득 없음. R1(1.723) 챔피언 유지.** 룰 "작은 메모리바운드 커널 제거" 신뢰도 −1.

### R2가 가르친 것 (= 룰 보정)

1. **30μs는 전체 1723μs의 1.7%.** 아껴봐야 측정 노이즈에 묻힘. ncu 교훈: 30μs 쫓지 말고 **126μs(FFN silu+mul)** 쫓아라. → 룰에 "타깃 비중 ≥ 5% 게이트" 추가.
2. **융합 시도가 더 비싼 백엔드를 깨울 수 있다** (enable_gqa→flash off). 변형이 다른 최적화를 끄지 않는지 확인 필수. → 룰에 "백엔드 회귀 체크".
3. **반증도 결과물.** 가설 2개 다 ncu 근거였고 2개 다 측정이 기각 = 루프가 추측 아니라 측정으로 굴러간다는 증거.

### 다음 (R3) — 진짜 타깃

FFN `silu(gate)*up` 융합 (ncu 126μs, 메모리바운드 확정, 비중 ~33%). torch.compile 한 줄 또는 직접 Triton 융합 커널.

## 얻은 교훈 (실제 시스템 가드로 직결)

1. **노트북판 ↔ 파일판 두 소스 손 복사 = 줄 누락 위험.** 이번에 probe 파일 만들 때 `h = _rmsnorm(x, w1)` 줄 누락 → `NameError`. 노트북판은 멀쩡했음.
   → 시스템은 **단일 소스에서 파일 생성**해야 (이중 작성 금지).
2. **파일 쓴 직후 핵심 줄 `assert` 검증 가드 필수.** `assert "h = _rmsnorm(x, w1)" in open(path).read()`. 통과 못 하면 ncu/subprocess 진행 금지. = 깨진 프로파일 차단.
   → design 구현 시 Trace Parser 앞단 가드로 박을 것.
3. ncu가 Python traceback 내면(실행파일 못 찾음 아님) **소스 파일이 범인**, 프로파일 하네스 아님.

## 다음

- [x] R2: ncu per-kernel 프로파일 → 병목 확정(elementwise 메모리바운드) → GQA 제거 2회 변형 → 둘 다 반증, R1 챔피언 유지.
- [ ] R3: FFN `silu(gate)*up` 융합 (torch.compile / Triton) — ncu 126μs 타깃, 메모리바운드 확정.
- [ ] 곡선 누적: R0→R1→R2→R2'→R3 latency 곡선 + 라운드별 가설 로그(반증 2건 포함) = 포폴 결과물(곡선+로그).
- [ ] LeetGPU 제출로 실제 PASS 도장 + percentile 교차검증 (Pro).
