---
title: Hard 문제 최적화 루프 PoC (수동 1회전)
status: done
created: 2026-06-22
tags: [gpu-solver, poc, loop, hard, optimization, leetgpu, colab]
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

**R1 가설(룰 발화)**: "seq 큼 + score 행렬 materialise = attention 메모리 병목 → flash로 융합."
- 변형 범위 = attention 6줄만. RMSNorm/RoPE/GQA/FFN 동일 (최소 타깃).
- 안전장치: 속도 보기 전 정확성 먼저 재확인 (PASS 유지). 가짜 개선 방지.
- 판정: 줄었나(✅) + 정확성 유지(✅) → **룰 신뢰도 +1.**

수동 "더 빨리 해줘"와의 차이 = **가설이 근거 있음** (트레이스/구조 → 병목 → 타깃 변형). = [[2026-06-22-agentic-gpu-optimizer-design]] §3 가설엔진의 실물.

## R2 (보류) — 추측 금지 게이트

- R1까진 병목이 눈에 뻔함(score 행렬). **R2부터는 ncu 트레이스로 진짜 병목 확인 후 가설.** 추측 변형 = 수동 루프 회귀.
- ncu 프로파일 시도 중. 첫 시도는 probe 파일 버그로 중단(아래 교훈). 파일 가드 통과 후 재프로파일 대기.
- 다음 병목 후보(미확정, ncu가 정함): FFN matmul(가장 큰 연산) / 작은 elementwise 커널 융합(RMSNorm·RoPE·residual) / fp16·bf16 tensor core.

## 얻은 교훈 (실제 시스템 가드로 직결)

1. **노트북판 ↔ 파일판 두 소스 손 복사 = 줄 누락 위험.** 이번에 probe 파일 만들 때 `h = _rmsnorm(x, w1)` 줄 누락 → `NameError`. 노트북판은 멀쩡했음.
   → 시스템은 **단일 소스에서 파일 생성**해야 (이중 작성 금지).
2. **파일 쓴 직후 핵심 줄 `assert` 검증 가드 필수.** `assert "h = _rmsnorm(x, w1)" in open(path).read()`. 통과 못 하면 ncu/subprocess 진행 금지. = 깨진 프로파일 차단.
   → design 구현 시 Trace Parser 앞단 가드로 박을 것.
3. ncu가 Python traceback 내면(실행파일 못 찾음 아님) **소스 파일이 범인**, 프로파일 하네스 아님.

## 다음

- [ ] R2: ncu 재프로파일 (파일 가드 통과 상태) → 커널별 시간표 → 진짜 병목 확정 → 변형 → R2 측정.
- [ ] 곡선 누적: R0→R1→R2 latency 곡선 + 라운드별 가설 로그 (포폴 결과물 = 곡선+로그).
- [ ] LeetGPU 제출로 실제 PASS 도장 + percentile 교차검증 (Pro).
