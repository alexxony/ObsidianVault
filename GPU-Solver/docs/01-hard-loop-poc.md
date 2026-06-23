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

## R4 — 새 스택 재실행 + flash 4D 재발견 (2026-06-24)

위 표는 옛 스택(A100-80GB, 옛 torch) 기록. **새 스택(A100-SXM4-40GB, torch 2.11.0+cu128)** 으로 자동 루프 baseline 옮기며 재측정. 코드 = [[gpu_solver_test]] `solve.py` (단일소스).

### 정적 가정 반증 → 재반증 (루프가 잡을 발견의 실물)

새 스택 초기 세션은 SDPA를 **3D `(H,T,D)`** 로 호출 → `"No available kernel. Aborting."` → math fallback. 이걸 보고 **"이 스택서 flash 죽음, naive가 챔피언"** 으로 오결론, solve.py 헤더에 박음.

이번 세션 측정으로 **그 오결론을 재반증.** 진범 = **호출 차원.** flash 커널은 **4D `(B,H,T,D)`** 필수. isolated probe:

```
flash avail: True | A100 (8,0)
4D (1,8,T,64) flash: OK
3D (8,T,64)   flash: FAIL "No available kernel. Aborting."
```

`q.transpose(0,1).unsqueeze(0)` 로 batch 차원 추가 → flash 정상 작동.

| 변형 | latency (seq=2048, 40GB) | 배속 | 정확성 |
|---|---|---|---|
| naive (score materialise) | 3.244 ms | 1.00× | PASS |
| **flash4d (4D SDPA)** | **1.433 ms** | **2.25×** | PASS |
| R3: flash4d + torch.compile FFN 융합 | 1.806 ms | 0.79× | PASS ❌ **반증** |
| R3': flash4d + 직접 Triton silu*up 융합 | 1.414 ms | 1.01× | PASS ⚠️ **약한 적중** |

**판정:**
1. **flash 챔피언 복귀.** 옛 스택 R1(2.03×) 재현 + 약간 상회(2.25×, 40GB). `solve()` 본체 = flash4d로 전환 (커밋 `0271aad`).
2. **R3 torch.compile FFN 융합 = 세번째 반증.** flash4d 대비 회귀(1.433→1.806). 원인 = 작은 행렬(T×512)에서 max-autotune이 고른 triton mm이 eager cuBLAS sgemm보다 느림 + 컴파일 호출 오버헤드. → R2 교훈("융합 시도가 더 비싼 백엔드 깨움")과 동형.
3. **R3' 직접 Triton 융합 = 약한 적중 (커밋 `145e4a7`).** silu*up elementwise만 1커널(`_silu_mul_kernel`), matmul=cuBLAS 유지. ncu 커널: eager silu 15μs+mul 21μs=36μs(2커널) → fused 21.8μs(1커널) = **-39% 커널 절감**. 그러나 전체는 1.01×(노이즈 가까움). **이유 = FFN elementwise 비중 36/1433 ≈ 2.5% < 5% 게이트.** 융합은 옳으나(회귀 아님, torch.compile R3과 반대) 타깃이 작아 천장 효과 미미.

### R5 — ncu 전체 천장 식별 → TF32 (이번 세션 최대 이득)

R3'까지 FFN만 봄. flash4d **전체 ncu 분포** 떠서 진짜 천장 식별 (`ncu --metrics gpu__time_duration.sum --csv`, 비중 %):

| 커널 | 비중 | ≥5% |
|---|---|---|
| matmul (sgemm) | **52.7%** | ★ |
| flash_attn | **28.7%** | ★ |
| elementwise (rope/residual) | **7.7%** | ★ |
| mul | **6.5%** | ★ |
| rmsnorm/reduce | 2.8% | |
| silu | 0.9% | |
| repeat_interleave | 0.8% | |

**충격 발견: matmul = 52.7% 압도적 1위.** 옛 PoC 교훈("matmul 공짜, elementwise가 천장")이 이 스택선 **뒤집힘.** 원인 = fp32 sgemm이 텐서코어 미사용.

**R5 가설(룰 발화)**: "matmul 52.7% = fp32 sgemm 텐서코어 미사용 → TF32로 텐서코어 태움." 변형 = `torch.set_float32_matmul_precision("high")` 전역 1줄 (코드 구조 무변경).

| 변형 | latency | 배속(vs flash4d) | 정확성 |
|---|---|---|---|
| flash4d fp32 | 1.432 ms | 1.00× | PASS (maxdiff 2e-7) |
| **flash4d + TF32 (R5)** | **0.840 ms** | **1.71×** | PASS (maxdiff 3.7e-4 < 1e-3) |

**판정: R5 대박 적중. 이번 세션 최대 이득.** 누적 **naive 3.244 → flash4d 1.432 → +TF32 0.840 = 3.86×** (커밋 `9e09b9b`). ncu 천장 식별→타깃 변형→재측정 루프가 큰 이득(71%) 직격. FFN 융합(1.4%)과 차원 다름 = "≥5% 게이트" 룰의 가치 입증. 부수: TF32 후 fused_ffn(0.98)>flash4d(0.96) — matmul 빨라지며 비중 재편, 챔피언=flash4d 단독.

### R4가 가르친 것 (= 시스템 가드 직결)

4. **"커널 없음"은 빌드 문제 아니라 호출 형태 문제일 수 있다.** SDPA flash는 dtype(fp16/bf16/fp32 다 시도해도 3D면 실패)이 아니라 **4D 텐서**를 본다. → 시스템 Trace Parser 앞단: "백엔드 unavailable" 신호 받으면 dtype·차원 둘 다 검증 후 결론. 단일 시도로 "기능 죽음" 단정 금지.
5. **자동 루프 baseline 정합성이 전부.** 잘못된 baseline(naive=챔피언) 위에서 최적화 라운드 돌면 전체 trajectory 오염. baseline 확정 = 측정 교차검증(`_reference`) 통과 + 챔피언 재측정 후 진행.
6. **"≥5% 비중 게이트"가 측정으로 자기검증됨 (= 룰DB 진화 실물).** R3'는 커널을 -39% 줄였는데도 전체 1%만 먹었다. R2가 세운 룰("타깃 비중 ≥5%만 쫓아라")이 다른 라운드에서 또 맞음 → 룰 신뢰도 +1. **이게 차별점(rule DB evolution)의 구체 증거**: 정적 시스템(CUDAMaster류)은 "메모리바운드 커널 = 융합" 규칙 고정 → R3' 같은 헛수고 반복. 우리 루프는 측정 피드백으로 "작은 비중은 건너뛰라"를 학습.

## 얻은 교훈 (실제 시스템 가드로 직결)

1. **노트북판 ↔ 파일판 두 소스 손 복사 = 줄 누락 위험.** 이번에 probe 파일 만들 때 `h = _rmsnorm(x, w1)` 줄 누락 → `NameError`. 노트북판은 멀쩡했음.
   → 시스템은 **단일 소스에서 파일 생성**해야 (이중 작성 금지).
2. **파일 쓴 직후 핵심 줄 `assert` 검증 가드 필수.** `assert "h = _rmsnorm(x, w1)" in open(path).read()`. 통과 못 하면 ncu/subprocess 진행 금지. = 깨진 프로파일 차단.
   → design 구현 시 Trace Parser 앞단 가드로 박을 것.
3. ncu가 Python traceback 내면(실행파일 못 찾음 아님) **소스 파일이 범인**, 프로파일 하네스 아님.

## 다음

- [x] R2: ncu per-kernel 프로파일 → 병목 확정(elementwise 메모리바운드) → GQA 제거 2회 변형 → 둘 다 반증, R1 챔피언 유지.
- [x] R4(새 스택): flash 4D 재발견 → naive=챔피언 오결론 재반증 → `solve()`=flash4d 전환 (2.25×, 커밋 `0271aad`). §R4 참조.
- [x] R3: FFN `silu*up` torch.compile 융합 = 반증(flash4d 1.445→1.806, 회귀). 작은 행렬서 triton mm < cuBLAS sgemm.
- [x] R3': FFN `silu*up` **직접 Triton 융합** (`_silu_mul_kernel`, matmul=cuBLAS). 커널 -39%지만 비중 2.5%<5% → 전체 1.01×(약한 적중). 커밋 `145e4a7`. 룰 "≥5% 게이트" 자기검증.
- [ ] 곡선 누적: naive→flash4d→R3(반증)→R3'(약적중) latency 곡선 + 라운드별 가설 로그 = 포폴 결과물.
- [x] 진짜 천장: flash4d 전체 ncu 분포 → matmul 52.7%/flash 28.7%/elementwise 7.7%/mul 6.5% (≥5% 4개). §R5.
- [x] R5: matmul 천장 → TF32 → **0.840ms, 누적 3.86×** (이번 세션 최대 이득, 커밋 `9e09b9b`).
- [ ] R6 후보: 남은 천장 = flash_attn 28.7%(이미 챔피언) / elementwise 7.7%(rope·residual 융합) / matmul 잔여(bf16면 더↓). TF32 후 재프로파일 필요(비중 재편됨).
- [ ] LeetGPU 제출로 실제 PASS 도장 + percentile 교차검증 (Pro).

## 자동화 결정 (수동 핸드오프 통증 → 시스템 1순위)

복붙 핸드오프(나↔Colab↔결과)가 루프 못 닫는 병목. [[2026-06-22-agentic-gpu-optimizer-design]] Runner/Trace Parser가 먹을 대상. 다음 세션 구현 방향 3건 확정:

1. **단일소스 = `.py` (ipynb 폐기).** probe 따로 안 만듦(이중작성 금지, 교훈1) → `solve.py --check|--bench|--profile` argparse. git diff 깨끗, nsys/ncu가 .py 직접 먹음, LLM 파일 통째 수정 쉬움. ipynb 장점(인터랙티브)은 무인 루프엔 불필요.
2. **Runner = VSCode↔Colab SSH 터널.** colab-ssh/cloudflared로 A100 셸 접근 → scp로 solve.py 보내고 `python solve.py` 실행, 출력 회수. 복붙 0 = Runner 실물.
3. **Trace Parser = 프로파일 파일저장 + 요약만 회수 (토큰 99%↓).** 오늘 ncu CSV 247열×165행이 통째 컨텍스트로 들어와 토큰 수만 소모. 대신 `ncu --csv -o p.csv` → `parse.py`가 nvtx 구간 합산 → top-5 커널 JSON(~50토큰)만 LLM에.

### 측정원 명세 (nsys 디폴트 / ncu 옵션)

- **측정원 = nsys (Nsight Systems) 디폴트.** 매 라운드 항상. 1회 실행=1회 측정(싸고 빠름), 전체 타임라인+커널 시간합+런치 갭, 권한 덜 까다로움.
  - `nsys profile -o r.nsys-rep python solve.py --bench`
  - `nsys stats --report cuda_gpu_kern_sum --format csv r.nsys-rep` → 커널별 시간합 → parse.py → top5.
- **ncu (Nsight Compute) = 옵션, 커널 깊이파기만.** nsys가 천장 커널 짚은 뒤 "왜 느린가"(SM% vs DRAM% = 메모리 vs 연산 바운드, occupancy, 레지스터) 알아야 변형 방향 정할 때만. replay라 느림(165커널=6패스), HW 카운터 권한 필요(ERR_NVGPUCTRPERM 위험).
  - `ncu --set basic --csv -o k.ncu-rep -- python solve.py --profile` (그 커널만 타깃).
  - 둘 다 Colab A100서 동작 확인(ncu 됨=권한 OK). 1차 구현 = nsys만, ncu는 부족할 때 추가.

### 실행 위치 (GPU는 Colab 하나면 끝)

| 노드 | GPU | 역할 |
|---|---|---|
| 사용자 PC (Windows) | 0 | VSCode 타자 + SSH. nsys/ncu **GUI는 선택**(eye check용), 루프엔 불필요 |
| LLM (WSL) | 0 | solve.py 생성/수정 + SSH 명령 발행 |
| Colab A100 | **1** | `python solve.py` + `nsys`/`ncu` **CLI** 실행. 유일한 GPU·측정 노드 |

- nsys/ncu는 **CLI 바이너리** — GUI 앱과 별개, 창 안 띄워도 돎. Colab 셸에서 실행.
- 사용자 PC GPU 불필요(단순 해석=타자+전송). GUI 앱 켜둘 필요 없음(.nsys-rep 눈으로 볼 때만).
