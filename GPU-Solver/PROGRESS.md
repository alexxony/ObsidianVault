---
title: GPU-Solver 진행 현황 (재개 진입점)
created: 2026-06-26
updated: 2026-06-30
tags: [gpu-solver, progress, resume]
---

# GPU-Solver — 진행 현황 (재개용 단일 진입점)

> 🧭 세션 재개 시 이 문서 먼저 읽기. 최신 상태 + 다음 할 일 + 재개 절차.
> 상세는 [[04-multiproblem-round-design]], 허브는 [[GPU-Solver-MOC]].

## 한 줄 요약

GPU 커널 최적화 루프. 차별점 = **룰표가 측정 피드백으로 진화** (CUDAMaster 등 선행은 정적).
현재: **메커니즘 전부 입증(룰진화·오발화retire) + 성능 gain layer 첫 돌파.** matmul서 루프가
fp32(9.6ms)→fp32_no_tensorcore 발화→TF32(1.5ms) **6.4× 실측** = "측정→가설→재작성→더 빠른 커널" 실증.
(이전 "환경 한계 3문제 null"은 측정 버그 `_profile_ncu --launch-count 1`였음, 고침.)
차별점 = **세 축 각각 입증: gain 축=matmul 6.4×(A100), 진화 축=sigmoid+conv fp32 오발화 retire(A100, 다문제), 칩 축=T4서 TF32 헛가설 가드 차단(2026-06-30). + T4 한 환경서 칩가드+진화retire 이중방어 동시 관찰(patience 6 픽스, 2026-06-30 후속). 진화 축 다문제 입증(sigmoid+conv, 2026-07-01). gain 다문제·두 축 동시는 future work.**

## ✅ 입증된 것 (전부 push·검증)

| 단계 | 상태 | 증거 |
|---|---|---|
| 인프라 (git-우편함) | ✅ | e2e PASS, 로컬↔Colab(A100) 자동 왕복 |
| 측정 (ncu `--launch-count 1`) | ✅ | 6배↓, 룰 신호 8개 유지 |
| 3문제 신호 프로필 | ✅ | llama(컴퓨트)·sigmoid(메모리)·groupnorm(reduction) 서로 다름 |
| A 발화 관찰 | ✅ | 신호가 다른 룰 깨움 + **fp32 오발화 발견** |
| B 옵션1 (진화 메커니즘) | ✅ | ON=retire→전환, OFF=영원 오발화 (fake 신호) |
| B 대행 GPU 닫힘 | ✅ | **진짜 A100** — TF32 헛방 실측 → fp32 룰 demote |
| **gain 신호 (진화 ON/OFF 비교)** | ✅ | **진짜 A100 14R** — sigmoid서 ON=fp32 retire→memory_bound_fusable 전환, OFF=fp32 ×6 영원 (커밋 `run_gain_compare.py`) |
| **칩 가드 (Context 1급)** | ✅ | **진짜 T4 실측** — matmul fp32서 A100=`fp32_no_tensorcore` 발화(TF32 6.4×) vs T4=가드 차단→`reg_pressure`. 칩만 바꿔 헛가설 사전차단 ([[07-chip-lang-context-design]]) |
| **T4 진화 retire (이중방어 동시)** | ✅ | **진짜 T4 실측 8R** — patience 6 픽스 후 reg_pressure 헛가설 8R 발화→ON=retire 1회, OFF=retire 0. 한 환경서 칩가드(사전차단)+진화retire(사후폐기) 동시 관찰 (2026-06-30 후속, raw_script 인프라) |
| **conv 진화 retire (진화 축 다문제)** | ✅ | **진짜 A100 실측 8R (2026-07-01)** — conv2d(LeetGPU 원본)서 `fp32_no_tensorcore` 헛발화(conv엔 TF32 무효, probe 실측) → ON=fp32 4R→retire 2회→`uncoalesced` 전환, OFF=fp32 7R 영원(retire 0). **= sigmoid retire의 conv 재현 = 진화 축 2문제.** gain은 null(합착 variant 0.79×, compute_tput 0.8 천장). 커밋 `problems/2d_convolution/` |

## ⬜ 미입증 (= 차별점 급소)

- **성능 gain** — "루프가 측정→가설→재작성→더 빠른 커널". ✅ **matmul서 첫 입증 (2026-06-29):**
  | 문제 | 시도 | 결과 |
  |---|---|---|
  | sigmoid | `--latency` 12R, BLOCK 변형 | null — 메모리바운드 ~770ms ±1% (진짜 천장) |
  | groupnorm | 단일블록·welford 1패스·분할병렬 3알고리즘 | null — 전부 ~36ms ±1%, DRAM BW 천장 |
  | llama | TF32 OFF vs ON | null — attention(flash) 54% 지배, matmul 23% 비병목 (TF32 무효 당연) |
  | **matmul (4096² fp32)** | **루프 R0(fp32)→fp32_no_tensorcore 발화→R1(TF32)** | ✅ **9.6ms→1.5ms = 6.4× gain 실측!** |
  - ✅ **gain layer 첫 돌파.** matmul: seed(fp32 9.6ms) → `fp32_no_tensorcore` 발화 → TF32 코드 선택 → 1.5ms.
    = **"측정→가설→재작성→더 빠른 커널" 첫 실증.** R1 후 `tensorcore_saturated`→stop = "충분히 최적화" 올바른 판정.
  - ⚠️ **단 진화 ON>OFF 차이는 없음**(ON best 1494 ≈ OFF 1493). 이유 = matmul 첫 발화 룰이 **이미 맞는 룰** → retire 불요 → 둘 다 R1서 TF32. **= gain 축은 입증, 진화 우월성 축은 sigmoid 오발화 retire가 별개 증거.**
  - ⚠️ **이전 "환경 한계 확정"은 측정 버그(`_profile_ncu --launch-count 1`, 커널 1개만 측정)였음.** 고침(전체 duration 합산, 커밋 `62235bb`). 환경은 TF32 7.2× 정상.
  - **원인 = 측정 환경 ≠ PoC.** llama 24ms = PoC 1.4ms의 **17배.** TF32는 실제 적용됨
    (err 2e-7→2.9e-4 변동 = 저정밀 확인)나 latency 무효 → **현 환경선 matmul 비병목.**
  - ✅ **(2026-06-29) ncu 비중 실측으로 확정** (`executor.ncu_breakdown` 신규, 전체 forward 23커널 그룹 집계):
    **attention(fmha_cutlassF_f32) 54% > matmul(cutlass gemm) 23% > elementwise 23%.** flash attention 단일 커널이
    전체의 ~48%(547us/1132us) 지배. = **"SDPA flash 지배" 추정 → 데이터 확정.** matmul 비병목이라 TF32 OFF/ON 무효가 당연.
    추가: fmha가 **fp32 폴백**(MemEffAttention) = attention 자체가 느린 주범. 진짜 여지 = matmul TF32 아니라 attention bf16 flash,
    단 challenge fp32 정확도 요구 시 불가 → **이 환경/문제선 최적화 여지 없음 확정.** PoC R5(1.71×)는 다른 환경.
  - ⚠️ **(2026-06-29) "환경 한계" 결론 = 측정 버그로 판명, 반전.** 처음엔 matmul TF32 OFF/ON `_profile_ncu` latency
    95040 vs 95232us 동률 → "환경이 TF32 못 냄" 결론냈으나 **틀렸음.** ncu_breakdown(전체 커널 duration)으로 재측정:
    **OFF=`ampere_sgemm_128x64`(FP32 코어) 9.38ms vs ON=`cutlass_80_tensorop_s1688gemm`(텐서코어 TF32) 1.31ms = 7.2× speedup 실재!**
    커널명 자체가 다름(sgemm vs tensorop) = **allow_tf32 플래그 정상 작동, TF32 7.2× 효과 진짜.** 검산: 4096³×2=137GFLOP,
    A100 fp32 19.5TF/s→7.0ms(측정 9.38 근접), TF32 156TF/s→0.88ms(측정 1.31 근접) = duration 신뢰.
  - 🐛 **근본 원인 = `_profile_ncu`의 `--launch-count 1`.** 커널 1개만 측정 → matmul 아닌 엉뚱한 커널/부분 잡아 OFF/ON 동률.
    **`latency_us`(=`gpu__time_duration.sum`, launch-count 1) 신뢰 불가.** ncu_breakdown group duration이 진짜. **= gain 가린 측정 버그.**
  - ➡️ **함의: gain 입증 길 열림.** matmul은 TF32로 진짜 7.2× 빨라짐 → 진화 ON(fp32룰 retire→TF32 켬)이 더 빠른 커널 만드는 것
    측정 가능. 단 sigmoid/groupnorm=메모리바운드 진짜 null, llama=attention 지배(matmul 23%)라 TF32 무효 = 별개 사유 유효.
    **다음 = executor latency를 ncu_breakdown 방식(전체 duration)으로 고치고 진화 ON/OFF gain 라운드.**
- **두 축 동시 (한 문제서 retire + gain)** — future work, **구조적 난제로 확인 (2026-06-29).** 시도:
  비합착 Triton matmul(`problems/matmul_tri/`)로 "fp32룰 헛발화→retire→uncoalesced→합착 fix→gain" 노렸으나,
  probe 측정 결과 **tc=True**(Triton `tl.dot`이 TF32 텐서코어 자동) → `fp32_no_tensorcore`(cond=`not tc`) **발화 막힘.**
  대신 `uncoalesced`(load_eff=0.0) 1순위 자연 발화 = **또 첫 발화가 맞는 룰 → retire 불요.** 근본 원인 = **시드 룰이
  워크로드에 잘 맞아 retire가 안 생김** (역설). 틀린 룰 자연 발화 = 메모리바운드(gain 천장), 맞는 룰 = compute(retire 없음)
  = **상호배타.** 연출 없이 한 문제서 둘 다 = 불가 확정. 부수 수확: uncoalesced 룰 비합착 Triton서 정상 발화 입증.
- **다문제 일반화** — gain=matmul 1문제, 진화 retire=sigmoid 1문제. 한 문제서 동시·여러 문제 일관은 미입증.
- 차별점 현 상태 = **"개념·메커니즘·룰진화 이득 OK, 최종 성능 gain은 환경 한계로 future work"** (정직 유지).

## 🧪 성능 gain 인프라 (재시도용 — 환경 갖춰지면)

- `loop/run_gain_hypcond.py` = **가설-조건부 콜백 드라이버** (run_gain_compare 큐 한계 극복).
  콜백이 발화 룰 라벨 보고 코드 선택 → 진화 차이가 코드 차이로 이어짐 → 성능 gain 갈림 측정 가능.
- groupnorm variants: `R_tf32.py`(fp32 가설 충실, null), `R_coalesced.py`(분할병렬, correctness PASS·latency 무효).
  - ⚠️ 분할병렬 버그 교훈: `mask = idx < group_elems`만이면 BLOCK>tile 시 타일 중복 합산.
    **`mask = (idx<group_elems) & (idx<tile_end)`** 필수. NTILE=1 PASS·NTILE=8 FAIL로 격리.
- llama variants: `R_seed_tf32off.py`(느림 base), `R_tf32on.py`(빠름)... 단 환경서 동률.
- baseline 신호 진단: `scratchpad/profile_baseline.py`, headroom: `scratchpad/check_llama_headroom.py`.

## 📋 2026-07-01 세션 작업 기록 (conv 진화 retire = 진화 축 다문제)

다문제 gain 노렸으나 conv gain=null(compute 천장) → **진화 축 다문제로 전환 성공.**

**한 일 (순서대로):**
1. **포폴 수치 차트 2장** — `charts/gain.svg`(matmul 6.4×)·`charts/retire.svg`(T4 retire) 순수 인라인 SVG + `index.html`(Pages). README 임베드. 08 노트에 **시각화 타이밍 규칙 박음**(phase 완료 때만, 매 세션 재고민 금지).
2. **conv2d seed 작성** — LeetGPU 2D Convolution(단일채널). naive Triton(픽셀당 K² 루프). executor 계약 준수. **canary로 A100 watch 생존+칩 실증**(재시작 불요, 문제는 REQ code 전송).
3. **conv seed 실측** — gate PASS(max_err 0), 신호: compute_tput 0.806·bw 0.027·occ 0.96·load_eff 0.44·tc=False. **compute-bound 확인.** match→`fp32_no_tensorcore` 발화 = **conv엔 헛가설**(TF32 GEMM용).
4. **합착 variant gain 실측 = null** — R1_coalesced(load_eff 0.44→0.70)나 latency 1690→2148us = **0.79× (느려짐).** occupancy 이미 0.96 포화 = 합착이 진짜 병목 아님. compute_tput 0.8 = **ALU 천장.** conv gain 없음(정직).
5. **✅ 진화 retire 재현 (A100 8R)** — variants R1~R6=seed복제(fp32 헛가설 반복 큐). **ON=fp32 4R→retire 2회→uncoalesced 전환, OFF=fp32 7R 영원(retire 0).** = sigmoid retire의 conv 재현 = **진화 축 2문제 입증.**

**산출:** `problems/2d_convolution/{solve.py,variants/R1-6.py,R1_coalesced.py}`, `charts/*.svg`, `index.html`. **교훈:** ⚠️ load_eff 낮다고 합착이 병목 아님(occupancy 포화 시 합착 개선해도 latency 악화). conv naive가 의외로 이미 효율(스칼라 계수 브로드캐스트).

## 📋 2026-06-30 세션 작업 기록 (칩 가드 ctx 배선 + T4 실측)

칩 컨텍스트 ctx 배선 완료 + Colab T4 첫 실측 = **세 번째 축(칩 이식성) 입증.**

**한 일 (순서대로):**
1. **ctx 배선** — `harness.run_loop(ctx=)`→`match(sig,rules,ctx)`, `runner.run_problem(ctx=)`, `run_gain_compare --chip=<key>`→`Context` 생성. 종전 `match(sig,rules)` ctx 누락 고침. self-check 7개 PASS. (커밋 loop master)
2. **matmul/variants/R1.py 추가** — 드라이버가 `R<N>.py`(순수숫자) 기대, 기존 `R_tf32on.py` 안 맞음.
3. **Colab T4 전환** (런타임 재시작 1회 — signals/rules/harness/runner=infra 변경). watch 재기동.
4. **T4 실측** — `run_gain_compare matmul ... --chip=t4`. RES 신호: compute_tput 0.967(sgemm 포화), occ 0.489, reg 122, tc=False, latency 48.4ms.
5. **칩 가드 ✅** — 같은 fp32 matmul 신호서 **A100=`fp32_no_tensorcore` 발화, T4=가드(chip_cap=tf32) 차단→`reg_pressure`.** 칩만 바꿔 헛가설 사전차단 실측.
6. **reg_pressure=헛가설 규명** — cond 맞으나 compute 포화라 무효(latency 평평). retire 시도(8R)했으나 patience3==fail3 충돌로 5R converged, retire 미관찰. retire는 sigmoid서 이미 입증 → T4 재입증 불요.

**산출:** `harness/runner/run_gain_compare.py` ctx 배선, `matmul/variants/R1.py`. 문서 [[07-chip-lang-context-design]] §7-b/7-c 갱신.

## 📋 2026-06-30 후속 (T4 retire 관찰 + raw_script 인프라)

지난 세션 미관찰이던 **T4 reg_pressure retire를 직접 관찰** — 한 환경서 이중방어 동시 입증.

**한 일 (순서대로):**
1. **patience 충돌 진단·픽스** — `harness.py CONVERGE_PATIENCE 3→6`. 종전 3==`RETIRE_AFTER_FAILS`(3)이라 수렴이 retire보다 먼저 터져 미관찰. retire 조건(`fail>=3 & conf<=0.25 & n>=4`) 채울 4R+ 확보. self-check 3/3 PASS.
2. **watch raw_script 분기 추가** — `execute_request`에 `cmd['raw_script']` 우선 분기 = 임의 파이썬 subprocess 실행(executor 우회). 새 측정 = REQ에 데이터로 실어 보냄 → **Colab 재시작 영원 불요**(sys.modules 캐시 회피). 운용교훈 "마찰 영구해결 길" 구현. self-check 3케이스(정상/exit/bad-JSON) PASS.
3. **Colab T4 재시작** (1회 — patience·raw_script=infra 변경) + reset --hard origin/master + watch 재기동.
4. **canary** — raw_script GPU probe로 chip=Tesla T4 sm75 확인 (raw_script 분기 실측 첫 작동).
5. **T4 8R gain_compare** (`matmul --chip=t4 --latency`) — **결과: ON retire=1 > OFF retire=0.** reg_pressure 헛가설 8R 발화, ON 트랙만 측정누적해 폐기. latency 평평 48.4ms(T4 fp32 천장, gain=null 정직). ON≈OFF latency(코드 R1 불변)=차이는 retire에서만=차별점 분리 정확.

**산출 코드:** `harness.py`(patience 6), `watch.py`(raw_script 분기+self-check). **교훈:** ⚠️ 백그라운드 폴링이 mailbox서 동시 git 작업하면 primary 프로세스 rebase 충돌로 죽음(rc=128) — 모니터링은 `ps -p` 프로세스 생존만, git 무접촉.

## 📋 2026-06-29 세션 작업 기록 (gain 돌파 + 두 축 정리)

큰 반전 세션. "환경 한계 null" → **측정 버그 발견 → gain layer 첫 돌파 → 두 축 동시 구조적 불가까지 정직 규명.**

**한 일 (순서대로):**
1. **포트폴리오 차별점 노트 작성** — [[05-portfolio-differentiator]] (한) + EN + 깃헙 README + `docs/SETUP.md`. 방법론 톤(가설→ablation→측정검증), 청중별 포지셔닝.
2. **llama ncu 비중** — `executor.ncu_breakdown` 신규. attention(flash) 54% > matmul 23% → matmul 비병목 확정.
3. **compute-bound matmul 추가**(`problems/matmul/`) — TF32 OFF/ON parity. 처음 95ms 동률="환경 한계" 오결론.
4. **🐛 측정 버그 발견·수정** — `_profile_ncu --launch-count 1`이 커널 1개만 측정. 전체 duration 합산으로 고침(커밋 `62235bb`). 재측정 **OFF 9.4ms(sgemm) vs ON 1.5ms(tensorop) = 6.4× 실재.** 커널명·FLOP 검산 확인.
5. **✅ gain layer 첫 돌파** — matmul 진화 라운드: 루프가 R0(fp32 9.6ms)→`fp32_no_tensorcore` 발화→R1(TF32 1.5ms). "측정→가설→재작성→더 빠른 커널" 첫 실증. 단 진화 ON≈OFF(첫 룰이 맞아 retire 불요).
6. **두 축 동시 시도→구조적 불가 확정** — 비합착 Triton matmul(`problems/matmul_tri/`) probe: tc=True(tl.dot TF32 자동)+load_eff=0.0 → 또 첫 발화가 맞는 룰. 시드 룰이 잘 맞아 retire 안 생김(역설). future work.

**산출 코드:** `executor.ncu_breakdown`+`_profile_ncu` 수정, `problems/{matmul,matmul_tri}/`, `run_gain_hypcond` matmul 분기.

## 🔖 다음 할 일

차별점 = **메커니즘 OK + gain 축(matmul 6.4×) 단일문제 + 진화 축(sigmoid retire) 단일문제 = 각각 실증.** 동시·일반화 future work.

**재개 시 선택지:**
0. **(칩 이식성 — ✅ 칩 가드 + ctx 배선 + T4 실측 완료 2026-06-30)** [[07-chip-lang-context-design]]. 칩·문법이 룰 cond에 1급 변수 아님(rationale 주석에만 A100·Triton) = matmul_tri/llama 난항 일부 원인. 설계+구현 확정: **칩=1급**(CHIP_CAPS 박힌 사실, 룰 cond에 `ctx.cap` 가드), **문법=신호 흡수**(Mojo 환경 0이라 1급 안 만듦). 황금규칙=**사용자/LLM은 Context(무대)만 설정, 최적 룰은 측정→진화가 발견**(룰 손수정 금지=차별점 보존).
   - ✅ **구현 + 배선 + T4 실측 완료**: `signals.py`에 `Context(chip,lang)`+`CHIP_CAPS`+`detect_chip`+`cap()`, `rules.py`에 `Rule.chip_cap`+2룰 `chip_cap="tf32"`+`match(sig,rules,ctx)`. **ctx 배선**: `harness.run_loop(ctx=)`→`match`, `runner.run_problem(ctx=)`, `run_gain_compare --chip=`. self-check 7개 PASS.
   - ✅ **T4 실측 입증 (Tesla T4 sm_75)**: matmul fp32 4096² → **A100선 `fp32_no_tensorcore` 발화(TF32 6.4×), T4선 가드 차단→`reg_pressure` 발화.** = 같은 신호서 칩만 바꿔 헛가설(TF32) 사전 차단 실증. compute_tput 0.967=sgemm 포화, latency 48.4ms 평평(T4 fp32 천장, gain=null 정직).
   - ✅ **reg_pressure retire 관찰 완료 (2026-06-30 후속)** — `CONVERGE_PATIENCE 3→6` 픽스로 4R+ 확보. T4 8R: **ON retire=1 > OFF retire=0.** reg_pressure 헛가설 8R 발화, ON만 측정누적 폐기. **= 한 환경서 칩가드(사전차단)+진화retire(사후폐기) 이중방어 동시 실측.** + watch raw_script 분기로 향후 재시작 불요.
   - **이중 방어 그림**: 칩 가드=알려진 헛가설 사전차단(선언적) + 진화 retire=모르는 헛가설 사후폐기(측정누적). 차별점 격상.
   - ⬜ **미완 = 칩 자동탐지** (지금 `--chip` 수동 명시, 입증엔 충분). watch가 `get_device_capability`로 RES에 싣는 배선 = 무인화 시. + patience 튜닝해 T4 retire 관찰(선택).
1. **(CudaForge 비교 실험 — 신규 설계됨)** [[06-cudaforge-comparison-design]]. 판정 주체(우리 룰 vs Judge LLM)만 통제변수. **실행 전 2개 미완**: (a) executor에 nsys/torch 측정 배선(→Colab 재시작 1회), (b) judge_mode arm 재구현(CudaForge Coder+Judge, 논문 프롬프트 verbatim). 그다음 A/B/C 문제셋×arm 매트릭스. **H1성능 불확실/H2품질 차별점 본질 — 어느 결과든 차별점은 성능과 독립.**
2. **(포트폴리오 마무리)** 현 차별점 서술 충분 — 노트 한/영·README 다 gain 반영됨. public 전환 가능(민감정보 스캔 clean).
   - ✅ **시각화 전부 완료** [[08-visualization-ideas]]. **Mermaid 2장** [[09-architecture-diagrams]] (루프 메타루프 + 칩 가드, mmdc 렌더 PASS, README 임베드 push). **수치 차트 2장 완료 (2026-07-01)**: `charts/gain.svg`(gain 막대 6.4×), `charts/retire.svg`(retire 곡선 ON1/OFF0). **순수 인라인 SVG 채택**(matplotlib 로컬 없음→설치 회피, 한 소스가 README `<img>`+Pages 양쪽). `index.html` = GitHub Pages 포폴 페이지(세 축 스토리 한/영 병기). XML+좌표 검증 PASS, push 완료(커밋 `b466e39`).
   - ⬜ **남음 = Pages 활성화** (repo Settings→Pages→master `/` 루트, 사용자 수동 1회) → `alexxony.github.io/gpu-solver-loop/` 라이브.
3. **(다문제 일반화)** — **진화 축 = ✅ 다문제 완료 (sigmoid+conv, 2026-07-01).** gain 축은 여전히 matmul 1문제.
   - ✅ **conv2d 진화 retire (A100 8R)**: conv서 `fp32_no_tensorcore` 헛발화(TF32 무효) → ON=retire 2회→uncoalesced 전환, OFF=영원. sigmoid retire 재현 = 진화 축 2문제.
   - ⚠️ **conv gain = null** (합착 variant 0.79×, compute_tput 0.8 천장). conv은 진화 축엔 좋으나 gain 축 안 됨. **교훈: load_eff 낮아도 occupancy 포화면 합착 개선이 latency 악화.**
   - ⬜ **gain 다문제 미완** = matmul 외 gain 확실한 compute 문제 필요 (batched GEMM 등). conv은 gain 실패라 gain 다문제엔 부적합.
   - ⬜ **두 축 동시 여전히 안 됨** = conv도 retire는 되나 그 후 gain 없음(합착 null) = matmul_tri 벽 재확인.

**(2026-06-29 후속) 다신호 확장 완료** — signals.py에 nsys(`launch_gap_pct`)·torch(`op_name/op_weight/op_shape`) 파서, rules.py에 `launch_overhead`·`attention_dominant` 룰 추가(가드 포함, self-check PASS). **왜 여태 ncu만?** = 룰 신호가 전부 커널 내부 메트릭이라 nsys 자리 안 만든 spec-구현 드리프트(정정). **단 파서·룰만 = executor 측정 미배선** → 새 룰 실발화는 executor가 nsys/torch 측정해 Signal 채워야(재시작 1회).
- ⚠️ mailbox 청소 = `cmd/* result/* done/*` 전부 (done 마커 확장자 없음).
- harness latency mode = `--latency` 플래그. `_profile_ncu`는 이제 전체 커널 duration 합산(launch-count 무제한).
- watch 살아있음(2026-06-29 세션 말). 문제만 추가는 재시작 불요, executor/loop 코드 바꾸면 재시작 필수.

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
- **Colab sys.modules 캐시**: loop 코드(executor/watch/signals) 바꾸면 **런타임 재시작** 필수 (재clone만으론 구버전). 단 **문제만**(새 solve.py/variant=REQ 전송) 바꾸면 재시작 불요. → **마찰 영구해결 길 = watch에 `cmd['raw_script']` 분기(임의 파이썬 subprocess 실행). 넣으면 새 측정 추가 시 executor 안 고쳐 재시작 영원 불요.** 미구현(별도 세션). breakdown은 executor 분기로 했으니 1회 재시작 필요.
- **노트북 ≠ 실험 셀**: `colab_mailbox.ipynb` = watch 전용. 측정/실험 셀 넣지 않음.
- **mailbox 정리**: gain 라운드 후 cmd/result/done clean + commit.

## 비용/환경 메모
- generate(LLM) = API 호출 = GPU 불요 → 로컬. 제품 무인 = RealGenerator + `ANTHROPIC_API_KEY`(user 미발급).
- PoC 입증 = CallbackGenerator(대행, 키 0). user = Claude 구독만(API 지갑 별도).
- **제공자 교체 경로 (call_fn 어댑터 1개만 교체, loop 코드 불변): 지금=Claude Code 세션 대행(키 0) → 차후=GLM-4.6/DeepSeek API(OpenAI 호환, 저렴) → 최후=로컬 LLM(vLLM/Ollama, 키 0·전기만). user 방향 = claude 탈피·최종 로컬 온리. 약한 LLM이어도 룰 진화 ablation 유효(가설 가이드 효과 오히려 큼). ⚠️ 로컬 시 generate↔measure GPU 경합 = 별도 GPU 또는 시분할.**
- A100 저활용 = 작은 커널 PoC 본질 (낭비 아님). 풀활용 원하면 로컬 LLM(Qwen) — 설정 큼, 보류.

## 링크
- [[05-portfolio-differentiator]] — 포트폴리오 차별점 서술 (방법론 톤, 청중별 포지셔닝)
- [[05-portfolio-differentiator-EN]] — 위 영문판 (깃헙 README·해외 채용용)
- [[GPU-Solver-MOC]] — 문서 허브
- [[04-multiproblem-round-design]] — A/B 실험 상세
- [[03-git-mailbox-runner]] — 인프라 설계 + e2e
- [[02-prior-art-survey]] — 차별점 검증
- [[2026-06-22-agentic-gpu-optimizer-design]] — 설계 스펙
