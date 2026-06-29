---
title: GPU-Solver 진행 현황 (재개 진입점)
created: 2026-06-26
updated: 2026-06-27
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
차별점 = **두 축 각각 입증: gain 축=matmul 6.4×, 진화 축=sigmoid 오발화 retire. 다문제 동시·일반화는 future work.**

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
1. **(CudaForge 비교 실험 — 신규 설계됨)** [[06-cudaforge-comparison-design]]. 판정 주체(우리 룰 vs Judge LLM)만 통제변수. **실행 전 2개 미완**: (a) executor에 nsys/torch 측정 배선(→Colab 재시작 1회), (b) judge_mode arm 재구현(CudaForge Coder+Judge, 논문 프롬프트 verbatim). 그다음 A/B/C 문제셋×arm 매트릭스. **H1성능 불확실/H2품질 차별점 본질 — 어느 결과든 차별점은 성능과 독립.**
2. **(포트폴리오 마무리)** 현 차별점 서술 충분 — 노트 한/영·README 다 gain 반영됨. public 전환 가능(민감정보 스캔 clean).
3. **(다문제 gain 일반화)** matmul 외 compute-bound 문제서도 6.4×류 gain 일관 확인.

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
