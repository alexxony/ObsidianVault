---
title: 09 — 아키텍처 다이어그램 (Mermaid)
created: 2026-06-30
tags: [gpu-solver, portfolio, visualization, mermaid]
status: active
---

# 09 — 아키텍처 다이어그램 (Mermaid)

> 🎯 [[08-visualization-ideas]] ④루프흐름 + ③칩가드 제작분.
> Mermaid = vault 렌더 + GitHub README ```mermaid 네이티브 렌더 (양쪽 재사용, 의존성 0).
> 코드 출처: `loop/harness.py`(run_loop), `loop/rules.py`(match), `loop/signals.py`(CHIP_CAPS).

## ① 루프 아키텍처 — 측정→가설→재작성→진화 메타루프

한 라운드 = generate→gate→profile→match(가설)→ledger→evolve. **차별점 = evolve가 rules를
되먹임**(retire/propose) = 룰표가 측정으로 진화. 선행(CUDAMaster 등)은 이 되먹임 화살표가 없음(정적).

```mermaid
flowchart TD
    seed([seed 커널 / 직전 변형]) --> gen[Generator<br/>LLM 코드 재작성]
    gen --> gate{Gate<br/>correctness}
    gate -->|FAIL| regen[재생성 라운드] --> gen
    gate -->|PASS| prof[Profiler<br/>ncu 신호 측정]
    prof --> sig[/Signal<br/>compute_tput·bw·occ·tc·reg/]
    sig --> match[Hypothesis Engine<br/>match sig × rules × ctx]
    ctx[/Context<br/>chip·lang 가드/] -.칩 능력 가드.-> match
    match --> hyp[발화 룰 = 다음 가설]
    hyp --> ledger[(Ledger<br/>라운드 이력)]
    ledger --> evolve[Rule Evolver<br/>성공·실패 누적]
    evolve -->|개선| promote[promote<br/>신뢰도 ↑]
    evolve -->|반복 실패| retire[retire<br/>틀린 룰 폐기]
    evolve -->|미설명 개선| propose[propose<br/>후보 룰 추가]
    promote -.되먹임.-> rules[(Rules 표)]
    retire -.되먹임.-> rules
    propose -.되먹임.-> rules
    rules --> match
    hyp -->|is_stop| done([포화 → 정직한 종료])
    hyp -->|계속| gen

    classDef diff fill:#ffe6cc,stroke:#d79b00,stroke-width:2px;
    class evolve,retire,promote,propose,rules diff
```

> 🟠 주황 = 차별점 (룰 진화 되먹임). CUDAMaster·KernelAgent 등 선행은 `rules`로 가는 점선 화살표 없음.

## ② 칩 가드 — 같은 신호, 칩 따라 다른 룰 발화

동일 fp32 matmul 신호라도 **칩 능력(CHIP_CAPS)이 헛가설을 사전 차단.** A100=TF32 가능→
`fp32_no_tensorcore` 발화(텐서코어로 6.4× gain). T4=TF32 없음(sm_75)→가드 차단→다음 적격 룰
`reg_pressure`. **= 환경을 룰 1급 입력으로** (07 설계). 실측 입증(2026-06-30).

```mermaid
flowchart TD
    sig[/같은 신호<br/>fp32 matmul<br/>tc=False, occ low/] --> guard{match: cond AND ctx.cap}

    guard -->|"A100 (sm_80)<br/>CHIP_CAPS.tf32=True"| a1[fp32_no_tensorcore 발화]
    a1 --> a2[가설: TF32 켜라]
    a2 --> a3([R1: cutlass tensorop<br/>9.6ms → 1.5ms = 6.4× ✅])

    guard -->|"T4 (sm_75)<br/>CHIP_CAPS.tf32=False"| t1[fp32_no_tensorcore 가드 차단]
    t1 --> t2[다음 적격 룰: reg_pressure]
    t2 --> t3([latency 평평 48.4ms<br/>= T4 fp32 천장, gain=null 정직])
    t2 -.8R 미개선 누적.-> t4[진화 retire<br/>ON=1 vs OFF=0]

    classDef chip fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;
    classDef guard fill:#ffe6cc,stroke:#d79b00,stroke-width:2px;
    class a1,a2,a3 chip
    class t1,t2,t3,t4 guard
```

> **이중 방어**: 칩 가드 = 알려진 헛가설(TF32 on non-TF32 chip) **사전 차단**(선언적, CHIP_CAPS) +
> 진화 retire = 모르는 헛가설 **사후 폐기**(측정 누적). 한 환경(T4)서 동시 관찰.

## 사용처

- **GitHub README**: 위 ```mermaid 블록 그대로 복붙 → GitHub 자동 렌더.
- **vault**: 이 노트가 곧 렌더 (옵시디언 Mermaid 네이티브).
- 수치 차트(gain 막대·retire 곡선)는 [[08-visualization-ideas]] ①② = PNG/HTML 별도 (미정).

## 링크
- [[08-visualization-ideas]] — 전체 시각화 후보·도구 비교
- [[07-chip-lang-context-design]] — ② 칩 가드 설계·실측 출처
- [[05-portfolio-differentiator]] — 차별점 서술
- [[PROGRESS]] — 진행 현황
