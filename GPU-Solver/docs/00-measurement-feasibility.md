---
title: Task 0 — 측정 가용성 검증 결과
status: done
created: 2026-06-22
tags: [project, gpu, leetgpu, colab, ncu, feasibility, task0]
---

# Task 0 — 측정 가용성 검증 (BLOCKING gate)

> [[2026-06-22-agentic-gpu-optimizer]] 구현 계획의 Task 0. spec [[2026-06-22-agentic-gpu-optimizer-design]] 섹션6 최우선 리스크("Pro 트레이스 필요성") 검증.

## 판정: **A — 진행 가능**

측정 인프라 확보됨. 프로젝트의 신호 의존성(트레이스 메트릭) 해결.

## 검증 경로

수동 LLM 대역(시스템 미존재 단계). 문제 = LeetGPU "RGB to Grayscale" (Easy, Triton, A100-80G).

### 1. LeetGPU 무료 등급
- 커널 제출 → **PASS.**
- **순위(percentile)·속도 = 안 나옴.** 무료는 통과/실패만.
- → LeetGPU 무료는 신호원으로 부적합. percentile은 Pro 필요(제한적).

### 2. Colab Pro + ncu (대안 신호원)
- GPU: **NVIDIA A100-SXM4-40GB** — LeetGPU 타깃(A100)과 **칩 일치.** (칩 불일치 리스크 소멸)
- ncu: `/usr/local/cuda/bin/ncu`, Nsight Compute 2025.1.1.0 설치됨.
- torch 2.11.0+cu128, triton 3.6.0.
- 정합성: grayscale 커널 Example1 통과 (`[76.245, 149.685, 29.07, 128.0]`).
- 벽시계: 2048×2048에서 0.0672 ms, effective BW ~998 GB/s.
- **ncu 권한 통과 + 메트릭 추출 성공:**
  ```
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed   83.29 %
  sm__warps_active.avg.pct_of_peak_sustained_active         81.59 %
  ```
  (Colab이 GPU 카운터 막지 않음 — `ERR_NVGPUCTRPERM` 없음.)

## 측정 아키텍처 결정

| 측정원 | 주는 것 | 제한 | 용도 |
|---|---|---|---|
| **Colab Pro + ncu** | 절대 메트릭 (occupancy, DRAM throughput, stall) | 컴퓨트 한도 내 사실상 무제한 | **루프 신호원.** 매 라운드 가설엔진 입력 |
| **LeetGPU Pro** | percentile (리더보드 순위) | 제출 횟수 제한 | 최종 검증. 문제당 1~2회 "도장" |

**핵심**: 루프는 Colab ncu로 무제한 회전, percentile은 최종 커널만 LeetGPU Pro에 제출. → 회수 제한이 루프를 막지 않음.

## 포폴 곡선 = 2종 (목표 재정의)

- **메인**: Colab ncu 메트릭 개선 곡선 (occupancy/BW 라운드별 상승). 무제한·인과 직결.
- **검증 점**: 최종 커널의 LeetGPU percentile (실제 리더보드 도장). 희소하지만 강함.
- 합: "시스템이 메트릭을 개선했고(인과), 실제 순위로 이어졌다(검증)."

## ncu 호출 메모 (구현 시 재사용)

- `--metrics <list>` 뒤 프로그램 앞에 **`--` 구분자 필수** (이 ncu 버전은 `--` 없으면 메트릭 문자열을 실행파일로 오인).
- 호출 예: `ncu --metrics <list> --target-processes all -- python probe.py`
- subprocess 리스트 인자로 호출 (셸 줄바꿈/백슬래시 문제 회피).
- `%%writefile` 셀과 `!ncu` 셀은 **반드시 분리** (같은 셀이면 `!`가 파일에 글자로 써짐 → SyntaxError).

## 잔여 리스크 (해소되지 않음)

- **LeetGPU Pro 제출 제한 수치 미확인** — 최종 검증 빈도 설계에 영향. Pro 가입 시 실측 필요.
- **A100 크레딧 소진** — Colab Pro A100은 가용성 변동. 부족 시 T4/L4 폴백 + 칩차이 명시.

## 산출물 정책

- 이 문서만 커밋. **probe 커널 코드는 커밋 안 함** (일회성 측정용, 라이선스/스코프 회피).
