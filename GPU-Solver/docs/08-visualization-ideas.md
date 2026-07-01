---
title: 08 — 포트폴리오 시각화 아이디어
created: 2026-06-30
tags: [gpu-solver, portfolio, visualization, ideas]
status: ideas
---

# 08 — 포트폴리오 시각화 아이디어

> 🎯 목적: 추상적 결과(6.4× gain·룰 진화·칩 가드)를 채용자가 **한눈에** 보게.
> 이 노트 = **아이디어·후보만**. 실제 차트 제작은 미정 (도구·범위 결정 후).
> 허브 [[GPU-Solver-MOC]], 상태 [[PROGRESS]], 차별점 서술 [[05-portfolio-differentiator]].

## 핵심 판단 (먼저)

- **vault ≠ 외부 면.** 채용자는 옵시디언 안 깖 → GitHub README·웹페이지를 봄.
- **Mermaid = 양쪽 다 됨** (vault 렌더 + GitHub ```mermaid 네이티브 렌더). 아키텍처·흐름은 Mermaid로.
- **수치 차트(막대·곡선)** = Mermaid 부적합 → 정적 PNG(matplotlib) or 단일 HTML(JS 차트) 필요.
- 데이터는 **이미 손에 있음** (코드 하드값 + T4 8R 로그). ledger 파싱 불요.

## 시각화 후보 (우선순위 순)

### ① 진화 ON/OFF retire 비교 — ★ 차별점 핵심
- **보여줄 것**: T4 8R서 ON=retire 1회→발화룰 전환 vs OFF=reg_pressure 영원 헛가설.
- **데이터**: T4 8R 로그. ON retire=1, OFF retire=0. latency 곡선 둘 다 평평(48.4ms).
- **형태**: 2-트랙 타임라인 (라운드별 발화룰 색칠 + retire 지점 마커). 또는 막대(retire 수 1 vs 0).
- **왜 1순위**: CUDAMaster 등 선행이 못 하는 "룰이 측정으로 폐기" = 유일 차별점. 글보다 그림이 압도.
- **주의**: latency 곡선 평평 = gain 아님 → "차이는 retire에서만"을 캡션에 명시 (오해 방지).

### ② gain 막대 — matmul 6.4×
- **보여줄 것**: fp32 9.6ms → TF32 1.5ms = 6.4× (A100). "측정→가설→재작성→더 빠른 커널".
- **데이터**: 코드 하드값 (`ampere_sgemm` 9.38ms vs `cutlass_tensorop` 1.31ms).
- **형태**: 2-막대 (before/after) + 6.4× 화살표. 커널명 라벨(sgemm vs tensorop)로 "진짜 다른 커널" 강조.
- **왜**: 성능 임팩트 즉시 전달. 단 1문제뿐 = "compute-bound 1례" 정직 캡션.

### ③ 칩 가드 분기 — 같은 신호, 칩 따라 다른 룰
- **보여줄 것**: 동일 fp32 matmul 신호 → A100=`fp32_no_tensorcore` 발화(TF32 6.4×) vs T4=가드 차단→`reg_pressure`.
- **데이터**: A100·T4 실측 (07 노트).
- **형태**: 분기 다이어그램 (Mermaid flowchart) — 신호 1개 → 칩 판정 → 갈라진 발화룰. **Mermaid로 GitHub 직접 렌더 가능.**
- **왜**: "환경을 룰 1급 입력으로" 설계를 한 장으로. 이중 방어(가드+retire) 스토리.

### ④ 루프 아키텍처 흐름 — Mermaid
- **보여줄 것**: gen→gate→profile→hyp(match)→ledger→evolve(retire/propose)→다음 라운드.
- **형태**: Mermaid `flowchart` (또는 `stateDiagram-v2`). 진화 피드백 화살표(evolve→rules) 강조 = 메타루프.
- **왜**: 전체 그림. **Mermaid = vault+GitHub README 양쪽 재사용** (의존성 0). **가장 먼저 만들기 쉬움.**

### ⑤ 신호 프로필 — 문제마다 다른 지문
- **보여줄 것**: llama(컴퓨트 지배)·sigmoid(메모리)·groupnorm(reduction) 신호 프로필 차이 = 룰 발화 근거.
- **데이터**: 3문제 ncu 신호 (03 노트).
- **형태**: 레이더 차트 (축=compute_tput/bw_pct/occupancy/load_eff) 또는 그룹 막대.
- **왜**: "신호가 다르니 다른 룰 발화" = 분류 정당성. 단 보조 (①②③보다 후순위).

## 도구 선택 (미결정 — 결정 후 채움)

| 도구 | 장점 | 단점 | 적합 |
|---|---|---|---|
| **Mermaid** | vault+GitHub 양쪽 렌더, 의존성 0 | 수치 차트 약함 | ③④ (흐름·분기) |
| **matplotlib PNG** | README 박기 쉬움, 표준 | 로컬 설치 필요(현재 없음), 정적 | ①②⑤ (수치) |
| **단일 HTML+JS** | 인터랙티브, 의존성 0, Pages 호스팅 | 제작 약간 큼 | 포폴 전용 페이지 |
| **obsidian-charts** | vault 내 ```chart 블록 | **GitHub서 깨짐**, 플러그인 필요 | vault 내부만 |

## 권장 진행 (가벼운 순)

1. ✅ **④ 루프 아키텍처 Mermaid** — 완료 [[09-architecture-diagrams]] ① (렌더 검증 PASS).
2. ✅ **③ 칩 가드 분기 Mermaid** — 완료 [[09-architecture-diagrams]] ② (렌더 검증 PASS).
3. ✅ **①② 수치 차트 = 완료 (2026-07-01).** 순수 인라인 SVG 채택 (matplotlib 로컬 없음, node만 있음).
   - `charts/gain.svg` = ② gain 막대 (fp32 9.6ms→TF32 1.5ms 6.4×, 커널명 sgemm≠tensorop 강조).
   - `charts/retire.svg` = ① retire 곡선 (T4 8R, ON retire=1 vs OFF retire=0, latency 평평 캡션 명시).
   - `index.html` = GitHub Pages 단일 포폴 페이지 (두 SVG + 세 축 스토리, 한/영 병기).
   - **왜 순수 SVG**: 한 소스가 README `<img>` + Pages 양쪽 재사용, 의존성 0 (Chart.js 캔버스는 README 임베드 불가라 배제).
   - README 임베드: §3(gain), §"두 축"(retire) 2곳. XML wellformed + 좌표 sanity 검증 PASS. push 완료 (커밋 `b466e39`).

## 미결정 → 결정됨 (2026-07-01)

- [x] 외부 면 형태: **둘 다** — README `<img>` SVG + Pages `index.html`.
- [x] 수치 차트 도구: **순수 인라인 SVG** (matplotlib PNG 아님 — 로컬 미설치, 설치 회피).
- [ ] ⑤ 신호 프로필 = 보조라 생략 (①②로 충분 판단).
- [x] 차트 캡션 언어: **한/영 병기.**
- ⬜ **남음 = Pages 활성화** (repo Settings → Pages → master `/` 루트, 사용자 수동 1회). 그러면 `alexxony.github.io/gpu-solver-loop/` 라이브.

## 링크
- [[05-portfolio-differentiator]] — 차별점 서술 (이 시각화가 뒷받침할 텍스트)
- [[05-portfolio-differentiator-EN]] — 영문판 (README용)
- [[07-chip-lang-context-design]] — ③ 칩 가드 데이터 출처
- [[03-git-mailbox-runner]] — ⑤ 신호 프로필 데이터 출처
- [[PROGRESS]] — 전체 진행 현황
