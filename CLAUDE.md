# CLAUDE.md — Obsidian Vault

이 폴더는 **Obsidian 보관함(vault)** 입니다. 모든 노트는 마크다운(`.md`) 파일이에요.
Claude Code가 이 폴더에서 작업할 때 아래 규칙을 따라 주세요.

## 절대 건드리지 말 것
- `.obsidian/` — Obsidian 앱 설정/플러그인/테마. 직접 수정·삭제 금지.
- `.trash/` — Obsidian 휴지통.
- `.git/` — 버전 관리 내부 파일.

## 노트 작성 규칙
- 새 노트는 마크다운으로 만들고, 가능하면 상단에 YAML 프론트매터를 둔다:
  ```yaml
  ---
  title: 제목
  created: YYYY-MM-DD
  tags: []
  ---
  ```
- 노트 간 연결은 Obsidian 위키링크 `[[노트 제목]]` 형식을 사용한다.
- 태그는 `#태그` 형식 또는 프론트매터의 `tags`에 적는다.
- 파일명은 노트 제목과 일치시키고, 가능하면 공백 대신 의미가 분명한 제목을 쓴다.

## 환경 메모
- 이 vault는 Windows 경로 `C:\ObsidianVault` 에 있다.
- WSL2의 Claude Code에서는 `/mnt/c/ObsidianVault` 로 접근한다.
- Obsidian(Windows 앱)과 Cowork도 같은 폴더를 본다. **같은 노트를 동시에 편집하지 않도록** 주의.

## 동기화
- 동기화/백업은 git으로 한다. 큰 변경 후에는 의미 있는 단위로 커밋한다.
- 기기별 상태 파일(`workspace.json` 등)은 `.gitignore`로 제외되어 있다.
