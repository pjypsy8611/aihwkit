# aihwkit 협업 가이드

둘 다 이 리포에 **write 권한**이 있습니다. `master`에 바로 push 해도 됩니다.
PR 승인 같은 절차는 없습니다. 대신 아래 두 가지만 지키면 사고가 안 납니다.

> **철칙 1. 작업 시작 전에 항상 `git pull`**
> **철칙 2. `git push --force` 금지** (필요하면 `--force-with-lease`)

---

## 처음 한 번 (상대방이 할 것)

초대 메일 수락 후:

```bash
git clone https://github.com/<소유자>/aihwkit.git
cd aihwkit

# 본인 신원 등록 (안 하면 커밋이 안 됩니다)
git config user.name  "이름"
git config user.email "깃헙에_등록된@이메일"

# IBM 원본을 upstream으로 연결 (선택)
git remote add upstream https://github.com/IBM/aihwkit.git

# 충돌 히스토리를 깔끔하게
git config pull.rebase true
```

빌드는 원래 aihwkit 방식 그대로입니다 (`README.md` 참고).

---

## 매일 작업 흐름

```bash
git pull                      # 1. 항상 먼저
# ... 코드 수정 ...
git add -p                    # 2. 바뀐 것만 골라 담기
git commit -m "fix(rpu): ..."
git pull                      # 3. push 직전에 한 번 더
git push                      # 4. 올리기
```

3번을 빼먹으면 push가 거부됩니다. 그때 당황하지 말고 `git pull` 하고 다시 `git push` 하면 됩니다.

## push가 거부될 때

```
! [rejected] master -> master (fetch first)
```

상대방이 그 사이에 push 한 겁니다. 정상 상황입니다.

```bash
git pull            # pull.rebase=true 라면 자동으로 내 커밋이 위로 정리됨
git push
```

## 충돌이 났을 때

```bash
git pull
# CONFLICT 메시지가 뜨면, 표시된 파일을 열어 <<<<<<< ======= >>>>>>> 부분을 정리
git add <해결한파일>
git rebase --continue
git push
```

엉켰다 싶으면 `git rebase --abort` 로 언제든 되돌릴 수 있습니다.

---

## 브랜치를 쓰면 좋은 경우

기본은 `master` 직접 작업이지만, 아래 상황에선 브랜치를 권합니다.

- 며칠 걸리는 큰 개편
- 될지 안 될지 모르는 실험
- 빌드를 깨뜨릴 가능성이 있는 변경

```bash
git checkout -b exp/sangkyu-chopped-transfer
# ... 작업 ...
git push -u origin exp/sangkyu-chopped-transfer
```

끝나면 합치기:

```bash
git checkout master && git pull
git merge exp/sangkyu-chopped-transfer
git push
git branch -d exp/sangkyu-chopped-transfer
```

브랜치 이름은 `feat/` `fix/` `exp/` `docs/` 접두사 + 본인 식별자를 붙이면 누구 것인지 바로 보입니다.

---

## 이 리포에서 특히 조심할 것

- **동시에 같은 파일 건드리지 않기.** 2인 직접 push 방식의 유일한 약점입니다. "나 오늘 `rpu_base.cpp` 만진다" 정도만 서로 알려도 충돌이 거의 사라집니다.
- **`uv.lock`, `pyproject.toml`, `setup.py`** — 충돌 나면 해결이 제일 골치 아픕니다. 의존성 변경은 **작업 전에 반드시 공유**하고, 한 사람이 바꿔서 push 하면 상대는 바로 pull 하세요.
- **큰 결과 파일 금지.** 100MB 넘으면 GitHub이 거부합니다. 실험 결과·체크포인트는 리포 밖 스토리지에 두세요.
- **노트북 출력 비우고 커밋.** `notebooks/`의 `.ipynb`는 출력까지 diff에 잡혀 충돌이 잦습니다.
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
- **빌드 산출물** — `.gitignore`가 `*.so`, `build/`, `dist/`, `*.egg-info/` 를 이미 처리합니다. 예외를 추가하지 마세요.

## IBM 최신 버전 따라가기

```bash
git fetch upstream
git checkout master && git pull
git merge upstream/master
git push
```

## 라이선스

IBM aihwkit은 MIT 라이선스입니다. private 사본은 허용되지만
`LICENSE.txt`와 원저작권 표시는 **삭제하지 마세요.**
