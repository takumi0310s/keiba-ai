# git push 詰まり 修復 (5/15)

実行: 2026-05-15、 Opus 4.7
状態: AI 自律 step A 完了、 destructive op (history rewrite) は user 認可待ち

## 問題

```
remote: error: File data/v20_training_data_full.csv is 112.83 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
```

`data/v20_training_data_full.csv` (118 MB) が **commit 8dfb595f (Phase 26)** で history に残存。 GitHub は push 全 commits を 検証 するので、 file が history に ある限り **push 永遠に失敗**。

## ★ AI 自律 完了 (step A) ★

| step | 内容 | 状態 |
|------|------|------|
| A1 | `.gitignore` 大幅補強 (50+ patterns 追加) | ✅ 本日 |
| A2 | `git rm --cached data/v20_training_data_full.csv` (index から削除、 filesystem 保持) | ✅ 本日 |
| A3 | 新 commit で 「削除」 を 記録 (push 試行) | 次 step |

→ **step A3 後 でも、 file が history に 残るため push まだ失敗**。 step B (history rewrite) が **必須**。

## ★ user 認可 必要 (step B、 destructive op) ★

### Option 1: git filter-repo (推奨、 modern + safe)

```bash
# 1. git-filter-repo install (Windows: pip install git-filter-repo)
pip install git-filter-repo

# 2. file を history 全 commits から 削除
cd C:\Users\takum\keiba-ai
git filter-repo --invert-paths --path data/v20_training_data_full.csv --force

# 3. remote 再 add (filter-repo は remote 自動 削除)
git remote add origin https://github.com/takumi0310s/keiba-ai.git

# 4. force push
git push origin main --force
```

★ 影響 ★:
- main branch 全 commits の SHA hash 変更
- 他 開発者 (もし いれば) は git clone 再実行 必要
- 本 prj は user 単独運用 のため 影響 最小

### Option 2: BFG Repo-Cleaner (高速、 別 install)

```bash
# 1. BFG dl (https://rtyley.github.io/bfg-repo-cleaner/)
curl -o bfg.jar https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 2. file 削除
java -jar bfg.jar --delete-files v20_training_data_full.csv .git

# 3. cleanup + force push
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push origin main --force
```

### Option 3: git lfs migrate (LFS 化)

```bash
# 1. LFS install (git lfs install)
git lfs install

# 2. 過去 file を LFS に migrate
git lfs migrate import --include="data/v20_training_data_full.csv"

# 3. force push
git push origin main --force
```

★ 注意 ★: LFS は GitHub の月 1GB 無料 quota あり、 超えると 課金。

### Option 4: GitHub Release で 大 file 別保管

```bash
# 1. file を GitHub Release asset として upload (手動)
# 2. repo から history 全削除 (Option 1-3 と組合せ)
# 3. README に 「v20_training_data_full.csv は Release から dl」 と明記
```

## ★ 推奨: Option 1 (git filter-repo) ★

理由:
- modern、 active maintained (filter-branch は obsolete)
- safe API (BFG より controlled)
- pip install で 完結
- 本 prj 用途 に最適

### user 実行 手順 (5 分)

```cmd
cd C:\Users\takum\keiba-ai
python -m pip install git-filter-repo

git filter-repo --invert-paths --path data/v20_training_data_full.csv --force

git remote add origin https://github.com/takumi0310s/keiba-ai.git

git push origin main --force
```

★ 注意 ★:
- `--force` push は **既 remote の commit を 上書き**、 注意必要
- 開始前に 全 work commit 済 確認
- 失敗時は `git reflog` で 復旧可能

## ★ 別 option: 並行 修復 (commit 8dfb595f を 編集) ★

過去 commit を 編集して file 削除:

```bash
# 1. interactive rebase
git rebase -i 8dfb595f~1

# 2. editor で 8dfb595f を 'edit' に 変更

# 3. file 削除 + amend
git rm --cached data/v20_training_data_full.csv
git commit --amend --no-edit

# 4. 続行
git rebase --continue

# 5. force push
git push origin main --force
```

★ 注意 ★: 8dfb595f 以降 の 全 commits が rewrite されるので conflict 多い 可能性。 Option 1 が安全。

## ★ AI 自律 step A 反映 commit ★

`.gitignore` 補強 + `git rm --cached` を 1 commit で:

```bash
git add .gitignore
git rm --cached data/v20_training_data_full.csv  # done
git commit -m "★ TOS review + .gitignore 強化 + 大 csv index 削除 (push 修復 Step A) ★"
```

→ filesystem 保持、 index 削除、 .gitignore で 今後 防止。

## ★ destructive op は AI 自律 不可、 user 5 分作業 で 完了 ★

AI 不可 理由:
- auto-mode classifier が history rewrite を hard block
- user verbal authorize 不足、 settings.local.json の permission rule 必要
- 安全側 設計 (= 良い security boundary)

→ user が 上記 5 分 手動 実行 で push 修復 完了。

## ★ 修復後 期待 ★

```
git push origin main
→ Enumerating objects: 1234, done.
→ remote: Resolving deltas: 100%
→ To https://github.com/takumi0310s/keiba-ai.git
   12345..67890  main -> main
```

→ V15 production 自動運用 + 全 commits remote sync OK、 1 週間 放置問題 解消。

## V15 投資保護 完全 (本日も遵守)

- V15 .pkl.gz / predict_core / app.py 完全不変
- history rewrite は repo metadata のみ、 production model / data 不変
- 累計 +5,240 円 / 撤退余裕 +55,240 円 ※ 旧 +13,530 / +63,530 は drift、 5/16 P0-1 真値 (docs/ROI_DISCREPANCY_2026_05_16.md)

## まとめ

| step | 担当 | 状態 |
|------|-----|------|
| A. .gitignore 強化 + git rm --cached | AI | ✅ 本日 |
| B. history rewrite + force push | **user 5 分** | ⏳ |

★ AI 自律可能 範囲 は 完了、 残 step B は user 認可 + 5 分 手動 で 完了 ★
