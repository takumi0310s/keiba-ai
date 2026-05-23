# FORCE PUSH INCIDENT レポート
**日付**: 2026-05-23 (21:27 JST 頃)
**対象リポジトリ**: https://github.com/takumi0310s/keiba-ai

---

## 1. 経緯

### 発生原因

`data/v20_training_data_full.csv` (114MB) が git 履歴に commit されていたため、
GitHub への通常 push が **100MB 制限エラー** で失敗した。

当該ファイルは `.gitignore` の line 135 に `data/v20_training_data_full.csv` として既に登録済みだったが、
**登録前に add/commit されていたため、歴史に残留していた**。

### 実行した操作 (時系列)

| 時刻 (JST) | 操作 |
|---|---|
| ~21:27 | `python -m git_filter_repo --path data/v20_training_data_full.csv --invert-paths --force` 実行 |
| ~21:27 | filter-repo が origin remote を自動削除 |
| ~21:27 | `git remote add origin https://github.com/takumi0310s/keiba-ai.git` で再登録 |
| ~21:33 | `git push --force origin main` 成功 |
| ~21:38 | 通常 commit 再開 (`[audit] paci_ninki_idx...`) |
| ~21:42 | 通常 push 成功 |
| ~22:04 | 次の commit 投入 (`[言語化] オオタニサーン vs ペッパーミル...`) |

---

## 2. git 操作ログ (reflog 抜粋)

```
# refs/remotes/origin/main の push 履歴
778e64db  refs/remotes/origin/main@{0}: 2026-05-23 22:07  update by push
a81c5a95  refs/remotes/origin/main@{1}: 2026-05-23 21:42  update by push
25a3eca5  refs/remotes/origin/main@{2}: 2026-05-23 21:33  update by push  ← force push 直後
```

```
# .git/filter-repo/ に残る痕跡
already_ran   : 2026-05-23 21:27 (実行日時確定)
commit-map    : 1064 行 (対象: 全 commit を走査)
changed-refs  : refs/heads/main (書き換えた ref)
first-changed-commits: 8dfb595f → edf094fb  (最初に差異が生じた commit)
```

### hash 書き換えの規模

| 項目 | 値 |
|---|---|
| commit-map 総行数 (header 除く) | 1,064 |
| hash が実際に変わった commit 数 | **189** |
| hash が変わらなかった commit 数 | 875 |
| 変更対象 ref | `refs/heads/main` のみ |

### 直前の HEAD hash の変化

filter-repo 実行前の main 先端 hash は `b679ce346a...` (ref-map の `old` 列から確認)。
force push 後の先端は `25a3eca5...`。その後の通常 commit で現在は `778e64db`。

---

## 3. 確認結果

### 3-1. V15 production モデル — 無事

| ファイル | パス | サイズ | 更新日 |
|---|---|---|---|
| keiba_model_v15_central.pkl.gz | プロジェクトルート | 2,099,552 bytes (~2MB) | 2026-04-08 |
| keiba_model_v15_central_live.pkl.gz | プロジェクトルート | 2,099,610 bytes (~2MB) | 2026-04-08 |

両ファイルは `.gitignore` 対象外かつローカルファイルのため、filter-repo の影響を受けない。
内部検証結果:

```
# keiba_model_v15_central.pkl.gz
LGB num_feature : 145  ✓
features list   : 145  ✓

# keiba_model_v15_central_live.pkl.gz
LGB num_feature : 145  ✓
features list   : 150 (Pattern B、booster 入力後 truncate で 145 になる)  ✓
auc (stored)    : 0.8939485520467574  ✓
ensemble_weights: {lgb: 0.5036, xgb: 0.4964, mlp: 0}  ✓
```

**V15 production は完全に無事。**

### 3-2. predict_core.py — feature count 記述確認

`tools/predict_core.py` line 478:
```python
(os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz'), False, 'v15 Pattern A (リークフリー, 145特徴量)'),
```

「145特徴量」の記述あり。実モデルの `num_feature()` = 145 と一致。

### 3-3. 失われたもの

| 項目 | 状態 |
|---|---|
| `data/v20_training_data_full.csv` の **git 履歴上の blob** | 削除 (これが目的) |
| `data/v20_training_data_full.csv` の **ローカルファイル** | 現存 (114MB、2026-05-12) |
| その他ソースコード / docs / 設定ファイル | 全て intact |
| 全 branch の コード内容 | intact (hash が変わっただけ) |
| dev/* / fix/* / feat/* ブランチ | intact (ref-map で old=new を確認) |

**ローカル上の .csv ファイル本体は削除されていない。** git 履歴から blob が消えただけ。

### 3-4. .gitignore の状態

`.gitignore` の line 135 に `data/v20_training_data_full.csv` が既登録済み。
line 129 に `data/v21_training_data_full.csv` も登録済み。
今後の再 add は防止される。

---

## 4. なぜ発生したか

### 根本原因

`.gitignore` へ登録するより先に `git add` → `git commit` されてしまった。
具体的には 2026-05-12 05:30 頃 (v20_training_data_full_builder.py 実行後) に
`v20_training_data_full.csv` が staging area に混入したと推定される。

filter-repo の `first-changed-commits` が指す commit:

```
edf094fb: "Phase 26 task #5+#8 完了: v20 merge fix + 5/17 動的 features tool"
date:      2026-05-12 05:32 +0900
```

この commit かその直前の commit でファイルが追加された。

### 副作用: origin 自動削除

`git-filter-repo` は安全策として **実行時に remote origin を自動削除** する。
再登録 (`git remote add`) を忘れると次回 push 先が消える点に注意。

---

## 5. 教訓と再発防止策

### 教訓

1. **大ファイル生成スクリプトの出力先は即座に .gitignore 登録する**
   - v20_training_data_full_builder.py を実装した時点で .gitignore に追加すべきだった
   - 「あとで登録する」は禁止

2. **push 前に `git status` で大ファイル混入チェックをする**
   - 100MB 超のファイルが staging にある場合は即 unstage

3. **filter-repo は origin を削除する**
   - 実行後は必ず `git remote -v` で確認し、ない場合は即再登録

4. **force push は全 collaborator の歴史を書き換える**
   - 本リポジトリはほぼ 1 人運用だが、Claude Code worktrees の古い checkout が残っている場合は `git fetch --force` が必要

### 再発防止策

#### A. 大ファイル生成スクリプトに .gitignore チェックを追加 (推奨)

新規 CSV/pkl を `data/` 以下に出力するスクリプトは、出力前に `.gitignore` への登録確認を組み込む。

```python
# 例: v20_training_data_full_builder.py 先頭に追加
import subprocess, sys
out_path = "data/v20_training_data_full.csv"
result = subprocess.run(["git", "check-ignore", "-q", out_path], capture_output=True)
if result.returncode != 0:
    print(f"[ERROR] {out_path} is NOT in .gitignore. Add it first.", file=sys.stderr)
    sys.exit(1)
```

#### B. pre-commit hook で 50MB 超ファイルをブロック

`.git/hooks/pre-commit` に追加:

```bash
#!/bin/bash
MAX_SIZE=$((50 * 1024 * 1024))  # 50MB
while IFS= read -r -d '' file; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
    if [ "$size" -gt "$MAX_SIZE" ]; then
        echo "ERROR: $file ($size bytes) exceeds 50MB limit. Add to .gitignore first."
        exit 1
    fi
done < <(git diff --cached --name-only -z)
```

#### C. push 前チェックリストへ追加

毎回の `git push` 前に以下を確認:

```bash
git diff --cached --stat | grep -E "^\s+[0-9.]+(M|G)" && echo "WARNING: Large file staged"
```

#### D. .gitignore の汎用ルール強化

現状は個別ファイル名指定。以下のパターン追加を推奨:

```gitignore
# Training data (large, regenerable)
data/*_training_data_full.csv
data/*_training_data*.csv
```

---

## 6. 現在の状態サマリー

| 項目 | 状態 |
|---|---|
| GitHub main 最新 hash | `778e64db` |
| ローカル main 最新 hash | `778e64db` (一致) |
| origin remote | `https://github.com/takumi0310s/keiba-ai.git` (正常) |
| V15 production モデル | **無事 (LGB num_feature=145 確認済)** |
| v20_training_data_full.csv | ローカル現存 / git 履歴から削除済 |
| 総 commit 数 | 903 |
| filter-repo が hash を書き換えた commit 数 | 189 / 1064 |
| その他コード / docs | 全 intact |

**production への影響: なし。V15 は完全に無傷。**
