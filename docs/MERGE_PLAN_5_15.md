# 5/15 22:00 dev branch merge plan (Session #74)

**作成**: 2026-05-09 (Session #74)
**実施予定**: 2026-05-15 22:00 (5/16 V18 trial 直前)
**main HEAD (5/9 時点)**: 5f5c3d43

---

## 0. 目的

5/16 V18 sib_w5 trial 投入のため、 dev branch 6 件 を main に merge。
V15 model + predict_core.py + daily_predict.py + app.py は **完全不変** で merge する。

---

## 1. merge 対象 6 branch

| # | branch | 主内容 | 想定 conflict |
|---|--------|--------|------------|
| 1 | `dev/training-poc` | Session #47-#72 PoC 集約 (基盤) | 低 (ほぼ docs/ + tools/) |
| 2 | `dev/audit-backtest` | AUDIT-1 + Session #69 / #70 | 低 (data/ + docs/) |
| 3 | `dev/two-stage` | Stage 2 二段階予測 (Session #48 + #65 + #68 + #72) | **中** (predict_core.py 周辺、 ただし新規 module で逃がす) |
| 4 | `dev/sprint1` | Sprint 1 5 idea (軽量改善) | 低 (新規 features module) |
| 5 | `dev/sprint2` | LEAK PASS idea のみ | 中 (sprint1 と同領域) |
| 6 | `dev/sprint6-kka` | KKA parser 修復 | **中** (race_id format 調整必要) |
| 7 | `dev/video-poc` | Phase 4 用保持 (5/16 では未活用) | 低 (新規 dir のみ) |

---

## 2. dependency / merge 順序

```
main (5f5c3d43)
  ├── 1. dev/training-poc       (基盤 PoC、 まず取込)
  ├── 2. dev/audit-backtest     (data/ + docs/、 干渉小)
  ├── 3. dev/sprint1            (新規 features、 単独)
  ├── 4. dev/sprint2            (sprint1 後、 LEAK PASS のみ cherry-pick)
  ├── 5. dev/two-stage          (Stage 2 module、 sprint 後)
  ├── 6. dev/sprint6-kka        (KKA、 race_id 調整後)
  └── 7. dev/video-poc          (Phase 4 保持、 最後)
```

理由:
- training-poc が他 branch の前提 (PoC 結果 docs を使うため)
- audit-backtest は独立 (data 系)
- sprint1 → sprint2 (順序依存、 sprint2 一部 idea が sprint1 module に依存)
- two-stage は sprint 後 (Stage 2 が sprint features を入力に取る)
- sprint6-kka は race_id format 確認後 (KKA parser 出力の整合性)
- video-poc は最後 (Phase 4 まで未使用、 干渉ゼロ目的)

---

## 3. 各 branch の merge 手順

### 共通前処理

```powershell
# main を最新化
git checkout main
git pull --rebase origin main

# main HEAD 確認 (5f5c3d43 から進んでいる場合は注意)
git log --oneline -5
```

### 1. dev/training-poc

```powershell
git fetch origin dev/training-poc
git merge --no-ff origin/dev/training-poc -m "Session #74 merge 1/7: dev/training-poc (Session #47-#72 PoC)"
# conflict なし想定
# 構文チェック (predict_core.py 触っていないことの確認)
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"
```

### 2. dev/audit-backtest

```powershell
git fetch origin dev/audit-backtest
git merge --no-ff origin/dev/audit-backtest -m "Session #74 merge 2/7: dev/audit-backtest (AUDIT-1 + #69 + #70)"
# conflict 想定なし (data/ + docs/)
```

### 3. dev/sprint1

```powershell
git fetch origin dev/sprint1
git merge --no-ff origin/dev/sprint1 -m "Session #74 merge 3/7: dev/sprint1 (5 idea LEAK PASS)"
# 新規 module は train/features_sprint1.py 想定 (干渉小)
python tests/test_features.py  # 5 項目 PASS 確認
```

### 4. dev/sprint2 (一部のみ)

```powershell
git fetch origin dev/sprint2
# 全 merge ではなく LEAK PASS idea のみ cherry-pick 推奨:
git log origin/dev/sprint2 --oneline | head -20
# LEAK PASS が確定している commit のみ pick
# (maiden / jump v2 などは 5/16 までに LEAK 確認できなければ 除外)
git cherry-pick <commit-hash>  # 該当 commit ごと
# 全 merge する場合:
git merge --no-ff origin/dev/sprint2 -m "Session #74 merge 4/7: dev/sprint2 (LEAK PASS subset)"
```

### 5. dev/two-stage

```powershell
git fetch origin dev/two-stage
git merge --no-ff origin/dev/two-stage -m "Session #74 merge 5/7: dev/two-stage (Stage 2 #48+#65+#68+#72)"
# 想定 conflict: predict_core.py の import 部 (新 module 追加)
# → 新規 module 配置で conflict 回避設計が前提
# conflict 発生時: 新 import は最後尾に追記、 既存 V15 logic 完全保護
python -c "import tools.predict_core" 2>&1 | tail -5
```

### 6. dev/sprint6-kka

```powershell
git fetch origin dev/sprint6-kka
# race_id format 調整 確認 (Session #53)
# KKA parser 出力 race_id が main の format と一致するか確認
git diff main origin/dev/sprint6-kka -- tools/jrdb_kka_parser.py | head -50
git merge --no-ff origin/dev/sprint6-kka -m "Session #74 merge 6/7: dev/sprint6-kka (KKA parser fix)"
# data/jrdb_kka_features.csv が untracked にあるが、 これは生成物で merge 対象外
```

### 7. dev/video-poc

```powershell
git fetch origin dev/video-poc
git merge --no-ff origin/dev/video-poc -m "Session #74 merge 7/7: dev/video-poc (Phase 4 retain)"
# data/v18/videos_5_9/ 系は untracked、 merge 対象外
# 動画解析 module は 5/16 では起動せず、 Phase 4 (7/1+) で活用
```

---

## 4. merge 後 必須 check

```powershell
# 1. 構文チェック
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"
python -c "import tools.predict_core" 2>&1 | tail -5
python -c "import tools.daily_predict" 2>&1 | tail -5

# 2. V15 model load 確認
python -c "import gzip, pickle; pickle.load(gzip.open('keiba_model_v135_central_live.pkl.gz')); print('V15 OK')"

# 3. 5 項目テスト
python tests/test_features.py

# 4. 1 レース予測 sanity (土曜分の URL 適当に)
# python tools/predict_one_race.py <race_id>

# 5. schtasks 既存 50 件 不変確認
schtasks /query /fo LIST 2>&1 | Select-String "Keiba" | Measure-Object -Line
```

期待値: schtasks Keiba 系 50 件 不変。

---

## 5. conflict 発生時 対処

| conflict 領域 | 解決方針 |
|------------|--------|
| `tools/predict_core.py` | **V15 logic 完全保護**。 新 import / 新関数は末尾追記。 既存 V15 関数は触らない |
| `tools/daily_predict.py` | 同上、 V15 path 不変 |
| `app.py` | 同上、 既存 UI 不変 |
| `train/features_v15_new.py` | sprint1 / sprint2 module は **新規 module** 推奨。 既存 features 触らない |
| `data/*.csv` (大型) | 通常 .gitignore 対象。 conflict 発生時は ours 採用 (main 側 維持) |
| `docs/*.md` | 両方残す方針 (`<<<<<<<` 手動解決) |

---

## 6. rollback 手順 (緊急時)

merge 後 V15 動作異常時:

```powershell
# 直前の main HEAD に戻す (merge commit のみ取消)
git log --oneline -10  # merge commit の hash 確認
git reset --hard <merge前 hash>  # 慎重に。 並行 Session 干渉リスクあり

# 推奨: 新 commit で revert (履歴保持、 安全)
git revert -m 1 <merge commit hash>
git push origin main
```

⚠ **destructive op (reset --hard / push --force) は並行 Session 干渉防止のため最終手段**。
通常は revert で対処。

---

## 7. archive 対象 branch (merge せず)

| branch | 理由 | 対処 |
|--------|------|------|
| `dev/sprint4` | V15.5 NO-GO (Session #50) | 削除しない、 archive tag のみ |
| `dev/nar-v5` | V5 NO-GO | 同上 |
| `dev/v20-expanding` | NO-GO (Session #55) | 同上 |
| `dev/v20-interaction` | NO-GO (Session #57) | 同上 |
| `dev/v20-ensemble` | **保持** (5/22+ V20 構築素材、 Session #56 AUC 0.90025) | merge せず branch 保持 |

archive tag 例 (実行は別 Session、 本 Session では実行しない):
```
git tag archive/sprint4-v15-5-nogo origin/dev/sprint4
```

---

## 8. 5/15 22:00 タイムライン

| 時刻 | 作業 |
|------|------|
| 22:00 | main pull --rebase + HEAD 確認 |
| 22:05 | 1. training-poc merge |
| 22:10 | 2. audit-backtest merge |
| 22:15 | 3. sprint1 merge + 5 項目テスト |
| 22:20 | 4. sprint2 (一部) merge or cherry-pick |
| 22:30 | 5. two-stage merge + import 確認 |
| 22:40 | 6. sprint6-kka merge (race_id 調整 確認) |
| 22:50 | 7. video-poc merge |
| 22:55 | 全構文チェック + V15 load + 5 項目テスト |
| 23:00 | git push origin main + Discord 通知 |
| 23:05 | 完了確認、 5/16 06:00 起床まで休止 |

所要 約 1 h、 余裕 30 min。

---

## 9. push 手順

```powershell
# pull --rebase で並行 Session 干渉回避
git pull --rebase origin main
# conflict あれば手動解決
git push origin main
```

⚠ `--force` は使わない (並行 Session 干渉防止)。
push 失敗時は pull --rebase してから再 push。

---

## 10. 想定リスク

| リスク | 確率 | 対策 |
|--------|------|------|
| predict_core.py 巨大 conflict | 低 | 新 module 設計、 V15 logic 末尾追記 |
| schtasks Keiba 系 影響 | 極低 | merge 対象は code 変更のみ、 schtasks XML は触らない |
| V15 model 動作異常 | 極低 | model file は merge 対象外 (バイナリ) |
| 並行 Session 干渉 | 中 | pull --rebase 必須、 force push 禁止 |
| 大量 untracked file 巻添え | 中 | git status で確認、 add は明示的 path のみ |

---

**Session #74 merge plan 完。 5/15 22:00 実行 ready。**
