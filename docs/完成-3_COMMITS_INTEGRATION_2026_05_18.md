# 完成-3: B-2 / B-3 / B-4 / B-5 commit + 累積整理 (2026-05-18)

## 1. uncommitted file list before (task 開始時)

work tree に存在 + git index 未追加 の主要 file:

- `CLAUDE.md` (modified、 drift 30 件 verify 反映)
- `docs/A_AUDIT_36_COMMITS_2026_05_18.md` (new)
- `docs/B1_NONE_JSON_BUG_FIX_2026_05_18.md` (new)
- `docs/B2_CLAUDE_MD_FULL_DRIFT_RESOLUTION_2026_05_18.md` (new)
- `docs/B3_BASELINE_UPDATE_5_18_2026.md` (new)
- `docs/B4_V15_FULL_VERDICT_2026_05_18.md` (new)
- `docs/B5_SCHTASK_DRY_RUN_VERIFY_2026_05_22.md` (new)
- `data/task_outcomes/baseline_v15.json` (modified、 95.67% 反映)

既 commit 済 (重複しない):
- `models/v15_full_candidate.pkl.gz` (3.3 MB) + `tools/train_v15_full.py` + `docs/V15_FULL_TRAINING_LOG_2026_05_18.md` → ★ 4db6cc44 で commit 済 ★

## 2. 各 sub-task deliverable verify

| sub-task | deliverable | status |
|----------|-------------|--------|
| B-1 | docs/B1_NONE_JSON_BUG_FIX_2026_05_18.md | exists (4,405 B) |
| B-2 | docs/B2_CLAUDE_MD_FULL_DRIFT_RESOLUTION_2026_05_18.md + CLAUDE.md 更新 | exists (6,048 B + diff) |
| B-3 | data/task_outcomes/baseline_v15.json + docs/B3_BASELINE_UPDATE_5_18_2026.md | exists (10,029 B + 4,218 B) |
| B-4 | models/v15_full_candidate.pkl.gz + train_v15_full.py + V15_FULL_TRAINING_LOG + B4_VERDICT | partial (3 file 既 commit 4db6cc44、 verdict doc 別) |
| B-5 | docs/B5_SCHTASK_DRY_RUN_VERIFY_2026_05_22.md | exists (13,263 B) |
| A | docs/A_AUDIT_36_COMMITS_2026_05_18.md | exists (7,717 B) |

## 3. 各 commit detail

### commit 1: 1ab658aa [B-2 + A audit]
```
[B-2 + A audit] CLAUDE.md drift 30 件全 verify + 36 commits audit
3 files changed, 307 insertions(+), 3 deletions(-)
- CLAUDE.md (drift 訂正反映)
- docs/A_AUDIT_36_COMMITS_2026_05_18.md (new)
- docs/B2_CLAUDE_MD_FULL_DRIFT_RESOLUTION_2026_05_18.md (new)
```

### commit 2: 29578209 [B-3]
```
[B-3] baseline 95.67% / -19,080 / n=629 update
2 files changed, 254 insertions(+), 34 deletions(-)
- data/task_outcomes/baseline_v15.json (updated)
- docs/B3_BASELINE_UPDATE_5_18_2026.md (new)
```

### commit 3: dc76aacd [B-1]
```
[B-1] None/JSON bug fix 完了 verify doc
1 file changed, 125 insertions(+)
- docs/B1_NONE_JSON_BUG_FIX_2026_05_18.md (new)
```

### commit 4: 06db0552 [B-4]
```
[B-4] v15_full GO verdict doc (累積 commit)
1 file changed, 202 insertions(+)
- docs/B4_V15_FULL_VERDICT_2026_05_18.md (new)
※ v15_full .pkl.gz / train script / training log は 4db6cc44 で既 commit 済
```

### commit 5: 29eca312 [B-5]
```
[B-5] 9 schtask 全 未登録 honest 確定
1 file changed, 311 insertions(+)
- docs/B5_SCHTASK_DRY_RUN_VERIFY_2026_05_22.md (new)
```

## 4. 累積 commit 数 final

- 本 task で追加 commit: **5 件** (1ab658aa, 29578209, dc76aacd, 06db0552, 29eca312)
- v15_full B-4 主体は事前 commit (4db6cc44) で計上、 verdict doc が今回 +1
- 5/18 17:00 以降 commit 累計: 6 件 (4db6cc44 含む)

## 5. 大 file accidental add 防止結果

確認結果: ★ 大 file 0 件 staged / 0 件 commit ★

| file | size | status |
|------|------|--------|
| data/v20_training_data_full.csv | 114 MB | untracked のまま (OK) |
| data/_v15_optuna_df_cache.pkl.gz | 104 MB | untracked のまま (OK) |
| models/v15_full_candidate.pkl.gz | 3.3 MB | 既 commit (4db6cc44、 100 MB 未満 OK) |
| keiba_model_v22_4ensemble.pkl.gz | (modified) | 未 stage (OK、 別 task で扱う) |
| models/v15_2_candidate.pkl.gz | (modified) | 未 stage (OK) |

method: `git add` を file by file で実行、 `git add -A` / `git add .` は完全回避。

## 6. V15 production 不変保証

- predict_core.py / daily_predict.py / race_auto_notify.py / app.py: 触らず
- keiba_model_v15_central*.pkl.gz: 触らず
- 5/18 朝 影響 0%

## verdict

完成-3 完了、 B-2/B-3/B-4/B-5 5 commits、 累積 6 commits (4db6cc44 含む)、 大 file add なし
