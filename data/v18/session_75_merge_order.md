# Session #75: Merge 順序設計 (5/15 実行)

**実施日**: 2026-05-09
**実行予定**: 5/15 (V15 投資保護下、 段階 merge)

## Dependency 分析

| Branch | 依存 | 内容 |
|---|---|---|
| dev/sprint1 | (none) | 軽量、 独立、 全 tools 新規 |
| dev/sprint6-kka | (none) | KKA parser、 独立 |
| dev/video-poc | (none) | video pipeline、 独立、 Phase 4 用 |
| dev/sprint2 | (LEAK 検証) | post_race_features_update.py の LEAK 検証必須 |
| dev/audit-backtest | sprint4 (素材共有 5 files、 全 SAME hash) | 三連複 7 vs 11 / Session #69+#70 |
| dev/two-stage | training-poc (12 files SAME) | Stage 2 framework (#65 + #68 + #72) |
| dev/training-poc | (巨大 38 commits) | session #47-#67 累積、 horse motion / video / 5system 全部 |

## 推奨 merge 順序

### Phase 1: 軽量 + 独立 branch (5/15 09:00-10:00)

| # | Branch | ahead | 理由 |
|---|---|---|---|
| 1 | **dev/sprint1** | 6 | 軽量、 全 tools 新規、 conflict なし |
| 2 | **dev/sprint6-kka** | 5 | KKA parser 独立、 race_id format は KKA 内部のみ |
| 3 | **dev/video-poc** | 1 | video pipeline 独立、 Phase 4 で即活用 |

→ 3 branch merge 後、 main +12 commits

### Phase 2: 検証必要 (5/15 10:00-11:30)

| # | Branch | ahead | 検証項目 |
|---|---|---|---|
| 4 | **dev/sprint2** | 9 | post_race_features_update.py が当日成績 leak していないか確認後 merge |

→ 検証 PASS で 4 branch 合計 +21 commits

### Phase 3: backtest + Stage 2 (5/15 11:30-13:00)

| # | Branch | ahead | 理由 |
|---|---|---|---|
| 5 | **dev/audit-backtest** | 17 | sprint4 の 5 file (SAME hash) 含む、 LEAK 防止 verification 内蔵 |
| 6 | **dev/two-stage** | 19 | Stage 2 (#65+#68+#72)、 training-poc と 12 file SAME hash |

→ 6 branch 合計 +57 commits

### Phase 4: 巨大累積 最後 (5/15 13:00-14:00)

| # | Branch | ahead | 理由 |
|---|---|---|---|
| 7 | **dev/training-poc** | 38 | session #47-#67 巨大累積、 他 branch と SAME hash 多数、 最後に merge |

→ 7 branch 合計 +95 commits、 main +95 commits

## Pair conflict 検証 (file overlap = SAME hash 確認済)

| Pair | 重複 file 数 | 内容 hash | 結果 |
|---|---|---|---|
| audit-backtest <-> sprint4 | 5 | SAME | sprint4 archive 後、 audit-backtest が引き継ぐ |
| training-poc <-> two-stage | 12 | SAME | merge 順序問題なし、 同一 commit 由来 |
| sprint6-kka <-> training-poc | 2 | SAME | 同一 commit、 conflict なし |
| sprint6-kka <-> v20-interaction | 1 | (不要、 v20-interaction archive) | - |
| v20-ensemble <-> v20-interaction | 14 | SAME | 両方 V20 構築素材、 archive 1 keep 1 |
| v20-ensemble <-> v20-expanding | 5 | SAME | v20-expanding archive |
| v20-expanding <-> v20-interaction | 5 | SAME | 両方 archive |
| training-poc <-> v20-interaction | 1 | (不要、 v20-interaction archive) | - |

→ **全 overlap が SAME hash**。 Git merge は同一 content を自動統合 → conflict なし

## Merge コマンド (5/15 当日 用、 caveman 形式)

```bash
git checkout main
git pull --rebase

# Phase 1
git merge --no-ff dev/sprint1 -m "merge: dev/sprint1 (Session #75 step 1)"
git merge --no-ff dev/sprint6-kka -m "merge: dev/sprint6-kka (Session #75 step 2)"
git merge --no-ff dev/video-poc -m "merge: dev/video-poc (Session #75 step 3)"

# Phase 2 (LEAK 検証 PASS 後)
python tools/post_race_features_leak_verification.py  # ★ 事前必須 ★
git merge --no-ff dev/sprint2 -m "merge: dev/sprint2 (Session #75 step 4)"

# Phase 3
git merge --no-ff dev/audit-backtest -m "merge: dev/audit-backtest (Session #75 step 5)"
git merge --no-ff dev/two-stage -m "merge: dev/two-stage (Session #75 step 6)"

# Phase 4
git merge --no-ff dev/training-poc -m "merge: dev/training-poc (Session #75 step 7)"

# 各 phase 後 必ず
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"
python tests/test_features.py

git push origin main
```

## ロールバック計画

- 各 phase の commit を tag 化 (Session #75 step N)
- 異常時は `git reset --hard <prev tag>` で巻き戻し
- predict_core / daily_predict / app.py は **全 branch で未変更** → 本番影響ゼロ前提
