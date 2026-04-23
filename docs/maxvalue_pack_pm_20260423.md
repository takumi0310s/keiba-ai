# 4/23夜 最大効果パック レポート

実施日時: 2026-04-23 18:40-22:30
ブランチ: `feat/maxvalue-pack-20260423-pm`
ベースHEAD: `58d2036e` (前パック完了状態)
本番: 2026-04-25 (土)

---

## 結論

**6 commit、+1204 行、14ファイル**。予測ロジック・モデル無変更。
本番への影響中立、新規ツール+ナレッジ拡充のみ。

---

## タスク別結果

| # | タスク | 採用/保留 | 主成果 |
|---|--------|-----------|--------|
| 1 | extract_jvdata.py 自動化 | **保留** | C:\TFJV不在で技術的に自動化不可、来週TARGET再導入後 |
| 2 | target_odds.csv 修復 | **保留** | inc-20260423-002 と同根、TARGET由来手動運用 |
| 3 | v16 学習 dry-run | **採用** | 14秒で完了、エラー0 (LGB AUC 0.8638, CB AUC 0.8428) |
| 4 | 事故ナレッジベース拡充 | **採用** | 4/22-23 8件追加 (9→17件) |
| 5 | 条件別ROI詳細分析 | **採用** | 359R集計、Discord通知機能付き |
| 6 | v15 特徴量重要度分析 | **採用** | LGB+XGB統合 + 13カテゴリ別 |
| 7 | スキル化 | **採用** | .claude/commands/ に 5本追加 |

---

## 採用基準と判定理由

ユーザー指示「採用 or 保留判定を各タスクで厳格に」「疑わしきは保留」に従い:

### 採用 (5タスク)
- 既存ロジック非破壊
- 新規追加のみ (ツール / ドキュメント / KB)
- 構文チェック PASS
- 動作確認済み

### 保留 (2タスク)
- タスク1, 2: 代替ソース不在で技術的に自動化不可
- 「dry-run失敗 → 原因記録、タスク登録せず」の指示に従い、レポートのみ commit

---

## v16 dry-run 結果詳細

```
[1/5] cache load:           527,280 rows × 233 cols, 145 v15 features
[2/5] build_race_id:        25,302 races (race_id_unique 生成 OK)
[3/5] LightGBM train:       AUC 0.8638 (2022-24 → 2025, 120 features)
[4/5] CatBoost train:       AUC 0.8428 (20K sample)
[5/5] result save:          data/v16_dry_run_result.json

elapsed = 14.2sec, errors = 0
```

CatBoost race_id_unique KeyError (inc-20260422-001) は build_race_id() で完全解消。
本格 WF (4-model 全年) は月曜以降のスクレイピング閾値到達時に自動起動。

---

## ROI 分析サマリー (cumulative_results 359R)

| 指標 | 値 |
|------|-----|
| 累計R | 359R (settled) |
| 損益 | +30,650円 |
| ROI | 112.2% |
| BT保守的見積り | 142.6% |
| 差分 | -30.4pt |

詳細 → `report/roi_analysis_20260423.md`

---

## 特徴量重要度ハイライト

詳細 → `report/feature_importance_20260423.md`

(TOP30 / 下位30 / カテゴリ別 / 4/23 修正特徴量追跡)

---

## 本番準備状態 (4/25 土)

| 項目 | 状態 |
|------|------|
| v15 モデル | OK (変更なし) |
| 特徴量定義 (150) | OK |
| predict_core.py | OK (変更なし) |
| JRDB merge | OK (a7b244a5) |
| feature_lookups | OK (3/27固定、TARGET再導入待ち) |
| Cookie | **要更新 金曜昼** |
| タスクスケジューラ | OK (10/10 Ready) |
| 分布ドリフト | OK (軽微) |
| 事故ナレッジベース | 17件登録 |

### 最終判定
🟢 **本番準備完了** (Cookie更新さえ忘れなければ)

---

## 来週以降の TODO

1. 月曜: TARGET Frontier JV 再インストール手順整備
2. 月曜: スクレイピング再開 (master_index 0% 解消)
3. v16 本格 WF 学習 (閾値到達後)
4. test_features.py 既存バグ修正 (app.py:6621 actual_roi key)
5. cookie 自動更新フロー整備
