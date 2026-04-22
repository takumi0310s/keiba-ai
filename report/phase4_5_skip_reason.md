# フェーズ5 (v16学習) スキップ理由

## 概要
2026-04-22 → 04-23 セッションで v16 学習を **スキップ** することを決定。

## スキップ理由

### 1. v16 閾値未達 (主因)
- training_eval: 100% ✅ (clear)
- **master_index: 2020-2022 が 0%** ❌ (閾値 30% 未達)
- バックグラウンドスクレイピング 4/18 02:04 を最後に停止 → 5日間進捗ゼロ
- v16 学習に必要なデータが揃っていない

### 2. CatBoost WF 検証スクリプトのバグ未修正
- `train/train_v135b_intra_ensemble.py` で `race_id_unique` KeyError
- v15 4-model アンサンブル学習自体が失敗していた (4/14)
- v16 (LGB+XGB+CB+FT+IR の 5-model) 構築前に修正必須

### 3. 時間制約超過
- セッション開始: 22:30 (4/22)
- 想定リミット: 23:30 (調査完了) → 02:00 (学習完了)
- 実時刻: 既に 03:25 を超過 → AM2:55 Pre-Fire-Check と競合

## 翌日以降の再開プラン

### Phase A (Mon 4/27)
1. SCRAPER-GUARD 解除 (06:00) 後にバックグラウンドスクレイピング再起動:
   ```bash
   nohup python tools/scrape_missing_all.py --years 2020,2021,2022,2023 \
     > logs/scrape_missing_all_restart3.log 2>&1 &
   ```
2. CatBoost `race_id_unique` KeyError の修正 (`train/train_v15_master.py` 周辺)
3. 修正後の CatBoost WF 5-model 単体テスト

### Phase B (5/2 以降)
4. coverage_report で v16 閾値突破確認
5. v16 学習計画書 `report/v16_training_plan_YYYYMMDD.md` 作成
6. 採用判定 (WF AUC > 0.8858 かつ gap < 0.05) を埋め込んだ retrain_v16.py 作成
7. 学習実行 → 採用なら本番モデル差し替え

## 影響評価

| 影響対象 | 評価 |
|----------|------|
| 来週末 4/25-26 本番運用 | **影響ゼロ** (v15 で継続運用、性能十分) |
| 自動運用 7 タスク | **影響ゼロ** (Pre/AM3-AM8/Morning/NightlySanity 全 Ready) |
| Cookie / Scheduler / Tests | **影響ゼロ** (全 OK 確認済み) |

→ スキップは安全な選択。AM3:00 発火直前まで作業せず就寝可能。
