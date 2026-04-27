# v16 開発 TODO リスト (4/27 月曜から)

## ✅ 4/26 (土) 完了済み

### 即実装パッチ (4/27 night 適用)
- [x] **戦略⑦自動化**: `race_auto_notify.py` に4種類のフィルタ追加
- [x] **course_renovated 永久化**: `predict_core.py` line 2065-2073 修正
- [x] **predict_core.py FutureWarning 修正**: 主要15箇所を `0` → `0.0`
- [x] **1レース再予測ツール**: `tools/predict_one_race.py` 本番化
- [x] **CSV重複削除**: `cumulative_results.csv` 1件削除
- [x] **CLAUDE.md 更新**: v16 セクション追加

### 動作確認済み
- [x] `predict_one_race.py` で東京11Rフローラ S 予測成功
- [x] `race_auto_notify.py` の構文チェック OK
- [x] `predict_core.py` の構文チェック OK

## 🔴 4/27 (月) - 朝一でやる

### 1. GitHub commit & push
```bash
git add tools/race_auto_notify.py tools/predict_core.py
git add tools/predict_one_race.py train/features_v15_new.py
git add data/cumulative_results.csv CLAUDE.md V16_TODO_NEXT_WEEK.md
git commit -m "v16 prep: strategy7 filter, course_renovated permafrost, FutureWarning fix"
git push origin main
```

### 2. ノートPCでの git pull 確認
```bash
# ノートPC側で
cd ~/keiba-ai
git pull origin main
python -c "import ast; ast.parse(open('tools/race_auto_notify.py').read())"
```

### 3. Discord通知 のテスト
- 既存の dryrun スクリプトで動作確認

## 🟡 4/28-5/2 (火-金) - 1週間以内

### 4. jrdb_paci 取得経路の修復
- T58 の調査結果を踏まえた修復
- gaisha_rank データの最新化 (4-5月分)
- もしくは v16 で gaisha_rank 削除なので不要かも

### 5. cumulative_results.csv の top1_num/score 書き込み修正
- `tools/daily_results.py` の修正
- 今後の特徴量分析の自動化のため必須

### 6. predict_and_log.py を v15 対応化
- `keiba_model_v9_central.pkl` → `keiba_model_v15_central_live.pkl.gz`
- 緊急時の予測手段確保

### 7. 戦略⑦ 5/3-5/4 GW京都開催での運用テスト
- パッチ適用済 → 本番投入
- ROI実績の収集

## 🟢 GW (5/3-5/6) - v16 訓練準備

### 8. v16 訓練データ作成
- 2026/1-4月のデータ追加
  - JRA レースデータ (取得スクリプト必要?)
  - JRDB 全種類 (現状2025年まで)
- course_renovated 永久化の effect 確認
- prev_race_pace_diff 削除
- gaisha_rank 削除
- 期待特徴量数: 148

### 9. v16 学習実行
- `train/retrain_v16.py` を改造して v16 化
- WF AUC 目標: 0.895+ (v15: 0.8939)
- Ryzen 7 + 32GB + 16GB GPU
- 想定時間: 2-3時間

## 🟢 5/11以降 - 検証期

### 10. v15 vs v16 A/B比較
- 1週間並行運用
- ROI比較

### 11. GW京都データを反映
- 5/3-5/4, 5/10-5/11 京都4日開催
- 約88R追加で京都計110R
- 京都ROI再評価

### 12. 5/末 v16 本番切替判断
- 全条件クリアで v16 本番投入

---

## 📊 期待される最終成果

| 項目 | v15 現状 | v16 (目標) |
|------|---------|-----------|
| AUC | 0.8939 | 0.895+ |
| 特徴量数 | 150 (実効145) | 148 (実効148) |
| 戦略⑦ ROI | - | 140% (適用後想定) |
| 全体ROI | 119% | 150%+ (戦略⑦+v16) |
| 京都ROI | 0% | 80%+ |
| 1レース再予測 | なし | あり (動作確認済) |

---

## 🛡️ ロールバック手順 (もし問題が起きたら)

```bash
# race_auto_notify を元に戻す
cp tools/race_auto_notify.py.bak_20260427 tools/race_auto_notify.py

# predict_core を元に戻す
cp tools/predict_core.py.bak_20260427 tools/predict_core.py

# features_v15_new を元に戻す
cp train/features_v15_new.py.bak_20260427 train/features_v15_new.py

# cumulative_results を元に戻す
cp data/cumulative_results.csv.bak_20260427 data/cumulative_results.csv

# CLAUDE.md を元に戻す
cp CLAUDE.md.bak_20260427 CLAUDE.md
```

---

## 📋 修正履歴

### 2026-04-27 (4/26 深夜セッション)
- `tools/race_auto_notify.py`: 戦略⑦フィルタ追加
- `tools/predict_core.py`: course_renovated 永久化、FutureWarning修正
- `train/features_v15_new.py`: course_renovated 永久化 (auto-fail時は手動修正必要)
- `tools/predict_one_race.py`: 新規作成 (1レース再予測ツール)
- `data/cumulative_results.csv`: 重複1件削除
- `CLAUDE.md`: v16 セクション追加

