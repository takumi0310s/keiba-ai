# keiba-ai

JRA (中央競馬) の予測 AI システム

## 🎯 現在のバージョン
- **v15** (2026/4/27 時点)
- AUC: 0.8939 (LightGBM Booster)
- 訓練データ: 2015/01 - 2025/12 (527,280行)
- 特徴量: 150 (実効145)

## 🛠 技術スタック
- LightGBM + XGBoost + FT-Transformer + IntraRace Attention (4モデルアンサンブル)
- Optuna ハイパーパラメータ最適化
- Walk-Forward (WF) 評価
- JRDB アドバンスコース データ
- netkeiba Super Premium データ

## 🚀 主要機能

### 自動運用
- DailyJrdbKyi: JRDB データ取得 (毎朝6:00)
- DailyPremiumScrape: Premium データ取得 (毎朝3:00)
- DailyPredict: 全レース予測 (毎朝8:00)
- RaceAutoNotify_Sat/Sun: 戦略⑦ 適用 + Discord通知 (土日 8:45)
- DailyResultsEvening: 結果集計 (毎晩20:00)
- KeibaAI_DriftDetector: モデルドリフト検出 (週次)

### 戦略⑦
本番運用中のフィルタ戦略。詳細は [docs/strategy7_specification.md](docs/strategy7_specification.md)

### 1レース再予測
取消発生時の緊急予測ツール:
```bash
python tools/predict_one_race.py 202605020211
```

### GW監視ダッシュボード
```bash
python tools/gw_monitor.py
```

## 📊 直近実績 (4/12-4/26)
- 4週間: 175R
- 戦略⑦適用後 ROI: 115.0%
- 戦略⑦の効果: +24pt 改善

## 🔮 v16 開発中
- 訓練データ拡張 (+2026年1-4月)
- prev_race_pace_diff 削除 (149特徴量)
- gaisha_rank 復活 (94.2%カバー)
- course_renovated 永久化
- 期待 AUC: 0.895+

## 📁 主要ディレクトリ
keiba-ai/
├── tools/                  # 本番運用スクリプト
│   ├── predict_core.py     # 予測コア
│   ├── race_auto_notify.py # Discord通知 + 戦略⑦
│   ├── predict_one_race.py # 1レース再予測
│   └── gw_monitor.py       # GW監視ダッシュボード
├── train/                  # 学習スクリプト
│   ├── train_v15_master.py # v15 学習
│   ├── retrain_v16.py      # v16 再学習
│   └── features_v16_premium.py # v16 新特徴量
├── data/                   # データ
│   ├── cumulative_results.csv   # 累積結果
│   ├── daily_predictions/  # 日別予測
│   ├── daily_results/      # 日別結果
│   ├── jrdb_.csv         # JRDB データ
│   └── netkeiba_.csv     # netkeiba データ
└── docs/                   # ドキュメント
├── strategy7_specification.md
└── 20260427_v16_prep_report.md

## 🛡 ライセンス
個人プロジェクト (private)

## 📝 履歴
- 2026/4/27: v16 学習開始 / GW準備完了
- 2026/4/8: v15 学習完了 (AUC 0.8939)
- 2026/2/15: v1 開発開始
