# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

競馬AI予測システム（中央競馬専用）
- Streamlit: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- GitHub: https://github.com/takumi0310s/keiba-ai

## 2段階モデル構成

### 設計思想
- **学習・バックテスト評価: Pattern A（リークフリー厳守）** → モデルの真の実力を評価
- **実運用の予測: Pattern B（当日情報込み）** → 使える情報は全て使って最高精度で予測

### Pattern A（評価用）
- `keiba_model_v9_central.pkl` - V9.3リークフリー
- AUC 0.8095（LGB+XGB ensemble）
- 67特徴量、680,381行の学習データ
- **当日オッズ(odds_log)、当日馬体重(horse_weight)、condition_enc + 派生特徴量を除外**

### Pattern B（実運用）
- `keiba_model_v9_central_live.pkl` - V9.3当日情報込み（83特徴量）
- Pattern A + 当日特徴量:
  - netkeibaから: odds_log, horse_weight, condition_enc, weight_change, pop_rank
  - JRA公式から: cushion_value（クッション値）, moisture_rate（含水率）
  - 気象庁APIから: temperature, humidity, wind_speed, precipitation, weather_enc
- AUCはPattern Aで評価（Pattern BのAUCは参考値: 0.8460）
- app.pyはPattern Bを優先使用（なければPattern Aにフォールバック）
- 馬場/天候データ取得失敗時はフォールバック（0=欠損として予測）

## アーキテクチャ

### app.py（~4360行）- Streamlitメインアプリ
- モデルロード: `load_v9_models()` → Pattern B優先、A→V8フォールバック
- 予測フロー: URL→出馬表→JRA馬場情報→気象庁天候→Pattern B予測→条件判定→買い目
- 当日情報: 馬体重・オッズ・馬場状態・天候・クッション値・含水率を自動取得
- 警告: 馬体重急変(±10kg)、混戦オッズを自動検知・表示
- 条件分類: `classify_race_condition()` → A-E,X条件でtrio 7点買い
- 記録: SQLiteに予測結果を保存、週次ROIレポート生成

### モデル構成
- `keiba_model_v9_central_live.pkl` - 実運用（Pattern B, 当日情報込み）
- `keiba_model_v9_central.pkl` - 評価用（Pattern A, リークフリー）
- `keiba_model_v8.pkl` - フォールバック用ベースライン

### 実運用テストツール
- `predict_and_log.py` - CLI予測ログ記録
- `check_results.py` - 結果照合・ROI計算
- `verify_real_roi.py` - 実配当ROI検証（netkeiba scraping）
- `monte_carlo_sim.py` - モンテカルロ破産確率シミュレーション

### 学習スクリプト
- `train/train_v92_central.py` - 中央V9.2学習（基盤関数群）
- `train/train_v93_leakfree.py` - Pattern A学習（リークフリー/評価用）
- `train/train_v93_pattern_b.py` - Pattern B学習（当日情報込み/実運用）
- `train/train_v10_ensemble.py` - LGB+XGB+MLP 3モデルアンサンブル（参考）

### データ取得・分析ツール
- `tools/extract_jvdata.py` - TARGET JV (C:\TFJV) → 7CSV抽出
- `scrape_jra_track.py` - JRA公式からクッション値・含水率取得
- `scrape_weather.py` - 気象庁APIから気温・湿度・風速・降水量取得
- `scrape_jra_payouts.py` - JRA公式DBから配当データ取得
- `calc_actual_roi.py` - 実配当ROI計算（JRA配当×WFバックテスト）

### アーカイブ
- `archive/nar/` - 地方(NAR)関連ファイル一式（モデル・学習・バックテスト）

## データ資産

- 中央: jra_races_full.csv(781,161行), training_times.csv(955,580行), odds_history.csv(778,387行), blood_full.csv(81,986行)
- 配当: jra_payouts.csv(27,541件, 2018-2025, JRA公式DB)
- TARGETデータ: C:\TFJV（SE_DATA/CK_DATA/HY_DATA/BR_DATA/KT_DATA）

## 条件定義（実配当ROI確認済み）

WF 2020-2025, 20,579レース, Pattern A (AUC 0.8017), JRA公式配当データ

| 条件 | 条件内容 | 買い目 | 実ROI | 推定ROI | 的中率 | N | 推奨 |
|------|----------|--------|-------|---------|--------|------|------|
| A | 8-14頭/1600m+/良〜稍重 | trio 7点 | **205.3%** | 439.6% | 44.5% | 6,438 | ○ |
| B | 8-14頭/1600m+/重〜不良 | trio 7点 | **236.9%** | 445.1% | 45.2% | 847 | ○ |
| C | 15頭+/1600m+/良〜稍重 | trio 7点 | **285.6%** | 538.8% | 33.7% | 4,774 | ○ |
| D | 1400m以下 | trio 7点 | **136.0%** | 236.0% | 27.0% | 7,254 | ○ |
| E | 7頭以下 | umaren 2点 | **118.0%** | 145.2% | 53.4% | 461 | ○ |
| X | 15頭+/重〜不良 | trio 7点 | **330.5%** | 544.2% | 35.5% | 805 | ○ |

- 全条件ROI 100%超え
- A,B,C,D,X: trio（三連複7点）、E: umaren（馬連2点）
- 推定ROIは実ROIの約2倍（推定式 o1*o2*o3*20 が過大評価）
- 詳細: data/actual_roi_results.json

## モンテカルロシミュレーション結果

- 初期資金1万円: 破産確率0.58%, 利益確率99.4%, 期待ROI 15,497%
- 初期資金3万円: 破産確率0.0%, 利益確率100%, 期待ROI 5,239%
- 初期資金10万円: 破産確率0.0%, 利益確率100%, 期待ROI 1,642%
- 詳細: data/monte_carlo_results.json

## コマンド

```bash
# 構文チェック（app.py変更時は必須）
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"

# テスト実行
python tests/test_features.py          # 5項目自動テスト
python tests/debug_all.py              # 25項目デバッグテスト

# 実運用テスト
python predict_and_log.py "URL"        # 予測→ログ記録
python check_results.py                # 結果照合
python check_results.py --summary      # 成績サマリー
python verify_real_roi.py              # 実配当ROI検証

# モンテカルロシミュレーション
python monte_carlo_sim.py              # 破産確率シミュレーション
python monte_carlo_sim.py --trials 50000  # 試行回数指定

# モデル学習
python train/train_v93_leakfree.py     # Pattern A学習（評価用）
python train/train_v93_pattern_b.py    # Pattern B学習（実運用）

# バックテスト
python backtest_central_leakfree.py    # 中央リークフリーBT

# データ取得
python tools/extract_jvdata.py         # TARGET JV → CSV抽出

# Streamlitローカル起動
streamlit run app.py
```

## 現行モデルのベースライン（これを下回る変更は一切採用しない）

- **Pattern A AUC: 0.8095**（time-split validation, LGB+XGB ensemble）
- **WF AUC: 0.8017**（walk-forward 2020-2025平均, LGB単体）
- **年別WF AUC**: 2020=0.7951, 2021=0.7997, 2022=0.8024, 2023=0.8012, 2024=0.8071, 2025=0.8048
- **実配当ROI**: A=205%(trio), B=237%(trio), C=286%(trio), D=136%(trio), E=118%(umaren), X=331%(trio)（全条件100%超え）
- **LGBパラメータ**: num_leaves=63, lr=0.05, feature_fraction=0.8, bagging_fraction=0.8, min_child_samples=50

## 重要ルール

1. **学習はPattern A、予測はPattern B**: バックテスト評価は常にPattern A（リークフリー）。実運用予測はPattern B（当日情報込み）
2. バックテストは必ずウォークフォワード（時系列分割）で実施
3. **改善が確認できない変更は採用しない**: WF AUC > 0.8017 かつ全年AUC > 0.78 かつ 実ROI全条件100%超え
4. app.pyを変更したら必ずpython構文チェックしてからcommit
5. 大きなデータファイル(.csv)は.gitignoreで除外、ローカル保持
6. モデル更新時はAUCが既存モデルを上回る場合のみ本番反映
7. 買い目の馬番は昇順ソート・カンマスペース区切りで表示
8. 中央競馬専用 — 地方(NAR)コードはarchive/nar/に保管

## 過去の失敗から学んだ教訓（リーク・過学習）

- **odds_log**: 確定オッズでありリーク。絶対に特徴量に使わない
- **horse_weight, condition_enc**: 当日情報。Pattern Aでは使わない
- **推定ROI**: 実配当ROIの約2倍に過大評価（実ROIで判断すること）
- **LGB+XGB+MLPアンサンブル**: 単体LGBに勝てなかった（V10: WF 0.8050 < 0.8083）
- **コース別専用モデル**: 汎用モデルに勝てなかった（過学習リスク大）
- **坂路調教**: マッチ率27%でAUC改善なし

## 定期タスク（Windows タスクスケジューラ）

```bash
# 毎朝8:00 - 当日全レース予測
python tools/daily_predict.py              # 当日予測
python tools/daily_predict.py --date 20260308  # 指定日予測

# 毎晩20:00 - 結果照合・ROI計算
python tools/daily_results.py              # 当日結果
python tools/daily_results.py --date 20260308  # 指定日結果

# 毎週月曜9:00 - 週次レポート
python tools/weekly_report.py              # 先週レポート
```

バッチファイル: `daily_predict.bat`, `daily_results.bat`, `weekly_report.bat`
ログ: `logs/` ディレクトリ

## Compaction対応

Claude Codeのコンテキスト圧縮時に失われやすい重要情報はこのCLAUDE.mdに集約。
特に「現行モデルのベースライン」と「過去の失敗から学んだ教訓」は常に参照すること。

## 未解決の課題

- LINE通知未実装
- GitHub Actionsによる自動化未実装（現在はWindows タスクスケジューラで代替）
