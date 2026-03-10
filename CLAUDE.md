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

| 条件 | 条件内容 | trio実ROI | trio推定ROI | 的中率 | N | 推奨 |
|------|----------|-----------|-------------|--------|------|------|
| A | 8-14頭/1600m+/良〜稍重 | **190.3%** | 381.0% | 44.7% | 6,438 | ○ |
| B | 8-14頭/1600m+/重〜不良 | **240.7%** | 478.4% | 45.2% | 847 | ○ |
| C | 15頭+/1600m+/良〜稍重 | **284.4%** | 511.2% | 33.6% | 4,774 | ○ |
| D | 1400m以下 | **135.0%** | 230.8% | 27.3% | 7,254 | ○ |
| E | 7頭以下 | **104.5%** | 300.3% | 75.3% | 461 | ○ |
| X | 15頭+/重〜不良 | **300.5%** | 490.7% | 35.5% | 805 | ○ |

- 全条件ROI 100%超え（trio 7点推奨）
- 推定ROIは実ROIの約2倍（推定式 o1*o2*o3*20 が過大評価）
- 条件Eはumaren(119.1%)がtrio(104.5%)を上回る
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

## 重要ルール

1. **学習はPattern A、予測はPattern B**: バックテスト評価は常にPattern A（リークフリー）。実運用予測はPattern B（当日情報込み）
2. バックテストは必ずウォークフォワード（時系列分割）で実施
3. app.pyを変更したら必ずpython構文チェックしてからcommit
4. 大きなデータファイル(.csv)は.gitignoreで除外、ローカル保持
5. モデル更新時はAUCが既存モデルを上回る場合のみ本番反映
6. 買い目の馬番は昇順ソート・カンマスペース区切りで表示
7. 中央競馬専用 — 地方(NAR)コードはarchive/nar/に保管

## 未解決の課題

- 実運用テスト未実施（predict_and_log.py → check_results.pyの実戦検証）
- LINE通知未実装
- GitHub Actionsによる自動化未実装

## 自動化計画

- GitHub Actions: 毎朝出馬表取得→予測→通知
- CI/CD: push時に自動テスト（リークチェック・AUC検証・構文チェック）
- モデル監視: 週次精度レポート自動生成
- Streamlit監視: エラー自動検知・修正
