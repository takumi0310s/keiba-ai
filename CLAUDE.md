# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

競馬AI予測システム（中央競馬専用）
- Streamlit: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- GitHub: https://github.com/takumi0310s/keiba-ai

## 現在のモデル

- 中央V9.3: AUC 0.8095（LGB+XGB ensemble, リークフリー Pattern A）
- 67特徴量、680,381行の学習データ
- **当日オッズ(odds_log)、当日馬体重(horse_weight)、condition_enc + 派生特徴量は使用禁止**

## アーキテクチャ

### app.py（~4360行）- Streamlitメインアプリ
- モデルロード: `load_model()`, `load_v9_models()`, `get_model_for_race()`
- 条件分類: `classify_race_condition()` → A-E,X条件でtrio 7点買い
- 予測: LightGBM + XGBoost ensemble、重み付き平均
- 買い目生成: `render_buy_section()` → 全条件trio 7点（700円/レース）
- リアルタイム: `fetch_realtime_odds()`, `fetch_lap_times()`, `fetch_training_data()`
- 記録: SQLiteに予測結果を保存、週次ROIレポート生成
- 実運用ダッシュボード: 予測ログ・成績・モンテカルロ結果表示

### モデル構成
- `keiba_model_v9_central.pkl` - 中央本番（LGB+XGB, 67特徴量）
- `keiba_model_v8.pkl` - フォールバック用ベースライン

### 実運用テストツール
- `predict_and_log.py` - CLI予測ログ記録
- `check_results.py` - 結果照合・ROI計算
- `verify_real_roi.py` - 実配当ROI検証（netkeiba scraping）
- `monte_carlo_sim.py` - モンテカルロ破産確率シミュレーション

### 学習スクリプト
- `train/train_v92_central.py` - 中央V9.2学習
- `train/train_v93_leakfree.py` - 中央V9.3リークフリー学習（本番）
- `train/train_v10_ensemble.py` - LGB+XGB+MLP 3モデルアンサンブル（参考）

### データ取得ツール
- `tools/extract_jvdata.py` - TARGET JV (C:\TFJV) → 7CSV抽出

### アーカイブ
- `archive/nar/` - 地方(NAR)関連ファイル一式（モデル・学習・バックテスト）

## データ資産

- 中央: jra_races_full.csv(781,161行), training_times.csv(955,580行), odds_history.csv(778,387行), blood_full.csv(81,986行)
- TARGETデータ: C:\TFJV（SE_DATA/CK_DATA/HY_DATA/BR_DATA/KT_DATA）
- ※TARGETにはtrio/umaren/wide実配当データなし（単勝オッズのみ）

## 条件定義（リークフリーWFバックテスト確認済み）

- A: 8-14頭/1600m+/良〜稍重 → trio 7点 ROI 420.7%
- B: 8-14頭/1600m+/重〜不良 → trio 7点 ROI 473.8%
- C: 15頭+/1600m+/良〜稍重 → trio 7点 ROI 498.6%
- D: 1400m以下 → trio 7点 ROI 247.0%
- E: 7頭以下 → trio 7点 ROI 330.4%
- X: 15頭+/重〜不良 → trio 7点 ROI 598.2%

※ROIはオッズ推定値。条件間の相対比較は有効

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

# バックテスト
python backtest_central_leakfree.py    # 中央リークフリーBT

# データ取得
python tools/extract_jvdata.py         # TARGET JV → CSV抽出

# Streamlitローカル起動
streamlit run app.py
```

## 重要ルール

1. **データリーク厳禁**: 当日オッズ・確定馬体重・当日馬場状態をモデルの特徴量に使わない
2. バックテストは必ずウォークフォワード（時系列分割）で実施
3. app.pyを変更したら必ずpython構文チェックしてからcommit
4. 大きなデータファイル(.csv)は.gitignoreで除外、ローカル保持
5. モデル更新時はAUCが既存モデルを上回る場合のみ本番反映
6. 買い目の馬番は昇順ソート・カンマスペース区切りで表示
7. 中央競馬専用 — 地方(NAR)コードはarchive/nar/に保管

## 未解決の課題

- 実運用テスト未実施（predict_and_log.py → check_results.pyの実戦検証）
- 実配当ROI検証（verify_real_roi.py）のサンプル蓄積が必要
- LINE通知未実装
- GitHub Actionsによる自動化未実装

## 自動化計画

- GitHub Actions: 毎朝出馬表取得→予測→通知
- CI/CD: push時に自動テスト（リークチェック・AUC検証・構文チェック）
- モデル監視: 週次精度レポート自動生成
- Streamlit監視: エラー自動検知・修正
