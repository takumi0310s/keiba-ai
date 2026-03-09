# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

競馬AI予測システム（中央・地方対応）
- Streamlit: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- GitHub: https://github.com/takumi0310s/keiba-ai

## 現在のモデル

- 中央V9.2: AUC 0.8083（リークフリー Pattern A）
- 地方NAR V2a: AUC 0.8243（リークフリー Pattern A）
- 全モデルはPattern A（厳密リークフリー）を厳守
- **当日オッズ(odds_log)、当日馬体重(horse_weight)、weight_changeは特徴量に使用禁止**

## アーキテクチャ

### app.py（~4100行）- Streamlitメインアプリ
- モデルロード: `load_model()`, `load_v9_models()`, `get_model_for_race(is_nar)`
- 条件分類: `classify_race_condition()` → A-E,X条件でbet種別を切り替え
- 予測: LightGBM + XGBoost ensemble、重み付き平均
- 買い目生成: `render_buy_section()` → 条件別にtrio/wide/umarenを自動選択
- リアルタイム: `fetch_realtime_odds()`, `fetch_lap_times()`, `fetch_training_data()`
- 記録: SQLiteに予測結果を保存、週次ROIレポート生成

### モデル構成
- `keiba_model_v9_central.pkl` - 中央本番（LGB+XGB, 67特徴量）
- `keiba_model_v9_nar.pkl` - 地方本番（LGB+XGB, 30特徴量）
- `keiba_model_v8.pkl` - フォールバック用ベースライン

### 学習スクリプト
- `train/train_v92_central.py` - 中央V9.2学習（jra_races_full.csv使用）
- `train/train_nar_v4.py` - 地方V4学習（nar_all_races.csv使用）
- `train/train_v10_ensemble.py` - LGB+XGB+MLP 3モデルアンサンブル

### データ取得ツール
- `tools/extract_jvdata.py` - TARGET JV (C:\TFJV) → 7CSV抽出
- `tools/scrape_nar_all.py` - netkeiba NAR全15場スクレイパー（requests+BS4, ~3000R/h）

## データ資産

- 中央: jra_races_full.csv(781,161行), training_times.csv(955,580行), odds_history.csv(778,387行), blood_full.csv(81,986行)
- 地方: chihou_races_full.csv(17,071行/KDSCOPE/南関4場/730-1500m), chihou_races_2020_2025.csv(1,803行/netkeiba/1600mのみ)
- 地方追加: nar_all_races.csv(49,915行/2025年/全15場) ※2015-2024はIPブロックで未取得
- TARGETデータ: C:\TFJV（SE_DATA/CK_DATA/HY_DATA/BR_DATA/KT_DATA）
- KDSCOPEデータ: C:\KDSCOPE\Data（NS.DAT/NR.DAT/O1等）

## 中央条件定義（リークフリーWFバックテスト確認済み）

- A: 8-14頭/1600m+/良〜稍重 → trio 7点 ROI 420.7%
- B: 8-14頭/1600m+/重〜不良 → trio 7点 ROI 473.8%
- C: 15頭+/1600m+/良〜稍重 → trio 7点 ROI 498.6%
- D: 1400m以下 → trio 7点 ROI 247.0%
- E: 7頭以下 → trio 7点 ROI 330.4%
- X: 15頭+/重〜不良 → trio 7点 ROI 598.2%

※ROIはオッズ推定値。条件間の相対比較は有効

## 地方条件定義

- A: 1600m+/8-14頭/良〜稍重 → trio ROI 366%
- B: 1600m+/8-14頭/重〜不良 → trio ROI 432%
- D: ~1400m/1-4頭 → wide（ROI未検証）
- E: 1600m+/7頭以下 → umaren ROI 350%
- F: ~1400m/5-7頭 → wide（ROI未検証）
- C: 1600m+/15頭+ → N不足
- G: ~1400m/8頭+ → データなし

※KDSCOPE(2009-2020)は730-1500mのみ。1600m+はnetkeiba1,803件のみ

## コマンド

```bash
# 構文チェック（app.py変更時は必須）
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"

# テスト実行
python tests/test_features.py          # 5項目自動テスト
python tests/debug_all.py              # 25項目デバッグテスト

# プロジェクトステータス
python project_status.py               # 全体確認
python project_status.py --section model  # モデル精度のみ
python project_status.py --json        # JSON出力

# バックテスト
python backtest_central_leakfree.py    # 中央リークフリーBT
python backtest_nar_leakfree.py        # 地方リークフリーBT

# データ取得
python tools/extract_jvdata.py         # TARGET JV → CSV抽出
python tools/scrape_nar_all.py         # NAR全場スクレイピング（再開可能）

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

## 未解決の課題

- 地方D/F/G条件のROI未検証（配当データ不足）
- netkeibaスクレイピング2015-2024年がIPブロックで未取得
- 実運用テスト（実レースでの予測→結果照合）未実施
- LINE通知未実装
- GitHub Actionsによる自動化未実装

## 自動化計画

- GitHub Actions: 毎朝出馬表取得→予測→通知
- CI/CD: push時に自動テスト（リークチェック・AUC検証・構文チェック）
- モデル監視: 週次精度レポート自動生成
- Streamlit監視: エラー自動検知・修正
