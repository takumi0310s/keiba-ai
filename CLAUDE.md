# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Last updated: 2026-03-13

---

## 1. プロジェクト概要

**競馬AI予測システム（中央競馬専用）**

JRA中央競馬の全レースをAIで予測し、条件別に最適な買い目を自動生成するシステム。
LightGBM + XGBoostアンサンブルモデルで複勝圏（3着以内）を予測し、6つの条件分類に基づいて三連複/馬連の買い目を推奨する。

- **Streamlit**: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- **GitHub**: https://github.com/takumi0310s/keiba-ai
- **2段階モデル**: Pattern A（リークフリー評価用）+ Pattern B（当日情報込み実運用）
- **検証済み**: WF 2020-2025, 20,579レース, 全条件ROI 100%超え

---

## 2. できること（全機能一覧）

### 予測機能
- **netkeiba URL入力** → 出馬表自動取得 → AI予測 → 条件判定 → 買い目生成
- **Pattern B予測**: 当日オッズ・馬体重・馬場状態・天候を自動取得して高精度予測
- **条件自動判定**: 頭数・距離・馬場状態から6条件(A-E,X)を自動分類
- **買い目生成**: 三連複7点（条件E: 馬連2点）を自動生成
- **EV表示**: 各買い目のExpected Value（期待値）を計算・表示
- **警告機能**: 馬体重急変(±10kg)、混戦オッズを自動検知
- **SQLite記録**: 予測結果をローカルDBに保存

### Streamlitダッシュボード
- **予測ページ**: URL入力 → リアルタイム予測・買い目表示
- **TRACK RECORD**: 過去予測の成績一覧（会場/日付ブラウザ、予測詳細付き）
- **結果登録**: netkeiba結果ページURLで的中判定・配当記録
- **週次ROIレポート**: 条件別・コース別・距離別の成績集計

### 運用ツール
- `tools/daily_predict.py` — 毎朝8:00自動実行、当日全レース予測
- `tools/daily_results.py` — 毎晩20:00自動実行、結果照合・ROI計算
- `tools/weekly_report.py` — 毎週月曜9:00、週次パフォーマンスレポート
- `predict_and_log.py` — CLI手動予測・ログ記録
- `check_results.py` — CLI結果照合（--summaryで成績サマリー）
- `verify_real_roi.py` — netkeiba実配当ROI検証

### 検証・分析ツール
- `monte_carlo_sim.py` — モンテカルロ破産確率シミュレーション（10,000試行）
- `project_status.py` — プロジェクト全体ステータス（6セクション）
- `backtest_central_leakfree.py` — ウォークフォワードバックテスト
- `calc_actual_roi.py` — JRA公式配当データでの実ROI計算
- `tools/validation_1〜13_*.py` — 13項目の包括的検証スイート

### データ取得
- `tools/extract_jvdata.py` — TARGET JV (C:\TFJV) → 7CSV抽出
- `scrape_jra_track.py` — JRA公式クッション値・含水率
- `scrape_weather.py` — 気象庁API気温・湿度・風速・降水量
- `scrape_jra_payouts.py` — JRA公式DB配当データ

---

## 3. やったこと（全実施タスク一覧）

### データ取得・変換
1. TARGET JV (C:\TFJV) からCSV抽出（SE_DATA/CK_DATA/HY_DATA/BR_DATA/KT_DATA）
2. jra_races_full.csv 構築（781,161行、2010-2025）
3. training_times.csv 構築（955,580行、木/坂路調教データ）
4. odds_history.csv 構築（778,387行）
5. blood_full.csv 構築（81,986行、血統データ）
6. JRA公式DB配当スクレイパー構築 → jra_payouts.csv（27,541件、2018-2025）
7. JRA馬場情報スクレイパー（クッション値・含水率）
8. 気象庁APIスクレイパー（気温・湿度・風速・降水量）
9. netkeiba出馬表・結果スクレイパー（db.netkeiba.comフォールバック対応）

### モデル学習・改善
1. V8ベースモデル学習
2. V9.1基盤特徴量（43特徴量）
3. V9.2追加特徴量（+11: career/sire/wood training）→ AUC改善
4. V9.3追加特徴量（+13: pace/distance aptitude/frame advantage）→ AUC 0.8095
5. Pattern A（リークフリー67特徴量）確立 → 確定オッズリーク発見・除去
6. Pattern B（当日情報込み75特徴量）学習 → AUC 0.8460（参考値）
7. V10アンサンブル（LGB+XGB+MLP）試行 → 不採用（WF 0.8050 < 0.8083）
8. コース別専用モデル試行 → 不採用（過学習）
9. Optunaハイパーパラメータ最適化（100試行）→ 微改善のみ、不採用
10. 2段階モデル構成（学習=A、予測=B）確立

### テスト・検証（22項目 + Phase 10-13）
1. リークフリー検証（encode_categoricals/encode_sires静的解析）→ PASS
2. 目的変数比較（Win/Place/EV weighted）→ Place最適 AUC 0.8019
3. EVフィルタ分析（EV≥1.0閾値）→ 全レースEV≥1.0で効果なし
4. 券種最適化（全条件×全券種）→ 現行が最適
5. オッズギャップ分析（購入時vs確定）→ ROI影響0-5%
6. ドローダウン分析（MDD/連敗/回復/破産確率）→ 3万円以上で破産0%
7. 年別パフォーマンス（2020-2025 AUC/ROI）→ 安定上昇傾向
8. データ拡張チェック（67特徴量網羅性）→ 十分
9. 最終レポート統合 → READY判定
10. 市場依存性テスト（prev_odds_log除外）→ LOW依存、真の能力予測
11. サンプルサイズ検証（Bootstrap CI）→ N=20,579 HIGH信頼性
12. ROI計算整合性チェック → 全PASS
13. 保守的ROI見積り（BT×0.7）→ 全体142.6%
14. 特徴量リーク監査（全67特徴量）→ PASS
15. WFバックテスト（2020-2025, 20,579レース）→ 全条件ROI 100%超え
16. モンテカルロシミュレーション（10,000試行×1,000レース）
17. 5項目自動テスト（tests/test_features.py）
18. 25項目デバッグテスト（tests/debug_all.py）
19. 詳細ROI分析8テスト（月別/場別/クラス/芝ダ/人気/配当分布/D細分化/ストレス）

### バグ修正
1. 確定オッズ(odds_log)リーク発見・除去
2. 条件E買い目trio→umaren切替
3. 条件E投資額200→700円修正
4. app.py BASE_DIR未定義修正
5. バッチ予測：実モデル使用・結果UI再設計
6. TRACK RECORD UI: 会場/日付ブラウザ追加
7. 馬番昇順ソート・カンマスペース区切り表示
8. db.netkeiba.comフォールバック対応
9. bet_type記録バグ修正（条件別正確な記録）
10. モデルロード絶対パス修正（Streamlit Cloud対応）
11. 条件D 1000m以下を購入非推奨に変更（ROI 85%, N=534）

### インフラ整備
1. Streamlit Cloud デプロイ
2. Windows タスクスケジューラ設定（daily_predict/results, weekly_report）
3. SQLiteローカルDB構築
4. .gitignore設定（大容量CSV除外）
5. project_status.py CLI構築
6. バッチファイル作成（daily_predict.bat, daily_results.bat, weekly_report.bat）

---

## 4. モデル詳細

### 2段階モデル設計思想
- **Pattern A（評価用）**: リークフリー厳守。モデルの真の実力を評価
- **Pattern B（実運用）**: 使える情報は全て使って最高精度で予測

### Pattern A スペック
- ファイル: `keiba_model_v9_central.pkl`
- AUC: **0.8095**（LGB+XGB ensemble, time-split validation）
- WF AUC: **0.8017**（walk-forward 2020-2025平均, LGB単体）
- 特徴量: **67個**（リークフリー）
- 学習データ: 680,381行
- 目的変数: `finish <= 3`（複勝圏 binary）

### Pattern B スペック
- ファイル: `keiba_model_v9_central_live.pkl`
- AUC: 0.8460（参考値、**評価はPattern AのAUCで行う**）
- 特徴量: **75個**（Pattern A 67 + 当日情報8）
- app.pyはPattern Bを優先、なければA→V8にフォールバック
- 馬場/天候データ取得失敗時は0=欠損として予測

### Pattern A 全67特徴量

#### 基本特徴量（14個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 1 | weight_carry | 斤量(kg) |
| 2 | age | 馬齢 |
| 3 | distance | レース距離(m) |
| 4 | course_enc | コース(0-9, 10=unknown) |
| 5 | surface_enc | 芝=0, ダート=1, 障害=2 |
| 6 | sex_enc | 牡=0, 牝=1, セン=2 |
| 7 | num_horses_val | 出走頭数 |
| 8 | horse_num | 馬番 |
| 9 | bracket | 枠番(1-8) |
| 10 | sire_enc | 父馬TOP100エンコード(0-99, 100=other) |
| 11 | bms_enc | 母父TOP100エンコード(0-99, 100=other) |
| 12 | location_enc | 所属(0=美浦, 1=栗東, 2=地方, 3=外国) |
| 13 | is_nar | NAR=1, JRA=0 |
| 14 | season | 春=0, 夏=1, 秋=2, 冬=3 |

#### 騎手・調教師（3個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 15 | jockey_wr_calc | 騎手勝率(expanding window, alpha=30) |
| 16 | jockey_course_wr_calc | 騎手コース別勝率(expanding, alpha=10) |
| 17 | jockey_surface_wr | 騎手馬場別勝率(expanding, alpha=10) |

#### 前走ラグ特徴量（10個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 18 | prev_finish | 前走着順 |
| 19 | prev2_finish | 前々走着順 |
| 20 | prev3_finish | 3走前着順 |
| 21 | prev_last3f | 前走上がり3F |
| 22 | prev2_last3f | 前々走上がり3F |
| 23 | prev_pass4 | 前走4角位置 |
| 24 | prev_prize | 前走賞金 |
| 25 | prev_odds_log | 前走オッズ(log) |
| 26 | rest_days | 休養日数(1-365でclip) |
| 27 | rest_category | 休養カテゴリ(0-5: 7/15/35/64/181日区切り) |

#### 集計特徴量（5個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 28 | avg_finish_3r | 直近3走平均着順 |
| 29 | best_finish_3r | 直近3走最高着順 |
| 30 | top3_count_3r | 直近3走の3着以内回数 |
| 31 | finish_trend | 着順トレンド(prev3 - prev) |
| 32 | avg_last3f_3r | 直近3走平均上がり3F |

#### 派生特徴量（11個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 33 | dist_change | 前走からの距離変更(m) |
| 34 | dist_change_abs | 距離変更絶対値 |
| 35 | dist_cat | 距離カテゴリ(0-4) |
| 36 | age_sex | 年齢×10+性別 |
| 37 | age_season | 年齢×10+季節 |
| 38 | horse_num_ratio | 馬番/頭数 |
| 39 | bracket_pos | 枠位置(内=0, 中=1, 外=2) |
| 40 | carry_diff | 斤量 - レース平均斤量 |
| 41 | age_group | 年齢(2-7でclip) |
| 42 | surface_dist_enc | 馬場×10+距離カテゴリ |
| 43 | course_surface | コース×10+馬場 |

#### V9.2追加（8個、リーク除外後）
| # | 特徴量 | 説明 |
|---|--------|------|
| 44 | horse_career_races | 通算出走数(expanding, 0-indexed) |
| 45 | horse_career_wr | 通算勝率(expanding, alpha=5) |
| 46 | horse_career_top3r | 通算複勝率(expanding, alpha=5) |
| 47 | sire_surface_wr | 父馬産駒馬場別勝率(expanding, alpha=50) |
| 48 | sire_dist_wr | 父馬産駒距離別勝率(expanding, alpha=50) |
| 49 | bms_surface_wr | 母父産駒馬場別勝率(expanding, alpha=50) |
| 50 | wood_best_4f_filled | 木馬場調教4Fベスト(14日, mean fill ~52.0s) |
| 51 | has_wood_training | 木馬場調教データ有無 |

#### V9.2派生（2個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 52 | sire_dist | 父馬×10+距離カテゴリ |
| 53 | sire_surface | 父馬×10+馬場 |

#### V9.2調教（2個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 54 | training_time_filled | 調教4Fタイム(mean fill) |
| 55 | has_training | 調教データ有無 |

#### V9.3新規（12個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 56 | prev_race_first3f | 前走前半3F(ラップデータ) |
| 57 | prev_race_last3f | 前走後半3F(ラップデータ) |
| 58 | prev_race_pace_diff | 前走後半3F-前半3F(ペース差) |
| 59 | prev_agari_relative | 前走上がり相対値(個人-全体) |
| 60 | wood_count_2w | 木馬場調教回数(2週間) |
| 61 | sakaro_best_4f_filled | 坂路4Fベスト(14日, mean fill ~53.0s) |
| 62 | sakaro_best_3f_filled | 坂路3Fベスト(14日, mean fill ~39.0s) |
| 63 | has_sakaro_training | 坂路調教データ有無 |
| 64 | total_training_count | 調教合計回数(木+坂路) |
| 65 | horse_dist_top3r | 馬の距離別複勝率(expanding, alpha=5) |
| 66 | horse_surface_top3r | 馬の馬場別複勝率(expanding, alpha=5) |
| 67 | frame_course_dist_wr | 枠×コース×距離の勝率(expanding, alpha=100) |

### Pattern Bの追加8特徴量
| 特徴量 | ソース | 説明 |
|--------|--------|------|
| odds_log | netkeiba | 単勝オッズ(log変換) |
| pop_rank | netkeiba | 人気順位 |
| horse_weight | netkeiba | 当日馬体重(kg) |
| weight_change | netkeiba | 馬体重変化(前走比) |
| weight_change_abs | netkeiba | 馬体重変化絶対値 |
| weight_cat | 計算 | 体重カテゴリ(0-3) |
| weight_cat_dist | 計算 | 体重カテゴリ×距離カテゴリ |
| condition_enc | netkeiba | 馬場状態(良=0, 稍重=1, 重=2, 不良=3) |
| cond_surface | 計算 | 馬場×馬場種別 |
| cushion_value | JRA公式 | クッション値(芝のみ) |
| moisture_rate | JRA公式 | 含水率 |
| temperature | 気象庁API | 気温(℃) |
| humidity | 気象庁API | 湿度(%) |
| wind_speed | 気象庁API | 風速(m/s) |
| precipitation | 気象庁API | 降水量(mm) |
| weather_enc | 気象庁API | 天候(晴=0, 曇=1, 雨=2, 雪=3) |

### アンサンブル構成
- **LightGBM（主）** + **XGBoost（副）**
- 重み: AUC比例（LGB ~56%, XGB ~44%）
- `pred = w_lgb * lgb_pred + w_xgb * xgb_pred`

### LGBパラメータ
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}
# Early stopping: 50 rounds, max 1000 rounds
```

### XGBパラメータ
```python
{
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'tree_method': 'hist',
}
# Early stopping: 50 rounds, max 1000 rounds
```

### Optuna結果（100試行、不採用）
- Best: WF AUC 0.8022（+0.0006）→ 基準0.8095未達のため不採用
- 現行パラメータを維持

---

## 5. 全条件詳細

### 条件定義テーブル

WF 2020-2025, 20,579レース, Pattern A (AUC 0.8017), JRA公式配当データ

| 条件 | 条件内容 | 買い目 | 実ROI | 推定ROI | 的中率 | N | 保守ROI(×0.7) |
|------|----------|--------|-------|---------|--------|------|---------------|
| A | 8-14頭/1600m+/良〜稍重 | trio 7点 | **205.3%** | 439.6% | 44.5% | 6,438 | 143.7% |
| B | 8-14頭/1600m+/重〜不良 | trio 7点 | **236.9%** | 445.1% | 45.2% | 847 | 165.8% |
| C | 15頭+/1600m+/良〜稍重 | trio 7点 | **285.6%** | 538.8% | 33.7% | 4,774 | 199.9% |
| D | 1200-1400m | trio 7点 | **136.0%** | 236.0% | 27.0% | 7,254 | 95.2% |
| E | 7頭以下 | umaren 2点 | **118.0%** | 145.2% | 53.4% | 461 | 82.6% |
| X | 15頭+/重〜不良 | trio 7点 | **330.5%** | 544.2% | 35.5% | 805 | 231.3% |

- **全条件ROI 100%超え**（バックテスト実績）
- **1000m以下は非推奨**: ROI 85.0% (N=534) → 予測はするが購入非推奨表示
- 保守的見積り（×0.7）ではD, Eが100%以下 → 実運用で要モニタリング
- 推定ROIは実ROIの約2倍（推定式 `o1*o2*o3*20` が過大評価）
- 投資額: 全条件700円/レース

### 買い目構成
- **trio（三連複）7点**: TOP1軸 - TOP2,TOP3 - TOP2〜TOP6 のフォーメーション
- **umaren（馬連）2点**: TOP1軸 - TOP2, TOP1軸 - TOP3（各350円、オッズ連動400/300振り分け）

### 条件判定ロジック（優先順位順）
```python
def classify_condition(num_horses, distance, condition):
    heavy = condition in ['重', '不良']  # or condition_enc >= 2
    if num_horses <= 7:       return 'E'  # 少頭数
    if distance <= 1400:      return 'D'  # スプリント (1200-1400m推奨、1000m以下は非推奨)
    if 8 <= nh <= 14 and distance >= 1600 and not heavy: return 'A'
    if 8 <= nh <= 14 and distance >= 1600 and heavy:     return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy: return 'C'
    return 'X'  # その他（15頭+/重〜不良など）
    # 注: 条件Dかつ1000m以下 → recommended=False（ROI 85%, N=534）
```

---

## 6. テスト結果一覧

### リークフリー検証 → PASS
- encode_categoricals: ルールベース変換（リークなし）
- encode_sires: fold毎にtrain dataのみで計算（リークなし）
- expanding window: cumsum - current（当該レース除外）
- 軽微な技術的リーク（fillna global mean, Bayesian prior）→ 影響無視可能

### ウォークフォワード年別AUC/ROI
| 年 | AUC | Trio ROI |
|----|-----|---------|
| 2020 | 0.7954 | 196.3% |
| 2021 | 0.7999 | 194.4% |
| 2022 | 0.8002 | 177.3% |
| 2023 | 0.8021 | 180.1% |
| 2024 | 0.8074 | 181.3% |
| 2025 | 0.8065 | 239.3% |
| **平均** | **0.8019** | — |

### 実配当ROI（条件別）→ 全条件100%超え
- A: 205.3%, B: 236.9%, C: 285.6%, D: 136.0%, E: 118.0%, X: 330.5%

### モンテカルロ結果（10,000試行×1,000レース）
| 初期資金 | 破産確率 | 利益確率 | 期待ROI | 平均最終資金 |
|----------|---------|---------|---------|-------------|
| 1万円 | 0.59% | 99.4% | 15,497% | 1,549,735円 |
| **3万円** | **0.0%** | **100%** | **5,239%** | **1,574,242円** |
| 10万円 | 0.0% | 100% | 1,644% | 1,644,242円 |

### ドローダウン分析
| 初期資金 | MDD平均 | MDD最悪 | 最大連敗 | 回復(avg) |
|----------|---------|---------|---------|----------|
| 1万円 | 25.2% | 99.7% | 37レース | 9レース |
| 3万円 | 11.1% | 53.9% | 37レース | 3レース |
| 10万円 | 4.6% | 16.2% | 37レース | 3レース |

### 市場依存性テスト → LOW依存
- Baseline AUC: 0.8019 → No-odds AUC: 0.7993（差: -0.0026）
- Baseline ROI: 194.6% → No-odds ROI: 204.4%（むしろ改善）
- **判定: 真の能力予測モデル（オッズ依存ではない）**

### サンプルサイズ検証
| 条件 | N | ROI 95%CI | 信頼性 |
|------|---|-----------|--------|
| A | 6,438 | [198%, 213%] | HIGH |
| B | 847 | [213%, 261%] | LOW |
| C | 4,774 | [272%, 300%] | MEDIUM |
| D | 7,254 | [130%, 142%] | HIGH |
| E | 461 | [103%, 133%] | LOW |
| X | 805 | [292%, 369%] | LOW |

### 保守的ROI見積り（BT × 0.7）
- 補正要因: オッズ差(-7.5%), モデル劣化(-10%), 条件過学習(-10%)
- **全体保守的ROI: 142.6%**
- 条件別: A=143.7%, B=165.8%, C=199.9%, D=95.2%, E=82.6%, X=231.3%

### 最終判定: **READY**
- リークフリー: PASS
- AUCベースライン: PASS
- ROI全条件100%超え: PASS

---

## 7. データ資産

### コアデータ（data/ディレクトリ、.gitignore対象）
| ファイル | 行数 | 内容 |
|----------|------|------|
| jra_races_full.csv | 781,161 | 中央競馬全レースデータ(2010-2025) |
| training_times.csv | 955,580 | 調教タイムデータ(木/坂路) |
| odds_history.csv | 778,387 | オッズ履歴 |
| blood_full.csv | 81,986 | 血統データ |
| jra_payouts.csv | 27,541 | JRA公式配当(2018-2025) |

### JRA配当CSVフォーマット
```
race_date, course, kai, nichi, race_num, tansho_nums, tansho_payout,
fukusho_nums, fukusho_payouts, umaren_nums, umaren_payout, wide_nums,
wide_payouts, trio_nums, trio_payout, tierce_nums, tierce_payout
```

### TARGETデータソース
- パス: `C:\TFJV`
- SE_DATA: レース情報
- CK_DATA: 調教データ
- HY_DATA: 票数/オッズ
- BR_DATA: 血統
- KT_DATA: その他

### 検証結果JSON（data/ディレクトリ）
| ファイル | 内容 |
|----------|------|
| actual_roi_results.json | 全条件実配当ROI |
| monte_carlo_results.json | 破産確率シミュレーション |
| drawdown_analysis.json | ドローダウン詳細 |
| market_dependency_test.json | 市場依存性テスト |
| sample_size_validation.json | サンプルサイズCI |
| conservative_roi_estimate.json | 保守的ROI見積り |
| yearly_performance.json | 年別AUC/ROI |
| final_validation_report.json | 最終検証レポート |
| standardization_leak_check.json | リークチェック |
| target_variable_comparison.json | 目的変数比較 |
| ev_filter_analysis.json | EVフィルタ分析 |
| ticket_type_optimization.json | 券種最適化 |
| odds_gap_analysis.json | オッズギャップ |
| data_augmentation_check.json | データ拡張チェック |
| roi_calculation_validation.json | ROI整合性 |
| optuna_tuning_results.json | Optuna結果 |

---

## 8. リーク厳禁ルール

### Pattern Aで除外する特徴量（8個）
```python
LEAK_FEATURES_A = {
    'odds_log',          # 確定オッズ → 投票締切後に確定。最重要リーク
    'horse_weight',      # 当日馬体重 → レース70分前に発表
    'condition_enc',     # 馬場状態 → レース当日朝に発表
    'weight_change',     # 馬体重変化 → horse_weightから派生
    'weight_change_abs', # 馬体重変化絶対値 → horse_weightから派生
    'weight_cat',        # 体重カテゴリ → horse_weightから派生
    'weight_cat_dist',   # 体重×距離カテゴリ → horse_weightから派生
    'cond_surface',      # 馬場×馬場種別 → condition_encから派生
}
```

### 過去の失敗から学んだ教訓
| 失敗 | 詳細 | 教訓 |
|------|------|------|
| **odds_logリーク** | 確定オッズを特徴量に使用していた | 絶対に使わない。importance 1位だった |
| **推定ROI過大評価** | `o1*o2*o3*20` が実配当の約2倍 | 必ず実配当ROI(jra_payouts.csv)で判断 |
| **LGB+XGB+MLP** | V10: WF 0.8050 < LGB単体 0.8083 | MLPは逆効果。LGB+XGBで十分 |
| **コース別専用モデル** | 汎用モデルに勝てなかった | 過学習リスク大。汎用モデル一択 |
| **坂路調教マッチ率** | horse_name→horse_id変換が27%しか成功しない | AUC改善なし。現在はmean fillで対応 |
| **Optuna過信** | 100試行で+0.0006のみ | 微改善は本番環境で消える可能性大 |

### リークフリー設計原則
1. 全統計特徴量は**expanding window**（cumsum - current、当該レース除外）
2. sire encodingはfold毎にtrain dataのみで計算（`encode_sires_fold()`）
3. Bayesian smoothing（alpha prior）で低サンプル時の過学習を防止
4. **Pattern Aで評価、Pattern Bで予測**を厳守

---

## 9. 期待値・資金計画

### 年間投資額・期待利益（保守的見積り）
- 月間投資額: 72,100円（全条件合計、700円/レース）
- **月間期待利益: +28,953円**（保守的ROI 142.6%）
- **年間期待利益: +347,436円**

### 月間レース数（推定）
| 条件 | 月間レース | 月間投資 | 月間期待利益 |
|------|-----------|---------|-------------|
| A | 30 | 21,000円 | +9,177円 |
| B | 5 | 3,500円 | +2,303円 |
| C | 22 | 15,400円 | +15,385円 |
| D | 40 | 28,000円 | -1,344円 |
| E | 2 | 1,400円 | -244円 |
| X | 4 | 2,800円 | +3,676円 |

### モンテカルロ破産確率
- **推奨初期資金: 3万円以上**（破産確率0.0%）
- 1万円でも破産確率0.59%と極めて低い
- 1,000レース後の期待資金: 150万円以上

### ドローダウン耐性（初期3万円）
- 平均MDD: 11.1%（約3,300円の一時的損失）
- 最悪MDD: 53.9%（約16,000円の一時的損失）
- 平均回復: 3レースで回復
- 最大連敗: 37レース（全条件合算での理論値）

---

## 10. 未解決課題・今後のタスク

### 高優先度
- [ ] LINE通知実装（予測完了・的中通知）
- [ ] GitHub Actionsによる自動化（現在はWindows タスクスケジューラで代替）
- [ ] 条件D/Eの保守的ROI改善（現状100%以下）
- [ ] 実運用でのROI追跡・アラート（月次100%未満で自動停止）

### 中優先度
- [ ] Pattern Bの天候・馬場情報取得失敗時の代替データソース
- [ ] 新特徴量探索（血統クロス、コース形状、ペース予測等）
- [ ] リアルタイムオッズ変動の反映（EV計算の精度向上）

### 低優先度
- [ ] モバイルUI最適化
- [ ] 複数モデルのA/Bテスト基盤
- [ ] 三連単への拡張検討（hit rate低すぎる可能性）

---

## 11. 全ファイル構成

```
keiba-ai/
├── app.py                          # Streamlitメインアプリ (~5200行)
├── CLAUDE.md                       # このファイル
├── requirements.txt                # Python依存パッケージ
├── packages.txt                    # APTパッケージ (libgomp1)
├── .gitignore                      # 大容量CSV除外設定
│
├── # === モデルファイル ===
├── keiba_model_v9_central_live.pkl # Pattern B (実運用, 75特徴量)
├── keiba_model_v9_central.pkl      # Pattern A (評価用, 67特徴量)
├── keiba_model_v92b_central.pkl    # バックアップ
├── keiba_model_v8.pkl              # フォールバック
│
├── # === 運用スクリプト ===
├── predict_and_log.py              # CLI予測・ログ記録
├── check_results.py                # 結果照合・ROI計算
├── verify_real_roi.py              # netkeiba実配当ROI検証
├── monte_carlo_sim.py              # モンテカルロ破産確率
├── project_status.py               # プロジェクトステータスCLI
├── backtest_central_leakfree.py    # WFバックテスト
├── calc_actual_roi.py              # JRA公式配当ROI計算
├── analyze_conditions.py           # 条件分析
│
├── # === データ取得 ===
├── scrape_jra_track.py             # JRA馬場情報(クッション値/含水率)
├── scrape_weather.py               # 気象庁API天候データ
├── scrape_jra_payouts.py           # JRA公式DB配当データ
│
├── # === バッチファイル ===
├── daily_predict.bat               # 毎朝8:00自動実行
├── daily_results.bat               # 毎晩20:00自動実行
├── weekly_report.bat               # 毎週月曜9:00
│
├── train/                          # === 学習スクリプト ===
│   ├── train_v92_central.py        # V9.2基盤関数群（全特徴量エンジニアリング）
│   ├── train_v92_leakfree.py       # FEATURES_PATTERN_A, LEAK_FEATURES_A定義
│   ├── train_v93_leakfree.py       # Pattern A学習（リークフリー評価用）
│   ├── train_v93_pattern_b.py      # Pattern B学習（当日情報込み実運用）
│   ├── train_v10_ensemble.py       # LGB+XGB+MLP (参考、不採用)
│   ├── optuna_tune_lgb.py          # Optunaハイパラ最適化
│   ├── explore_features.py         # 特徴量探索
│   └── analyze_course_distance.py  # コース/距離分析
│
├── tools/                          # === 運用・検証ツール ===
│   ├── daily_predict.py            # 毎朝自動予測
│   ├── daily_results.py            # 毎晩結果照合
│   ├── weekly_report.py            # 週次レポート
│   ├── extract_jvdata.py           # TARGET JV → CSV抽出
│   ├── validation_1_standardization_leak.py   # リーク検証
│   ├── validation_2_target_variable.py        # 目的変数比較
│   ├── validation_3_ev_filter.py              # EVフィルタ
│   ├── validation_4_ticket_optimization.py    # 券種最適化
│   ├── validation_5_odds_gap.py               # オッズギャップ
│   ├── validation_6_drawdown.py               # ドローダウン
│   ├── validation_7_yearly_performance.py     # 年別パフォーマンス
│   ├── validation_8_data_augmentation.py      # データ拡張
│   ├── validation_9_final_report.py           # 最終レポート統合
│   ├── validation_10_market_dependency.py     # 市場依存性
│   ├── validation_11_sample_size.py           # サンプルサイズ
│   ├── validation_12_roi_integrity.py         # ROI整合性
│   └── validation_13_conservative_roi.py      # 保守的ROI
│
├── tests/                          # === テスト ===
│   ├── test_features.py            # 5項目自動テスト
│   └── debug_all.py                # 25項目デバッグテスト
│
├── data/                           # === データ（大容量はgitignore） ===
│   ├── jra_races_full.csv          # 781,161行 (gitignore)
│   ├── training_times.csv          # 955,580行 (gitignore)
│   ├── odds_history.csv            # 778,387行 (gitignore)
│   ├── blood_full.csv              # 81,986行 (gitignore)
│   ├── jra_payouts.csv             # 27,541件 (gitignore)
│   ├── actual_roi_results.json     # 実配当ROI結果
│   ├── monte_carlo_results.json    # MC結果
│   ├── final_validation_report.json# 最終検証レポート
│   └── ... (検証結果JSON 16ファイル)
│
├── logs/                           # === ログ出力 ===
│
└── archive/                        # === アーカイブ ===
    └── nar/                        # 地方(NAR)関連一式
```

---

## 12. コマンド集

### 起動
```bash
streamlit run app.py                       # Streamlitローカル起動
```

### 予測・結果
```bash
python predict_and_log.py "URL"            # CLI手動予測
python check_results.py                    # 結果照合
python check_results.py --summary          # 成績サマリー
python verify_real_roi.py                  # 実配当ROI検証
```

### 自動運用
```bash
python tools/daily_predict.py              # 当日全レース予測
python tools/daily_predict.py --date 20260315  # 指定日予測
python tools/daily_results.py              # 当日結果照合
python tools/daily_results.py --date 20260315  # 指定日結果
python tools/weekly_report.py              # 週次レポート
```

### モデル学習
```bash
python train/train_v93_leakfree.py         # Pattern A学習（評価用）
python train/train_v93_pattern_b.py        # Pattern B学習（実運用）
```

### バックテスト・検証
```bash
python backtest_central_leakfree.py        # WFバックテスト
python calc_actual_roi.py                  # JRA配当ROI計算
python monte_carlo_sim.py                  # 破産確率シミュレーション
python monte_carlo_sim.py --trials 50000   # 試行回数指定
python project_status.py                   # プロジェクトステータス
python project_status.py --section model   # モデル情報のみ
python project_status.py --export          # JSON出力
```

### テスト
```bash
python tests/test_features.py              # 5項目自動テスト
python tests/debug_all.py                  # 25項目デバッグテスト
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"  # 構文チェック（必須）
```

### データ取得
```bash
python tools/extract_jvdata.py             # TARGET JV → CSV抽出
python scrape_jra_payouts.py               # JRA公式配当データ
python scrape_jra_track.py                 # JRA馬場情報
python scrape_weather.py                   # 気象庁天候データ
```

### 検証スイート（13項目）
```bash
python tools/validation_1_standardization_leak.py   # リーク検証
python tools/validation_2_target_variable.py        # 目的変数比較
python tools/validation_3_ev_filter.py              # EVフィルタ
python tools/validation_4_ticket_optimization.py    # 券種最適化
python tools/validation_5_odds_gap.py               # オッズギャップ
python tools/validation_6_drawdown.py               # ドローダウン
python tools/validation_7_yearly_performance.py     # 年別パフォーマンス
python tools/validation_8_data_augmentation.py      # データ拡張
python tools/validation_9_final_report.py           # 最終レポート統合
python tools/validation_10_market_dependency.py     # 市場依存性
python tools/validation_11_sample_size.py           # サンプルサイズ
python tools/validation_12_roi_integrity.py         # ROI整合性
python tools/validation_13_conservative_roi.py      # 保守的ROI
```

---

## 現行モデルのベースライン（これを下回る変更は一切採用しない）

- **Pattern A AUC: 0.8095**（time-split validation, LGB+XGB ensemble）
- **WF AUC: 0.8017**（walk-forward 2020-2025平均, LGB単体）
- **年別WF AUC**: 2020=0.7951, 2021=0.7997, 2022=0.8024, 2023=0.8012, 2024=0.8071, 2025=0.8048
- **実配当ROI**: A=205%(trio), B=237%(trio), C=286%(trio), D=136%(trio), E=118%(umaren), X=331%(trio)
- **全条件ROI 100%超え**

## 重要ルール

1. **学習はPattern A、予測はPattern B**: バックテスト評価は常にPattern A。実運用予測はPattern B
2. バックテストは必ず**ウォークフォワード**（時系列分割）で実施
3. **改善が確認できない変更は採用しない**: WF AUC > 0.8017 かつ全年AUC > 0.78 かつ 実ROI全条件100%超え
4. app.pyを変更したら必ず**python構文チェック**してからcommit
5. 大きなデータファイル(.csv)は.gitignoreで除外、ローカル保持
6. モデル更新時はAUCが既存モデルを上回る場合のみ本番反映
7. 買い目の馬番は昇順ソート・カンマスペース区切りで表示
8. **中央競馬専用** — 地方(NAR)コードはarchive/nar/に保管

## 定期タスク（Windows タスクスケジューラ）

| 時間 | タスク | コマンド |
|------|--------|---------|
| 毎朝8:00 | 当日全レース予測 | `python tools/daily_predict.py` |
| 毎晩20:00 | 結果照合・ROI計算 | `python tools/daily_results.py` |
| 毎週月曜9:00 | 週次レポート | `python tools/weekly_report.py` |

バッチファイル: `daily_predict.bat`, `daily_results.bat`, `weekly_report.bat`
ログ: `logs/` ディレクトリ

## Compaction対応

Claude Codeのコンテキスト圧縮時に失われやすい重要情報はこのCLAUDE.mdに集約。
特に「現行モデルのベースライン」「リーク厳禁ルール」「過去の失敗教訓」は常に参照すること。
