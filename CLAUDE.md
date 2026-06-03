# CLAUDE.md

> **caveman mode**: respond like caveman. short word. no verbose. do thing, say little. result only.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Last updated: **2026-05-19 (Session #91 E-2、 slim 化 78k→43k bytes / 30k chars)**

> Session #88-90 (5/16-18): drift 5 件全訂正完了。V15 真値確定 (V15-audit-1〜5 / data-audit-1〜4)。
> 5/17-18 memory drift: ROI 119.2%→98.34% / arch 4-model→LGB+XGB 2-model / AUC 0.8939→0.8678(WF) / features 150→145 / formation 永久喪失。
> 全詳細: docs/MEMORY_DRIFT_FINAL_RESOLUTION_2026_05_17.md + docs/B2_CLAUDE_MD_FULL_DRIFT_RESOLUTION_2026_05_18.md

### JRA-VAN 加入 + JV-Link 環境 (2026-05-07 夜 確定)

| 項目 | 値 |
|------|----|
| JRA-VAN DataLab | 加入完了 |
| JV-Link DLL | C:\\Windows\\SysWow64\\JVDTLAB\\JVDTLab.dll (32-bit only, ver 1.18) |
| ProgID | JVDTLab.JVLink |
| 動作確認 (5/7 夜) | 過去日付 5/3 で 29 ファイル取得 OK |
| 32-bit Python venv | C:\\Users\\takum\\jvlink-venv\\ (推奨、 5/24+ 着手) |
| 既存 64-bit 環境 | 完全維持 (predict_core / daily_predict 含む) |

---

## 1. プロジェクト概要

**競馬AI予測システム（中央競馬専用）**

JRA中央競馬の全レースをAIで予測し、条件別に最適な買い目を自動生成するシステム。
★ V15 production の真の architecture = **LGB+XGB 2-model** ★ (FT-Transformer / IntraRace Attention は v15_master の WF 評価専用、 `.pkl.gz` には未保存。 production inference は LGB+XGB のみ — V15-audit-1)。 6 条件分類に基づき 三連複/馬連 の買い目を自動生成。

- **Streamlit**: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- **GitHub**: https://github.com/takumi0310s/keiba-ai
- **現行モデル**: **V15** (本番、★ 145 features booster ★ / Pattern B features list 150 だが 5 件 truncate で booster 145、★ LGB+XGB 2-model ★、 stored `.pkl.auc = 0.8939` は **LGB train-set self-eval (in-sample)**、 真の genuine WF LGB+XGB = **0.8678** / Grid 4-model 5-fold = **0.8858** — V15-audit-2)
- 本番運用 ROI: **98.34%** / **PnL ¥-6,920** / n=596 (≤2026-05-17、 全 settled、 V15-audit-4)、 bootstrap 95% CI **[66.33%, 138.05%]** 100% 含む = ★ 統計的有意 勝ち なし ★
- 旧モデル: v13.5b は historical reference (124 特徴量、Grid Ensemble、WF AUC 0.8788)
- 5/9 V15 案B改 単独継続 (絶対)。 5/16 V15.1 / V18/V19 共に NO-GO 確定 (Session #38)
- Phase 3 (5/24+): sib_*_exp 修正版 + V20 学習 + JV-Link 加入 → 7/1 V20 投入候補
- Phase 4 (7-8 月): 調教動画 AI 解析 PoC → 9/1 V21 投入候補
- 現行 累計収支: **¥-6,920** / 撤退余裕 **¥43,080** (5/17 反映、 V15-audit-4) ※ 旧値 +13,530 / +5,240 は drift、 5/16 P0-1 → 5/17 audit で 真値確定
- **2段階モデル**: Pattern A（リークフリー評価用）+ Pattern B（当日情報込み実運用）
- **検証済み**: WF 2020-2025, 実配当ROI 428.4%, 全条件PASS

### Session #38 確定事項 (2026-05-07)

| # | 確定 | 影響 |
|---|------|------|
| 1 | **V15.1 SKB POST-RACE LEAK 確定** (skb_kishi_code_3 +480bp / corr_target 0.137 / monotonic 1着→364, 10着→176) | V15.1 採用 NO-GO、 V20 で SKB 全 10 features 完全除外 |
| 2 | **V18/V19 sib抜き hybrid** (LIVE winner_top1 -10.3pt + shift 30.4x→8.3x) | sib = リーク + 識別能力 hybrid、 expanding window 修正版が必要 |
| 3 | **5/16 V18/V19 投入 NO-GO** | V15 案B改 単独継続、 6/15+ sib_*_exp 版で再判定 |

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
- `tools/weekly_report.py` — 毎週月曜9:00、週次パフォーマンスレポート（条件別・特徴量率・乖離率・累積ROI警告付き）
- `tools/refresh_cookie.py` — netkeibaのCookie自動更新（Playwright自動ログイン）
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

---

## 4. モデル詳細

### 2段階モデル設計思想
- **Pattern A（評価用）**: リークフリー厳守。モデルの真の実力を評価
- **Pattern B（実運用）**: 使える情報は全て使って最高精度で予測

### ★ V15 真値 (5/17 V15-audit-1〜5 確定) ★

| 項目 | 真値 | 出典 |
|------|------|------|
| ensemble architecture | **LGB + XGB 2-model** (mlp=None, FT/IR は .pkl 未保存) | V15-audit-1 |
| ensemble_weights (stored) | `{lgb: 0.5036, xgb: 0.4964, mlp: 0}` | V15-audit-1 |
| Pattern A booster features | **145** | V15-audit-1/3 |
| Pattern B features list | 150 (但し booster 入力後 5 件 truncate → 145) | V15-audit-1 |
| stored `.pkl.auc` | 0.8939485520467574 ★ = LGB train-set self-eval (in-sample) ★ | V15-audit-2 |
| stored .pkl 自己 inference 6-fold mean (2020-25) | 0.8929 | V15-audit-2 (LEAKY upper bound) |
| **genuine WF 6-fold mean LGB+XGB** (fold ごと retrain) | **0.8678** | V15-audit-2 |
| **genuine WF 5-fold mean LGB+XGB** (2021-25) | 0.8695 | V15-audit-2 |
| v15_master_report.json Grid 4-model 5-fold mean | 0.8858 | V15-audit-2 |
| RED_IMP_BUT_CONST | **0 件** | V15-audit-3 / T1 monitor |

★ drift 訂正: CLAUDE.md の旧記述 「WF AUC 0.8939 / 4-model ensemble / 150 features」 は全て drift。 真値は 上表参照。 production の inference は LGB+XGB only。

### Pattern A スペック（v13.5b、historical reference）
- ファイル: `keiba_model_v135_central.pkl.gz`（LGB+XGB）+ FT/IRモデルは動的学習
- WF AUC: **0.8788**（walk-forward 2020-2025平均, 4-model grid ensemble）
- 特徴量: **124個**（リークフリー、JRDB連携含む）
- 学習データ: ~527,000行
- 目的変数: `finish <= 3`（複勝圏 binary）
- アンサンブル: LGB + XGB + FT-Transformer + IntraRace Attention（Grid重み最適化）
- 実配当ROI: **428.4%**（JRA公式配当、WF 2023-2025、全条件PASS）

### Pattern A スペック（v12、旧版）
- ファイル: `keiba_model_v12_central.pkl.gz`
- WF AUC: **0.8037**（walk-forward 2020-2025平均, LGB単体）
- 特徴量: **74個**（リークフリー、V9.3の67 + V12の7）

### Pattern B スペック
- ファイル: `keiba_model_v135_central_live.pkl.gz`（LGB+XGB）
- 特徴量: **132個**（Pattern A 124 + 当日情報8）
- app.pyはPattern Bを優先、なければA→V12→V8にフォールバック
- 馬場/天候データ取得失敗時は0=欠損として予測

### Pattern A 主要特徴量（v12 74個 → V15 145個、詳細は train/train_v92_central.py 参照）

<!-- 74特徴量詳細テーブルは削除 (slim化)。カテゴリ: 基本14/騎手3/前走ラグ10/集計5/派生11/V9.2追加12/V9.3新規12/V12新規7 -->


Pattern B 追加特徴量 (8個): odds_log / pop_rank / horse_weight / weight_change / weight_change_abs / weight_cat / weight_cat_dist / condition_enc / cond_surface / cushion_value / moisture_rate / temperature / humidity / wind_speed / precipitation / weather_enc

### アンサンブル構成（v13.5b、現行）
- **4モデル Grid Ensemble**: LightGBM + XGBoost + FT-Transformer + IntraRace Attention
- 重み: Grid Search最適化（年ごとに異なる）
- 典型的重み: LGB=0.25, XGB=0.25-0.30, FT=0.10-0.15, IR=0.35
- `pred = w_lgb * lgb + w_xgb * xgb + w_ft * ft + w_ir * ir`
- IntraRace Attentionがレース内相対関係を捕捉し、最大の貢献（重み0.35）

### アンサンブル構成（v12以前）
- **LightGBM（主）** + **XGBoost（副）**
- 重み: AUC比例（LGB ~56%, XGB ~44%）

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
- Best: WF AUC 0.8022（+0.0006）→ 基準未達のため不採用
- 現行パラメータを維持（V12でも同一パラメータ）

---

## 5. 全条件詳細

### 条件定義テーブル

#### v13.5b（現行）WF 2023-2025, 10,314レース, 4-model Grid Ensemble (AUC 0.8788), JRA公式配当データ

| 条件 | 条件内容 | 買い目 | v13.5b実ROI | 的中率 | N | v13.4実ROI |
|------|----------|--------|------------|--------|------|-----------|
| A | 8-14頭/1600m+/良〜稍重 | trio 7点 | **355.4%** | 63.2% | 3,212 | 308.0% |
| B | 8-14頭/1600m+/重〜不良 | trio 7点 | **346.8%** | 61.3% | 398 | 259.4% |
| C | 15頭+/1600m+/良〜稍重 | trio 7点 | **623.0%** | 51.6% | 2,473 | 521.5% |
| D | 1200-1400m | trio 7点 | **360.8%** | 48.2% | 3,581 | 314.7% |
| E | 7頭以下 | umaren 2点 | **195.7%** | 72.7% | 267 | 141.4% |
| X | 15頭+/重〜不良 | trio 7点 | **701.2%** | 54.0% | 383 | 498.9% |
| **全体** | | | **428.4%** | | **10,314** | 361.9% |

#### v12（旧版）WF 2020-2025, 20,579レース, LGB単体 (AUC 0.8037), JRA公式配当データ

| 条件 | 条件内容 | 買い目 | 実ROI | 的中率 | N |
|------|----------|--------|-------|--------|------|
| A | 8-14頭/1600m+/良〜稍重 | trio 7点 | **205.3%** | 44.5% | 6,438 |
| B | 8-14頭/1600m+/重〜不良 | trio 7点 | **236.9%** | 45.2% | 847 |
| C | 15頭+/1600m+/良〜稍重 | trio 7点 | **285.6%** | 33.7% | 4,774 |
| D | 1200-1400m | trio 7点 | **136.0%** | 27.0% | 7,254 |
| E | 7頭以下 | umaren 2点 | **118.0%** | 53.4% | 461 |
| X | 15頭+/重〜不良 | trio 7点 | **330.5%** | 35.5% | 805 |

- **全条件ROI 100%超え**（v13.5b、v12ともに）
- **1000m以下は非推奨**: ROI 85.0% (N=534) → 予測はするが購入非推奨表示
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

## 6. テスト結果 (historical, 全詳細は data/*.json 参照)
- WF AUC: V12=0.8037 / v13.5b=0.8788 / V15 genuine=0.8678 (LGB+XGB) / Grid=0.8858
- モンテカルロ: 初期3万円→破産0.0%、期待資金150万+
- 保守的ROI: 全体142.6% (BT×0.7)

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

### V20 で追加除外する特徴量（Session #38 確定、計 18 個）
```python
# train/v15_1_features.py V20_LEAK_FEATURES (Session #39 C で実装)
SKB_LEAK_FEATURES = [  # Session #38 で SKB POST-RACE LEAK 確定
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1',  'skb_baba_code_2',  'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
]  # 10 features (V15.1 SKB 全体)
V20_LEAK_FEATURES = LEAK_FEATURES_A | SKB_LEAK_FEATURES  # 8 + 10 = 18
```

V20 学習時は `merge_v15_1_features(skip_skb=True)` で完全除外。

### 過去の失敗から学んだ教訓
| 失敗 | 詳細 | 教訓 |
|------|------|------|
| **odds_logリーク** | 確定オッズを特徴量に使用していた | 絶対に使わない。importance 1位だった |
| **推定ROI過大評価** | `o1*o2*o3*20` が実配当の約2倍 | 必ず実配当ROI(jra_payouts.csv)で判断 |
| **LGB+XGB+MLP** | V10: WF 0.8050 < LGB単体 0.8083 | MLPは逆効果。ただしFT-TransformerとIntraRaceは有効（v13.5b） |
| **コース別専用モデル** | 汎用モデルに勝てなかった | 過学習リスク大。汎用モデル一択 |
| **坂路調教マッチ率** | horse_name→horse_id変換が27%しか成功しない | AUC改善なし。現在はmean fillで対応 |
| **Optuna過信** | 100試行で+0.0006のみ | 微改善は本番環境で消える可能性大 |
| **dam_top3rリーク** | 全年データで母産駒成績を計算→WF AUC+0.023の大半がリーク | 外部CSVの集計値は必ずexpanding windowで。静的CSVをそのまま使うな |
| **SKB POST-RACE LEAK** (Session #38, 5/7) | skb_kishi_code_3 単独 +480bp、 corr_target 0.137、 1着馬 0-rate 15% / 敗者 49%、 finish と monotonic | JRDB SKB ファイル = "成績拡張" = post-race。 **V20 で全 10 features 完全除外**、 LEAK_FEATURES の追加 list 化 |
| **sib_top3_rate hybrid** (Session #38) | 旧 sib_top3_rate corr_target 0.2939 → 新 sib_top3_rate_exp 0.1689 (-0.125 リーク除去後の真の信号 0.169 残存) | 静的 CSV の集計値は必ず date 順 cumsum-current で expanding 化。 dam_top3r 教訓と同根の再発 |
| **jrdb_ze_* リーク** (2026-06-02 発見) | `jrdb_features.py:788` が ZED(過去走成績=結果) を blood_num で **日付カットオフ無し全期間平均** → 当該/未来の成績混入。 4特徴(ze_idm_avg/ze_ten_avg/ze_agari_avg/ze_furi_count)。 override test で「市場本命を覆した馬が44%勝つ」=結果を見ていた。 dam_top3r/SKB と同型 | **当該race日付より前のZEDのみで expanding 平均**。 leak-free cache `data/_v15_optuna_df_cache_leakfree.pkl.gz` で除去。 ★本番liveは過去ZEDのみで元々leak-free・リークはbacktest/cacheのみ・実運用98%は本物★ |

### 🧪 leak-free 監査 + 評価基準 (2026-06-02/03, 検証専用・本番不変)
- **ze リーク発見→除去**: 上表参照。 leak-free cache = `data/_v15_optuna_df_cache_leakfree.pkl.gz`(元cache不変・ze4特徴のみ当該日付前でexpanding再計算)。 検証scripts = `tools/v16_anaba_*.py` / `v16_make_leakfree_cache.py` / `v16_leakfree_roi_grid.py` / `v16_pastmodels_leakfree.py`。
- **V15 真値(leak-free)**: WF AUC **~0.842**・単勝ROI **108%**(実運用98%と整合)。★リーク版の AUC 0.8696 / 単勝ROI 156% は無効。CLAUDE.md 旧記載の「genuine WF 0.8678」もzeリークで嵩上げの疑い★。
- ★**評価は AUC でなく leak-free ROI で行う**★: 穴特化 **s2b** は AUC を犠牲(0.829)にして ROI を獲得 — leak-free 全券種で V15 超(単勝 **111.6%** / 三連複top4box **194.3%** [95%CI 179-212, V15は146-166でCI非重複=有意] / 馬連top3box 146%)。 自信度top10%に絞ると三連複top4box **211%**(N=1034)。 市場(人気)は全券種70-84%。 過去モデル V24/V24b は V15 とほぼ同一(ROIで s2b 未満)。 ★当時のAUC基準NO-GO判定では穴特化の価値が見えなかった★。
- **s2b 定義**: V16能力137 − 人気代理"族"13(`paci_jockey_exp_wr/_3rd` + 印4(`paci_jockey_mark/sogo_mark/train_mark/idm_mark`) + `jrdb_cid_idx/ls_idx/training_idx/stable_idx` + `paci_goal_rank/goal_diff/dochu_rank`) + レース相対特徴(脚質構成 n_front/`front_advantage`、距離適性合致、脚質×バイアス×枠)。 `tools/v16_anaba_s2_eval.py`。 候補=`models/v16_anaba_s2b_candidate.pkl.gz`(検証専用・投票未使用)。
- **騎手指数の正体**: `paci_jockey_exp` = JRDB「騎手期待率」= **93%が人気代理**(残差6-7%、odds_dependency_analysis.json)。 ルメール反証: 騎手内 corr(値,人気)≈−0.83・ルメールでもJEがレース内最高は41%のみ。 脚質/距離適性は **per-horse単一コードでは死(gain≈0)、レース相対化で蘇生**(front_advantage 0.1→0.86%)。
- **未完(次段階)**: 前向き paper trading(唯一リーク不可能な確証)、 horse_name→blood_num 100%カバレッジ化での leak-free 忠実度UP。 本番 `jrdb_features.py` の ze集計への日付フィルタ追加は防御的別件(live は既に安全・要承認)。

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

---

## 10. 未解決課題・今後のタスク

### 高優先度
- [ ] LINE通知実装（予測完了・的中通知）
- [ ] GitHub Actionsによる自動化（現在はWindows タスクスケジューラで代替）
- [ ] 条件D/Eの保守的ROI改善（現状100%以下）
- [x] 実運用でのROI追跡・アラート（tools/roi_monitor.py、daily_results.py後に自動実行、Discord警告通知）

### 中優先度
- [ ] Pattern Bの天候・馬場情報取得失敗時の代替データソース
- [ ] 新特徴量探索（血統クロス、コース形状、ペース予測等）
- [ ] リアルタイムオッズ変動の反映（EV計算の精度向上）

### 低優先度
- [ ] モバイルUI最適化
- [ ] 複数モデルのA/Bテスト基盤
- [ ] 三連単への拡張検討（hit rate低すぎる可能性）

---

## 11. 主要ファイル構成

| ファイル | 用途 |
|---------|------|
| app.py | Streamlit メインアプリ (~5200行) |
| tools/predict_core.py | 共通予測ロジック |
| tools/race_auto_notify.py | レース自動通知・戦略フィルタ |
| tools/daily_predict.py | 毎朝自動予測 |
| tools/daily_results.py | 毎晩結果照合 |
| tools/weekly_report.py | 週次レポート |
| tools/kelly_criterion.py | Kelly 基準投票額計算 |
| tools/strategy_filters.py | 戦略フィルタ関数 (C4/C3/B1/B2/C2) |
| tools/strategy_rollback.py | 異常時自動 rollback |
| tools/paper_shadow_v15_full.py | candidate model 並行予測 |
| tools/anomaly_auto_detector.py | 異常検知 + strategy anomaly |
| tools/race_notify_log_v2.py | 8 strategy 並行追跡 |
| tools/daily_discord_report.py | Discord 日次収支レポート |
| tools/admin_verify_v2.py | schtask/bat/py verify ツール |
| train/train_v15_master.py | V15 学習スクリプト |
| tests/ | pytest テストスイート |
| models/ | candidate .pkl.gz (v15_full_optuna / v15_2) |
| data/cumulative_results.csv | 累計予測・収支ログ |
| data/race_notify_log_v2_summary/ | 8 strategy 日次集計 |

---

## 実戦前チェックリスト（毎週土曜朝に実行）

開催日の朝、予測を始める前に必ず以下を実行すること。

### チェック項目

1. **モデルファイル（pkl.gz）の読み込み確認** — `keiba_model_v9_central_live.pkl.gz` / `keiba_model_v9_central.pkl.gz` が正常にロードできるか
2. **feature_lookups.pklの存在・サイズ確認** — `data/feature_lookups.pkl` または `.pkl.gz` が存在し、キー数10以上あるか
3. **netkeibaへのアクセス確認** — 出馬表ページ・オッズAPIからレスポンスが返るか
4. **JRA馬場データのアクセス・エンコーディング確認** — `scrape_jra_track.py` がShift_JISで正しくパースできるか
5. **気象庁APIの確認** — `scrape_weather.py` が気温・湿度・風速を返すか
6. **DBの存在・レコード数確認** — SQLite DBが存在し、過去予測レコードが読めるか
7. **今日の最初のレースURLで特徴量テスト** — 87/87特徴量が全て生成されるか
8. **ゼロ特徴量が5個以上なら警告** — 0値の特徴量が多い場合は原因調査

### チェック基準

| 結果 | 判断 |
|------|------|
| 全項目OK | **実戦開始** |
| 警告あり | 影響確認してから判断 |
| エラーあり | **修正するまで実戦禁止** |

### よくあるバグと対処法

| 症状 | 原因・対処 |
|------|-----------|
| 特徴量18/87 | app.pyのバージョン条件（`use_version in ('v5','v6','v8','v9')` 等）にモデルバージョンが含まれているか確認 |
| cushion_value=0 | `scrape_jra_track.py`のencodingが`shift_jis`になっているか確認（JRA公式はShift_JIS） |
| モデル読み込み失敗 | `pkl.gz`形式対応を確認（`gzip.open` + `pickle.load`） |
| netkeiba 403 | `User-Agent`ヘッダーが設定されているか確認 |
| オッズ全て0 | `fetch_realtime_odds()`のJSON APIエンドポイント確認。レース前はオッズ未発売の場合あり |
| RaceName=Unknown | `soup.find(class_="RaceName")`でタグ種別を限定せず検索しているか確認（`<h1>`の場合あり） |

---

## 毎回の起動手順（コピペ用）

### 土曜朝（実戦前）
```bash
cd keiba-ai
git pull
python tools/pre_race_check.py
python -m streamlit run app.py
```

### エラーが出た場合
```bash
claude --dangerously-skip-permissions
```
→「pre_race_checkでエラーが出た。修正して」

### Claude Code起動（開発作業）
```bash
cd keiba-ai
claude --dangerously-skip-permissions
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
python train/train_v135b_intra_ensemble.py # v13.5b 4-model ensemble WFバックテスト+学習（現行）
python train/train_v135_ft_transformer.py  # v13.5 FT-Transformer学習
python train/train_v12_comprehensive.py    # V12 WFバックテスト+学習（旧版）
python calc_actual_roi_v135b.py            # v13.5b 実配当ROI検証（JRA公式配当）
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
python tools/refresh_cookie.py             # Cookie自動更新（対話式）
python tools/refresh_cookie.py --check     # Cookie有効性チェック
python tools/refresh_cookie.py --auto      # 期限切れ時のみ自動更新
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

## 現行モデルのベースライン（V15 真値、 5/17 V15-audit 反映）

★ ★ ★ 重要: 旧 CLAUDE.md に書かれた「WF AUC 0.8788 (v13.5b)」「WF AUC 0.8939 (V15)」 は何れも production の genuine WF 値ではない (前者は v13.5b の Grid mean / 後者は LGB train-set self-eval)。 真の V15 production の genuine WF baseline は以下。 ★ ★ ★

### V15 production (5/17 V15-audit-1〜5 真値)
- ★ architecture: **LGB + XGB 2-model** (mlp=None, FT/IR は .pkl.gz 未保存) ★
- ★ features: **145** (booster) / Pattern B 150 だが truncate で 145 ★
- ★ **genuine WF 6-fold mean LGB+XGB = 0.8678** ★ (V15-audit-2、 fold ごと retrain)
- ★ 5-fold mean (2021-25) LGB+XGB = 0.8695 / Grid 4-model 5-fold mean = 0.8858 ★
- stored `.pkl.auc` = 0.8939 は LGB train-set self-eval (in-sample、 LEAKY)、 generalization 指標ではない
- ★ 累計運用 ROI = **98.34%** / PnL **¥-6,920** / n=596 (≤2026-05-17、 V15-audit-4) ★
- bootstrap 95% CI [66.33%, 138.05%] 100% 含む → ★ 統計的有意 勝ち なし ★

### 旧ベースライン (historical reference)
- v13.5b 4-model Grid Ensemble: WF AUC 0.8788, 実配当ROI 428.4% (歴史 reference)
- v13.4 LGB+XGB: WF AUC 0.8656, 実配当ROI 361.9%
- V12 LGB単体: WF AUC 0.8037, 実配当ROI ~205%

注: v13.5b の「実配当ROI 428.4%」 は backtest 計算値、 V15 の cumulative 98.34% (実運用 settled) と直接比較不能 (期間 / 投票戦略 / オッズ取得タイミング 全く異なる)。

## 重要ルール

1. **学習はPattern A、予測はPattern B**: バックテスト評価は常にPattern A。実運用予測はPattern B
2. バックテストは必ず**ウォークフォワード**（時系列分割）で実施
3. **改善が確認できない変更は採用しない**: V15 真値 baseline (genuine WF LGB+XGB = 0.8678 / Grid 4-model = 0.8858) を超え、 cumulative ROI (98.34%、 5/17) を有意改善 (CI [66.33%, 138.05%] を上抜け)
4. app.pyを変更したら必ず**python構文チェック**してからcommit
5. 大きなデータファイル(.csv)は.gitignoreで除外、ローカル保持
6. モデル更新時はAUCが既存モデルを上回る場合のみ本番反映
7. 買い目の馬番は昇順ソート・カンマスペース区切りで表示
8. **中央競馬専用** — 地方(NAR)コードはarchive/nar/に保管

## 定期タスク（Windows タスクスケジューラ）

| 時間 | タスク | コマンド | バッチ |
|------|--------|---------|--------|
| 毎日 03:00 | プレミアムデータ事前取得 | `python tools/daily_premium_scrape.py` | `daily_premium_scrape.bat` |
| 毎日 08:00 | 当日全レース予測 | `python tools/daily_predict.py` | `daily_predict.bat` |
| 土日 08:45 | レース5分前自動予測＆Discord通知 | `python tools/race_auto_notify.py` | `race_auto_notify.bat` |
| 土日 18:00 | 結果照合・ROI計算 | `python tools/daily_results.py` | `daily_results.bat` |
| 毎晩 20:00 | 結果照合（平日含む） | `python tools/daily_results.py` | `daily_results.bat` |
| 月曜 08:00 | 週次レポート | `python tools/weekly_report.py` | `weekly_report.bat` |

一括登録: `setup_all_tasks.bat`（管理者権限で実行）
ログ: `logs/` ディレクトリ

---

## netkeibaプレミアムデータ連携

### 概要
netkeibaスーパープレミアム会員のCookie認証でプレミアムデータを取得。
Cookie設定: `.env` の `NETKEIBA_COOKIE` に保存。
Cookie期限切れ時は `python tools/refresh_cookie.py` で自動更新可能（Playwright）。
認証情報を保存済みなら `python tools/refresh_cookie.py --auto` で期限切れ時のみ自動実行。

### 取得データ一覧

| データ | ソース | 取得タイミング | 用途 |
|--------|--------|------------|------|
| 調教タイム(4F/3F/1F) | `oikiri.html` | 予測時リアルタイム | モデル特徴量（wood_best_4f_filled等） |
| 追い切りランク(A/B/C/D) | `oikiri.html` | 予測時リアルタイム | タイム取得失敗時のフォールバック |
| タイム指数 | `speed.html` | 予測時リアルタイム + 事前取得 | UI表示（モデル組込は再学習後） |
| 厩舎コメント | `comment.html` | 予測時リアルタイム | スコア化(-3〜+3)してUI表示 |

### 調教タイム取得の4段階フォールバック
1. Premium実タイム (Cookie有効 → oikiri.html 4F/3F/1F秒数)
2. ランク推定 (Cookie無効 → A:51.5s, B:53.0s, C:54.5s, D:55.5s)
3. feature_lookups.pkl (キャッシュ値)
4. デフォルト値 (52.0/53.0/39.0s)

### 関連ファイル

| ファイル | 用途 |
|----------|------|
| `scrape_training.py` | 調教タイム・コメント取得モジュール |
| `tools/scrape_speed_index.py` | タイム指数一括取得 |
| `tools/scrape_premium_data.py` | 調教タイム一括取得 |
| `tools/bulk_scrape_history.py` | 過去データ一括取得（手動実行） |
| `tools/daily_premium_scrape.py` | 週末レースデータ事前取得（AM3:00自動） |
| `tools/weekly_premium_update.py` | 週末Premium更新 |

### 蓄積データ状況

| データ | ファイル | 行数 | 年度 |
|--------|---------|------|------|
| タイム指数 | `data/netkeiba_speed_index.csv` | ~143K | 2020-2025 |
| 調教タイム | `data/netkeiba_training_times.csv` | ~2.6K | 2025（部分） |
| 厩舎コメント | `data/netkeiba_stable_comments.csv` | ~857 | 2025（部分） |
| レース短評 | `data/netkeiba_race_review.csv` | ~277K | 2020-2025（全年カバー） |
| 新馬評価 | `data/netkeiba_shinba_eval.csv` | ~8K | 2024-2025 |

---

## Discord通知システム

### チャンネル振り分け

| チャンネル | 環境変数 | 内容 |
|-----------|---------|------|
| #買い目 | `DISCORD_WEBHOOK_BETS` | レース予測、フォーメーション、配当レンジ |
| #アップデート | `DISCORD_WEBHOOK_UPDATES` | スクレイピング完了、結果照合、週次レポート |
| フォールバック | `DISCORD_WEBHOOK_URL` | 上記未設定時 |

### 通知形式

**買い目通知（三連複）**:
```
🏇 中山11R 発走15:45
アネモネS 芝1600m 良 条件A ★★★

三連複フォーメーション 7点
1列目: 1
2列目: 2, 3
3列目: 2, 3, 4, 5, 6

軸: ホワイトオーキッド (1) スコア0.85
💰 配当レンジ: 1,200円〜15,600円
📊 指数: 1127 / 調教: A / 厩舎: 好調
```

### セットアップ
```bash
python tools/setup_discord.py  # 対話式Webhookセットアップ
```

---

## V12特徴量追加結果（2026-03-29、historical reference）

v12総合再学習: 10特徴量投入→7採用3不採用。全詳細は git log 参照。

詳細: `docs/V15_AUDIT_*_2026_05_17.md` 参照。

---

---

## タスク完了通知

コマンドラインからDiscord通知を送信:
```bash
python tools/notify_done.py "タスク名" "詳細メッセージ"
python tools/notify_done.py "エラー" "内容" --color red
```

## プロンプト標準テンプレート

今後の全タスクプロンプトの末尾に以下を含めること:
```
git commit & push して。
完了したら以下を実行：
powershell -Command "[System.Media.SystemSounds]::Exclamation.Play()"
python tools/notify_done.py "タスク名" "完了内容"
```

## Compaction対応

Claude Codeのコンテキスト圧縮時に失われやすい重要情報はこのCLAUDE.mdに集約。
特に「現行モデルのベースライン」「リーク厳禁ルール」「過去の失敗教訓」は常に参照すること。

| `prev_race_pace_diff` | 0%充填 | **削除** (JRDB SED は15.2%しかカバーできず修復困難) |
| `gaisha_rank` | 0%充填 | **削除** (`jrdb_ranch_rank` 94.2%充填で代替済み) |
| `course_renovated` | 1.3%充填 | **永久化適用済 (4/27)** (京都2万件を活性化) |
| `jrdb_tb_homestr_inner` | 99.7%充填 | **健全** (修復不要) |

### 🛠 戦略⑦ 自動化 (4/27 適用済)
**実装場所**: `tools/race_auto_notify.py` (`predict_and_notify` 関数内)

**フィルタ内容**:
- `06_特別` (G/L/OPEN特別 ではない平場特別) を除外: -9,470円損失源
- `京都` を除外: データ蓄積待ち、5/11 以降に再評価
- `条件E` (頭数<=7) を除外: サンプル少
- `条件B` (重~不馬場) を除外: サンプル少

**期待効果 (旧 drift 記述)**: ROI 119.2% → 140.3% (+21.1pt) / 298R → 242R, 損益 +28,240円改善 ※ ★ 旧 119.2% 自体 drift、 5/17 V15-audit-4 真値 98.34% / PnL ¥-6,920 ★
※ 旧値は drift。 5/16 P0-1 真値: baseline ROI 101.33%、 戦略⑦ applied ROI 96.90% (n=466、 ≤5/10、 PnL -¥10,120)。 戦略⑦込み 100% 超え の 仮想値は cumulative 集計では 再現不能 (docs/ROI_DISCREPANCY_2026_05_16.md)

### 🎯 1レース再予測ツール (4/26 動作確認)
**ファイル**: `tools/predict_one_race.py`
**用途**: 取消発生時、1レース指定で予測再実行
**動作確認**: 4/26 東京11Rフローラ Sで成功

```bash
python tools/predict_one_race.py 202605020211
```

### 📋 v16 ロードマップ
- **Week 1 (4/27-5/3)**: 基盤修正 + 戦略⑦運用開始
- **Week 2 (5/4-5/10)**: 死特徴量整理 + 訓練データ拡張
- **Week 3 (5/11-5/17)**: v16 学習実行 (Ryzen 7 + 32GB + 16GB GPU で2-3時間)
- **Week 4 (5/18-5/24)**: A/B検証 → 5/末 v16 本番投入

### 🎯 v16 目標 (旧 plan、 V15 真値 audit 後 v15_full / v15.2 / V22 への再 frame 想定)
- **AUC** (旧記述): 0.895+ (v15: 0.8939) ※ 真値 V15 genuine WF LGB+XGB 0.8678 / Grid 4-model 0.8858 (V15-audit-2)、 目標は 0.8678 → 0.88+ 推移
- **特徴量数** (旧記述): 148 (-2) ※ 真 V15 booster 145、 -2 = 143 が真の意図
- **ROI** (戦略⑦込み): 140%+ ※ V15 実 cumulative 98.34% (5/17) からの増分、 統計的有意 (CI 100% を上抜け) が必要
- **京都ROI**: 80%+ (course_renovated 永久化効果)
- **本番切替**: 5月末

### 🔍 既知のバグ
- jrdb_paci.csv が4/4から更新停止 (取得経路不明、要修復) → **JV-Link O1 で代替経路 (Session #39 B)**
- predict_core.py に FutureWarning が15箇所以上 (4/27 主要箇所修正済)
- cumulative_results.csv に top1_num/score 書き込まれていない (95%欠損)
- nightly_sanity の SCRAPER-GUARD 認識バグ (誤検知)
- jra_payouts.csv が4/6で更新停止 → **JV-Link HR で代替経路 (Session #39 B)**
- JRDB データの2026年分が未取得 (race_id 列なしの可能性)

---

## Phase 3-4 統合 roadmap (Session #39 J、 5/24-8月)

### Phase 3 前半 (5/24-6/8): 基盤整備 + sib_*_exp 統合

| 期間 | 内容 |
|------|------|
| 5/24 | JRA-VAN DataLab 加入、 JV-Link DLL インストール、 jvlink_fetcher.py 動作確認 |
| 5/25-5/27 | sib_expanding_features.py を train/features_v15_new.py に統合 (旧 sib_top3_rate / sib_shinba_wr 入れ替え) |
| 5/28-5/30 | V18/V19 sib_*_exp 版 6-fold WF (LGB+XGB) |
| 6/1-6/5 | V18/V19 LIVE retro (5/30 + 5/31 + 6/1)、 winner_top1 ≥ 30% / shift ≤ 12x 検証 |
| 6/6-6/8 | sib_*_exp GO/no-go 判定。 GO なら 6/15+ V18/V19 段階投入 (週末のみ、 上限 5,000円/日) |

### Phase 3 後半 (6/9-6/30): V20 構築

| 期間 | 内容 |
|------|------|
| 6/9-6/13 | JV-Link parser 実装 (RACE/HR/O1/TCOV/WOOD/BLOD)、 過去 1 年 bulk fetch + 整合チェック |
| 6/14-6/20 | V20 学習 data spec 確定 (JRA + NAR 統合、 共通 80 features、 SKB 完全除外、 sib_*_exp 込み)、 V20 v1 学習 (4-model ensemble 計画、 但し V15-audit-1 で V15 production が LGB+XGB 2-model のみ判明 → V20 は ★ FT/IR の .pkl 保存 ★ も含めた full ensemble production 化を必須化) |
| 6/21-6/25 | V20 WF 検証 (6-fold、 2020-2025)、 LIVE retro |
| 6/26-6/28 | V20 paper trading (週末) |
| 6/29-6/30 | GO/no-go 最終判定 (WF AUC ≥ 0.880 / LIVE retro ≥ 30% / shift ≤ 12x / NAR AUC ≥ 0.83 / paper ROI ≥ 110% / LEAK 監査 PASS) |

### Phase 3 投入 (7/1+): V20 production

- 7/1: V20 段階投入 (週末のみ、 上限 5,000円/日)
- 7/15: 順調なら投資額 増額 (週末 1万円/日 + 平日 5,000円/日)
- 8/1: V15 archive 判定 (1 か月並行運用後)

### Phase 4 (7-8 月): 動画解析 PoC

| 期間 | 内容 |
|------|------|
| 7/1-7/14 | データ蓄積 (JRA-VAN ネクスト + netkeiba 動画、 50 レース 1,500 動画) |
| 7/15-7/31 | YOLOv8 馬体検出 + DLC SuperAnimal 姿勢推定 動作確認 (zero-shot) |
| 8/1-8/15 | 時系列特徴量抽出 (stride / gait_symmetry / head_bobbing / ear_pos / posture / 5 件)、 fine-tune 必要なら DLC HORSE-10 ベース |
| 8/16-8/31 | V21 学習 (V20 + VIDEO_FEATURES) + WF 検証 |
| 9/1 | V21 投入判定 (WF AUC ≥ V20 + 0.005 / LIVE retro winner_top1 ≥ V20 + 1pt) |

### 投資保護 (絶対遵守)

- **5/9 V15 案B改 単独継続** (Session #38 NO-GO 確定後の唯一 path)
- **撤退ライン**: 累計 -50,000円 (現在 **¥-6,920**、 撤退余裕 **¥43,080**、 5/17 V15-audit-4 反映) ※ 旧値 +13,530 / +63,530 / +5,240 / +55,240 は drift、 5/16 P0-1 → 5/17 audit で 真値確定
- **取り返し禁止** (損切り後 翌日へ持ち越さない)
- **Phase 3-4 着手中も V15 production 完全不変保証**
- **★ formation record drift (5/17 data-audit-3) ★**: race-time 実通知 formation は ★ 永久喪失 ★ (race_auto_notify.py が独立予測 → Discord 送信のみ、 不揮発化なし)。 cumulative_results.csv の trio_bets_str は AM 8:00 morning prediction のみ。 5/18+ race_notify_log v2 で record 開始予定 (Sub-task C)

### 月額コスト + ROI 試算

| source | 月額 | 開始 |
|--------|------|------|
| netkeiba Premium | 4,500円 | 既存 |
| JRDB Advance | 約 2,000円 | 既存 |
| JV-Link (DataLab) | 2,090円 | 5/24 |
| JRA-VAN ネクスト (Phase 4 用) | +1,000円 | 7/1 |
| Colab Pro (Phase 4 GPU) | 1,178円 | 7/1 |
| **合計** | **約 10,768円/月** (7/1 以降) | — |

ROI 想定:
- V15 (現状): **98.34%** (5/17 V15-audit-4 反映、 戦略⑦込み 96.90% 戦略適用後 subset) → 月利 期待値 統計的にゼロ近辺 (CI [66.33%, 138.05%] 100% 含む) ※ 旧値「119.2% / 140% / 月利 2-3 万円」 は drift、 5/16 P0-1 → 5/17 V15-audit-4 で 真値確定
- V20 (7/1+): WF AUC 0.880-0.890 / 戦略⑦込み 145-150% 想定 → 月利 5-10 万円
- V21 (9/1+): V20 + 動画 features 中位想定で 145-150% → 月利 6-11 万円

→ 月額コスト 約 1万円は V20 以降の月利増分で十分回収。

### 📁 重要ファイル
- 1レース予測: `tools/predict_one_race.py`
- 戦略⑦適用: `tools/race_auto_notify.py` (4/27 修正)
- course_renovated: `tools/predict_core.py` 2061-2073 (4/27 修正)
- v16訓練ベース: `train/train_v15_master.py`, `train/retrain_v16.py`
- 訓練データ: `data/_v15_optuna_df_cache.pkl.gz` (104MB)
- TODO: `V16_TODO_NEXT_WEEK.md`

### 💾 バックアップ
- 4/27 修正前のバックアップ: `*.bak_20260427`
  - tools/race_auto_notify.py
  - tools/predict_core.py
  - train/features_v15_new.py
  - data/cumulative_results.csv
  - CLAUDE.md
