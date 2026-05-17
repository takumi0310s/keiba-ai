# V15-audit-1: V15 model file 構造 真値 audit

Date: 2026-05-17
Mode: ★ read-only audit (V15 production / v15.2 training 完全不変) ★
Source: `keiba_model_v15_central.pkl.gz` / `keiba_model_v15_central_live.pkl.gz` + `data/v15_master_report.json` + `train/train_v15_master.py`

---

## 0. 結論 (★ 核心 ★)

| 項目 | CLAUDE.md 記述 | 真値 | 整合? |
|------|----------------|------|-------|
| ensemble architecture | 4-model (LGB+XGB+FT+IR) | **2-model (LGB + XGB) のみ** | ❌ |
| ensemble_weights | (記述なし) | `{lgb: 0.5036, xgb: 0.4964, mlp: 0}` | — |
| FT-Transformer / IntraRace Attention | 含む (4-ensemble) | **.pkl.gz に存在しない** | ❌ |
| MLP | (記述なし) | key 存在 / 値 None / weight 0 = dead | — |
| stored AUC 0.8939 source | "WF AUC 0.8939" | **LGB train-set AUC (overfit metric)** | ❌ |
| 真の WF mean grid AUC | (記述なし) | **0.8858** (`v15_master_report.json` grid_mean) | — |
| 145 features (Pattern A) | "150 特徴量" | 145 features (booster) / Pattern A 145 / Pattern B 150 (5 件は inference 段で未投入) | ⚠ |
| Pattern B (live) | "132 features" or "150" 記述あり | features list 150 だが booster 145 (Pattern B 専用 5 件は booster 未学習) | ⚠ |
| T1 audit (145 features) | — | ★ T1 (commit `542c2c0b`) と完全整合 ★ | ✅ |

★ V15 production の真の architecture = **LGB+XGB 2-model** ★
★ CLAUDE.md の「4-ensemble」「WF AUC 0.8939」記述は誤り (要訂正) ★
★ T1 145 features audit は真の booster 構造と完全一致 ★

---

## 1. model file 特定

| 用途 | file | size | mtime |
|------|------|------|-------|
| Pattern B (production live) | `keiba_model_v15_central_live.pkl.gz` | 2,099,610 B (2.00 MB) | 2026-04-08 23:32:38 |
| Pattern A (リークフリー評価用) | `keiba_model_v15_central.pkl.gz` | 2,099,552 B (2.00 MB) | 2026-04-08 23:32:37 |

predict_core.py L477-478 で確認:
- 第1優先: `keiba_model_v15_central_live.pkl.gz` (is_live=True, Pattern B)
- 第2優先: `keiba_model_v15_central.pkl.gz` (Pattern A, fallback)

両 file は同一の booster pair を共有 (Pattern B は features list と一部 metadata のみ差分)。

---

## 2. model dict 構造

両 file 共通 keys (差分は metadata のみ):

| key | type | 値 / 詳細 |
|------|------|----------|
| `model` | `lightgbm.Booster` | LGB booster, num_trees=500, num_features=145 |
| `xgb_model` | `xgboost.Booster` | XGB booster, num_boosted_rounds=500, num_features=145 |
| `features` | `list` | A: 145 / B: 150 (Pattern B-only 5 件は booster 入力後 truncate) |
| `version` | `str` | A: `'v15'` / B: `'v15_live'` |
| `auc` | `float` | **0.8939485520467574** (= LGB train-set AUC、 後述) |
| `leak_free` | `bool` | A: True / B: False |
| `leak_pattern` | `str` | A: `'A'` / B: `'B'` |
| `leak_removed` | `list` (8 件) | `cond_surface, condition_enc, horse_weight, odds_log, weight_cat, weight_cat_dist, weight_change, weight_change_abs` |
| `sire_map` | `dict` | top-100 父馬 encoding (Shift-JIS 文字化け raw 状態) |
| `bms_map` | `dict` | top-100 母父 encoding |
| `n_top_encode` | `int` | 100 |
| `trained_at` | `str` | `'2026-04-08T23:32:37.533143'` |
| `model_type` | `str` | `'central'` |
| `mlp_model` | `NoneType` | **None** (dead key、 weight 0) |
| `mlp_scaler` | `NoneType` | **None** |
| `ensemble_weights` | `dict` | `{lgb: 0.5036339408713449, xgb: 0.496366059128655, mlp: 0}` |
| `course_map` | `dict` | 10 競馬場 encoding |
| `is_live` | `bool` | Pattern B のみ存在 (True) |

★ 含まれない key (検証済): `ft_model_state`, `ft_model_config`, `ft_scaler_mean`, `ft_scaler_scale`, `cb_model`, `ir_model` ★
→ predict_core.py L2173-2178 はこれらを optional ensemble component として参照するが、 V15 .pkl.gz には全て存在しない (`has_ensemble` 判定では xgb_m のみで True、 LGB+XGB 2-model 推論のみ実行)。

---

## 3. 各 component 詳細

### 3.1 LightGBM (`model`)

- type: `lightgbm.Booster`
- num_trees: **500**
- num_features: **145**
- best_iteration: 0 (early stop 未使用、 500 round 固定)
- current_iteration: 500
- feature_name(): `['Column_0', 'Column_1', ..., 'Column_144']`
  - ★ Booster 内では anonymized `Column_N` ★ (numpy.ndarray から train、 column 名 propagate なし)
  - features list の順序が真の feature 名と対応

LGB top-10 gain importance (Pattern B 推論、 anonymized index):

| Column | gain | features list 該当 |
|--------|------|----|
| Column_129 | 314,750 | `paci_jockey_exp_3rd` |
| Column_130 | 295,197 | `paci_ninki_idx` |
| Column_128 | 266,346 | `paci_jockey_exp_wr` |
| Column_107 | 175,971 | `jrdb_ze_idm_avg` |
| Column_43 | 96,546 | `training_time_filled` |
| Column_45 | 53,711 | `training_per_dist` |
| Column_108 | 36,674 | `jrdb_ze_ten_avg` |
| Column_74 | 35,433 | `jrdb_idm` |
| Column_83 | 29,243 | `jrdb_class_code` |
| Column_48 | 21,692 | `horse_career_wr` |

### 3.2 XGBoost (`xgb_model`)

- type: `xgboost.Booster`
- num_boosted_rounds: **500**
- num_features: **145**
- best_iteration: **N/A** (early stop 未使用 → 例外発生)
- feature_names: **None** (anonymized `f0`〜`f144`)

XGB top-10 gain importance:

| feature | gain | features list 該当 |
|---------|------|----|
| f128 | 1515.46 | `paci_jockey_exp_wr` |
| f129 | 1466.58 | `paci_jockey_exp_3rd` |
| f130 | 630.17 | `paci_ninki_idx` |
| f141 | 582.52 | `paci_idm_mark` |
| f37 | 194.47 | `age_group` |
| f107 | 92.31 | `jrdb_ze_idm_avg` |
| f125 | 83.03 | `paci_goal_rank` |
| f119 | 72.03 | `pop_rank_change` |
| f43 | 71.60 | `training_time_filled` |
| f1 | 66.81 | `age` |

LGB / XGB は **完全に同じ 145 features (同順序)** で学習されており、 ensemble_weights `lgb≈0.504, xgb≈0.496` で加重平均推論。

### 3.3 FT-Transformer (FT) / IntraRace Attention (IR)

★ V15 production .pkl.gz には **存在しない** ★

検証:
```python
'ft_model_state' in m  → False
'ir_model' in m        → False
'cb_model' in m        → False
```

但し `data/v15_master_report.json` の WF 評価では FT/IR を含む 4-model grid を実行している (3.5 章参照)。

### 3.4 MLP (dead)

- `mlp_model`: None
- `mlp_scaler`: None
- `ensemble_weights['mlp']`: **0**

→ training 時に key が ensemble_weights に残った dead key、 推論では完全に無視される。

### 3.5 ★ 重要: WF 評価時の 4-model と production .pkl の乖離 ★

`data/v15_master_report.json` (= train_v15_master.py `run_all_in` 出力):
- WF fold ごとに **4 model 全て (LGB+XGB+FT+IR) を学習・評価**
- grid_weights を年ごとに最適化 (typically `[lgb=0.20, xgb=0.25-0.30, ft=0.10-0.15, ir=0.40]`)
- 年別 grid AUC (= WF 評価実値):

| year | lgb_auc | xgb_auc | ft_auc | ir_auc | **grid_auc** | grid_weights (lgb,xgb,ft,ir) |
|------|---------|---------|--------|--------|--------------|-------------------------------|
| 2021 | 0.8643 | 0.8669 | 0.8627 | 0.8738 | **0.8836** | [0.20, 0.30, 0.10, 0.40] |
| 2022 | 0.8673 | 0.8684 | 0.8656 | 0.8755 | **0.8841** | [0.20, 0.30, 0.10, 0.40] |
| 2023 | 0.8689 | 0.8698 | 0.8676 | 0.8772 | **0.8860** | [0.20, 0.25, 0.15, 0.40] |
| 2024 | 0.8703 | 0.8720 | 0.8705 | 0.8800 | **0.8887** | [0.20, 0.25, 0.15, 0.40] |
| 2025 | 0.8686 | 0.8700 | 0.8684 | 0.8794 | **0.8868** | [0.20, 0.30, 0.10, 0.40] |
| **平均** | 0.8679 | 0.8694 | 0.8669 | 0.8772 | **0.88584** | — |

★ WF 評価では 4-model grid (IR 重み 0.40 が dominant) で AUC 0.8858 ★

しかし train_v15_master.py L573-616 で **production .pkl に保存されるのは LGB+XGB 2-model のみ** で、 FT/IR は学習 + WF 評価専用、 production 推論では使われない。
→ ensemble_weights も `{lgb, xgb, mlp}` の 3 key dict に簡略化 (FT/IR は dropped)。

---

## 4. features list (145/150 件)

### 4.1 Pattern A (145 件、 booster 学習対象)

T1 audit (commit `542c2c0b`, `data/T1_features_audit_2026_05_17.json`) と **完全一致** (145/145 names match)。

代表的 feature group (順序保持):
- 基本 (0-8): `weight_carry, age, distance, course_enc, surface_enc, sex_enc, num_horses_val, horse_num, bracket`
- 騎手 (9-11): `jockey_wr_calc, jockey_course_wr_calc, trainer_top3_calc`
- 前走ラグ (12-27): `prev_finish, prev_last3f, prev_pass4, prev_prize, ..., rest_days, rest_category`
- 集計 (28-29 含): `sire_enc, bms_enc, dist_cat, age_sex, season, ...`
- V9.2/V9.3 拡張 (43-66): `training_time_filled, has_training, ..., frame_course_dist_wr`
- V12 (67-72): `index_max_filled, ..., training_intensity_enc`
- V13 拡張 (73): `pci`
- JRDB (74-113): `jrdb_idm, ..., jrdb_bms_rensho_avg` (40 features)
- v14.x / v15 新規 (114-144): `stable_comment_score, oz_tansho_base_log, ..., gaisha_rank, paci_sogo_mark, paci_idm_mark, paci_jockey_mark, paci_train_mark`

### 4.2 Pattern B (live) 追加 5 件

Pattern A 145 件に加え:
- `jrdb_paddock_idx`
- `jrdb_odds_idx`
- `jrdb_live_composite_idx`
- `jrdb_body_code`
- `jrdb_demeanor_code`

★ これら 5 件は features list には存在するが、 LGB/XGB booster は **145 features で学習されており、 推論時に truncate される (predict_core.py L2162-2163 `X = X_full[:, :n_lgb_features]`)** ★
→ 実質的に Pattern B 推論でもこの 5 件は使われていない (booster 入力に届かない)。

---

## 5. metadata + 学習履歴

| 項目 | 値 |
|------|----|
| trained_at | 2026-04-08T23:32:37.533143 |
| 学習 script | `train/train_v15_master.py` |
| 学習 data | `data/_v15_optuna_df_cache.pkl.gz` (約 104 MB) |
| WF fold 構成 | year 2021-2025 (5 fold)、 baseline=v14.1 WF AUC 0.8856 |
| 学習 round | LGB 500 / XGB 500 (early stop なし、 全 round 固定) |
| baseline AUC (v14.1) | 0.8856 (記載値) |
| **WF mean grid AUC (4-model)** | **0.8858** (delta +0.000236, all-in adopted) |
| stored `auc` (.pkl) | **0.8939** (= LGB train-set AUC、 後述) |

### 5.1 ★ stored AUC 0.8939 の真の source ★

train_v15_master.py L575-595 (Pattern A 保存箇所):
```python
lgb_model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500)
lgb_auc = roc_auc_score(y_all, lgb_model.predict(X_all))  # ← train set 自己評価
xgb_auc = roc_auc_score(y_all, xgb_model.predict(dxtrain))  # ← train set 自己評価
...
pkl_a = {
    ...
    'auc': lgb_auc,  # ← .pkl の auc field は LGB train AUC
    'ensemble_weights': {
        'lgb': lgb_auc / (lgb_auc + xgb_auc),
        'xgb': xgb_auc / (lgb_auc + xgb_auc),
        'mlp': 0,
    },
    ...
}
```

数値整合性 verify:
- stored `auc` = 0.893949 = LGB train AUC
- `ensemble_weights['lgb']` = 0.5036339 = lgb_auc / (lgb_auc + xgb_auc)
- 逆算: xgb_auc = 0.893949 × (0.4964 / 0.5036) = **0.881048** = XGB train AUC

⇒ **`auc=0.8939` は train-set self-evaluation 値 (overfit metric)**、 WF mean ではない。
⇒ 真の汎化性能 (WF mean grid) = **0.8858** (`v15_master_report.json`)。

### 5.2 yearly train_auc vs grid_auc gap (overfit 度合い)

| year | train_auc | grid_auc | gap |
|------|-----------|----------|------|
| 2021 | 0.8952 | 0.8836 | 0.0309 |
| 2022 | 0.9005 | 0.8841 | 0.0331 |
| 2023 | 0.9048 | 0.8860 | 0.0360 |
| 2024 | 0.8969 | 0.8887 | 0.0266 |
| 2025 | 0.8975 | 0.8868 | 0.0289 |

gap 0.025-0.036、 健全な overfit 範囲。 stored `auc=0.8939` は production-fold (全データ retrain) の train AUC 相当で、 fold ごとの train_auc 0.8952-0.9048 とも整合する。

---

## 6. CLAUDE.md 整合 audit (★ 要訂正リスト ★)

| CLAUDE.md 記述箇所 | 真値 | 整合? | 訂正案 |
|--------------------|------|-------|--------|
| "150 特徴量" | Pattern A=145 / Pattern B features list=150 (うち 5 は booster 未投入) | ⚠ | "145 特徴量 (booster 学習)、 Pattern B features list 150 (5 件は inference 段で truncate)" |
| "AUC 0.8939" (現行モデル V15) | LGB train-set AUC = 0.8939 / **WF mean grid AUC = 0.8858** | ❌ | "WF mean grid AUC 0.8858 (4-model WF評価)、 production .pkl stored auc=0.8939 は LGB train self-eval" |
| "4-model grid ensemble" / "LGB+XGB+FT+IR" (v13.5b 説明流用) | **production .pkl は LGB+XGB 2-model のみ**、 FT/IR は WF 評価専用で .pkl 保存なし | ❌ | "V15 production .pkl: LGB+XGB 2-model (weight 0.504/0.496)、 FT/IR は WF 評価専用、 推論未使用" |
| "アンサンブル構成 (v13.5b、 現行): 4モデル Grid Ensemble" | V15 推論は 2-model | ❌ | "V15 production 推論: LGB+XGB 2-model、 v13.5b 4-model 記述は v13.5b 時代の遺物" |
| "LGB ~56%, XGB ~44%" (v12 以前) | V15: LGB 50.36% / XGB 49.64% | — | (V15 では更新済の重みに) |
| 145 features integrity (T1 audit) | T1 (commit 542c2c0b) と完全整合 | ✅ | — |
| `leak_removed` 8 features | `cond_surface, condition_enc, horse_weight, odds_log, weight_cat, weight_cat_dist, weight_change, weight_change_abs` (CLAUDE.md と一致) | ✅ | — |
| `mlp_model`, `mlp_scaler`, ensemble_weights['mlp']=0 | dead key、 production 影響なし | — | (記述なしのまま OK) |

---

## 7. predict_core.py 推論 path 検証

`tools/predict_core.py` L2162-2243 抜粋分析:

1. `X = X_full[:, :n_lgb_features]` (L2162-2163) → 145 列に truncate (Pattern B の 5 extra も剥がす)
2. `ai_scores = use_model.predict(X)` → LGB 予測
3. `xgb_m = model_data.get('xgb_model')` → XGB 取得 (V15 では存在)
4. `cb_m = model_data.get('cb_model')` → None (V15 に存在しない)
5. `ft_state = model_data.get('ft_model_state')` → None (V15 に存在しない)
6. `has_ensemble = (xgb_m is not None or ...)` → True (XGB のみで True)
7. ensemble 加重平均: `w_lgb=0.5036 * lgb_pred + w_xgb=0.4964 * xgb_pred` (FT/IR は code path 入らず skip)

→ 実際の production 推論 = **LGB+XGB 2-model 加重平均** で確定。

---

## 8. V15 production / v15.2 training 不変保証 ✅

本 audit で実施したのは以下のみ:
- `keiba_model_v15_central.pkl.gz` read-only load (`gzip.open(..., 'rb')` + `pickle.load`)
- `keiba_model_v15_central_live.pkl.gz` read-only load
- `data/v15_master_report.json` read-only
- `train/train_v15_master.py` 読み取り
- `tools/predict_core.py` 読み取り
- 新規 docs 作成 (本 file のみ)

**変更なし**:
- 🟢 V15 .pkl.gz: 一切改変なし
- 🟢 predict_core / daily_predict / race_auto_notify / app.py: 一切改変なし
- 🟢 v15.2 training process (PID 23528 想定): 一切干渉なし
- 🟢 git commit/push: 行わない (★ 親集中 ★)

---

## 9. 主要 finding まとめ (★ 親への報告用 ★)

1. ★ V15 真 architecture = **LGB+XGB 2-model 加重平均 (lgb=0.5036, xgb=0.4964)** ★
2. ★ FT-Transformer / IntraRace Attention は **.pkl 内に存在せず**、 推論で使われていない ★
3. ★ MLP は key が残存するが値 None / weight 0 = dead ★
4. ★ stored `auc=0.8939` の真の source = **LGB train-set AUC (overfit self-eval)**、 WF 評価値ではない ★
5. ★ 真の WF mean grid AUC = **0.8858** (`v15_master_report.json` 4-model grid 5-fold mean) ★
6. ★ 145 features (Pattern A booster 学習) = T1 audit (commit 542c2c0b) と完全整合 ★
7. ★ Pattern B features list 150 のうち 5 件 (`jrdb_paddock_idx, jrdb_odds_idx, jrdb_live_composite_idx, jrdb_body_code, jrdb_demeanor_code`) は booster 未学習で inference 段 truncate される ★
8. ★ CLAUDE.md の「4-ensemble」「WF AUC 0.8939」記述は V13.5b 時代の遺物 → V15 では訂正が必要 ★
