# TYB merge bug 修正 audit (Sub-task 6)

**実施日**: 2026-05-16
**audit script**: `tools/v21/_tyb_merge_audit.py`
**output JSON**: `data/v21/tyb_merge_audit.json`
**V15 production 影響**: 0% (read-only audit、 commit/push なし、 親集中)
**input data**:
- `data/_v15_optuna_df_cache.pkl.gz` (527,280 rows × 232 cols、 145 features、 read-only)
- `data/jrdb_tyb.csv` (550,115 rows、 read-only)
- `keiba_model_v15_central.pkl.gz` / `keiba_model_v15_central_live.pkl.gz` (read-only)
- `data/cumulative_results.csv` / `data/daily_predictions/20260516.csv` (read-only)

---

## 0. 結論 (★ TL;DR ★)

| 項目 | 結果 |
|------|------|
| **真の root cause** | **(b) 学習 pipeline に TYB merge 関数が 存在しない**。 `train/train_v135_ft_transformer.py:build_v134_dataframe()` は KYI / SED / CHA / JO / KTA / ZE / SR / KKA / OZ 等 merge は 呼ぶが、 **TYB merge は 一切 無い**。 `fill_v141_defaults()` が 5 columns を constant default で 埋めるだけ。 race_id format 不一致 (10-digit JV vs 12-digit nk) も 副因として 存在するが、 そもそも merge 自体が無いので format 違いは 二次的問題。 |
| **修正後 推定 +AUC delta** | **0** (V15 model retrain なしの場合、 ★ effectively zero ★ ) — V15 model.num_feature()=145、 predict_core が `X[:, :145]` で slice するため 追加 5 columns は **必ず truncate される**。 retrain 必須。 |
| **shadow merge match rate** | **100.0%** (10→12-digit 変換後、 cache 527,280 rows 全て TYB に 対応 key あり) |
| **5 TYB features 中 LEAK 疑い** | **3/5** が LEAK 疑い: `odds_idx` (corr +0.4214 ≈ \|popularity\| 0.4242)、 `paddock_idx` (corr +0.3539、 17:00 JST 配信 delivery leak)、 `sogo_idx` (corr +0.2573、 odds+padock 合成) |
| **5 TYB features 中 safe 候補** | **2/5**: `body_code` (corr +0.0121)、 `demeanor_code` (corr +0.0041) — ただし 信号 極小、 +AUC delta は 0 に近い |
| **推奨案** | **★ case D (シャドウ用 V21 専用 features file)、 V15 production 不変、 P0-3 TYB leak audit (5-4) と 整合 ★**。 V15 retrain は LEAK 解消が confirmed まで NO-GO。 |
| **5/18+ 実装** | shadow first、 paper trading 30R で validate、 leak audit 通過 + delivery timing 解決後 のみ V20/V21 で 採用検討 |

---

## 1. race_id format audit 結果

| source | length | 例 |
|--------|--------|-----|
| `data/jrdb_tyb.csv` | **12 digit** | `201506010101` (2015 + 06_train + 01_kai + 01_nichi + 01_race) |
| `data/cumulative_results.csv` | **12 digit** | `202643050511` |
| `data/daily_predictions/20260516.csv` | **12 digit** | `202608030701` |
| `data/_v15_optuna_df_cache.pkl.gz` | **10 digit (JV format)** | `0915150804` (course_2 + year_2 + kai_1 + nichi_1 + race_2 + extra_2) |

**format mismatch 確認**:
- V15 cache は JV native format (10 digit、 `data/jra_races_full.csv` 由来)
- 他は全部 netkeiba format (12 digit、 `20` + YY + CO + KK + NN + RR)
- `_build_nk_race_id_from_jv()` が変換 logic、 既に `tools/jrdb_features.py:300` に 実装済

★ ただし `train/train_v135_ft_transformer.py:build_v134_dataframe()` は TYB merge を call しないので、 format 不一致は 副次的 ★

---

## 2. V15 cache TYB columns audit (bug の証拠)

| column | unique | mean | sample[0] | features list? |
|--------|--------|------|-----------|----------------|
| `jrdb_paddock_idx` | **1** | 50.0 | 50.0 | ❌ NOT in V15 features (Pattern A) |
| `jrdb_odds_idx` | **1** | 50.0 | 50.0 | ❌ NOT in V15 features |
| `jrdb_live_composite_idx` | **1** | 50.0 | 50.0 | ❌ NOT in V15 features |
| `jrdb_body_code` | **1** | 4.0 | 4 | ❌ NOT in V15 features |
| `jrdb_demeanor_code` | **1** | 2.0 | 2 | ❌ NOT in V15 features |

★ **全 5 columns は cache 内に 存在するが、 unique=1 (定数 default)** ★ — `fill_v141_defaults()` (`train/train_v141_paci_tierA.py:146`) で `JRDB_DEFAULTS` (`tools/jrdb_features.py:114`) の値が 一律 fill される。

参考: 同じ cache 内 **他 JRDB 40 features は健全** (例: `jrdb_idm` unique=711、 `jrdb_ze_idm_avg` unique=7637)。 これらは `merge_jrdb_train_features()` (KYI) / `merge_ze_features()` 等で 正常に merge されている。 ★ TYB だけ merge function が無い ★

---

## 3. 真の root cause 特定 (検証順序 a-f)

| # | 仮説 | 検証 | 結果 |
|---|------|------|------|
| (a) | race_id format 不一致 | format audit | 確かに 10 vs 12-digit、 ただし副因 |
| (b) | **merge logic bug (関数なし)** | `train/train_v135_ft_transformer.py:486-527 build_v134_dataframe` grep | **★ TYB merge 関数 unfound、 関数自体が無い ★** |
| (c) | 列名 mismatch | `tools/jrdb_features.py:215 extract_tyb_features` 確認 | 関数は 存在するが、 train pipeline から 呼ばれていない |
| (d) | dtype mismatch | shadow merge で 確認 | 影響なし (`_uma` int で OK) |
| (e) | umaban mismatch | shadow merge で 確認 | 100% match (df_keys 527,280、 intersection 527,280) |
| (f) | cache build 時点で jrdb_tyb 未存在 | jrdb_tyb.csv は 2015-2026 まで 揃う | 否定 |

**確定 root cause = (b)** :
- `build_v134_dataframe()` (v13.5b 系の base) は `merge_jrdb_train_features` を call して KYI/SED を merge するが、 TYB merge は **呼ばれない**
- `tools/jrdb_features.py:215 extract_tyb_features()` 関数は **predict-time のみ** (`per-race lookup`、 line 543) で 使用
- training cache build 時に TYB を統合する path が 存在しない

**副因 (a)** : 仮に TYB merge を training に 追加する場合、 必ず `_build_nk_race_id_from_jv()` で 10→12 変換 が 必要。

---

## 4. shadow merge logic 試行結果

### 4-1. 全体 fill rate

| metric | bug state (cache 内) | shadow merge 修正版 |
|--------|---------------------|---------------------|
| `jrdb_paddock_idx` unique | 1 | **25** |
| `jrdb_paddock_idx` nonzero rate | 0% (全 50.0 = 1 値) | 36.1% (実 data、 残り 64% は 0 = JRDB の "未観測" zero) |
| `jrdb_odds_idx` unique | 1 | **7** |
| `jrdb_odds_idx` nonzero rate | 0% | 35.8% |
| `jrdb_live_composite_idx` (sogo) unique | 1 | **897** |
| `jrdb_live_composite_idx` nonzero rate | 0% | 100.0% |
| `jrdb_body_code` (batai) unique | 1 | **8** |
| `jrdb_body_code` nonzero rate | 0% | 38.4% |
| `jrdb_demeanor_code` (kehai) unique | 1 | **9** |
| `jrdb_demeanor_code` nonzero rate | 0% | 38.4% |

★ 修正版で merge 自体は 完全成功 (key match 100%)、 ただし JRDB TYB 内部の zero rate (パドック未観察 / odds 未計算) は data quality 由来 ★

### 4-2. corr_target audit (LEAK detection)

| feature | corr_target (filled) | 参考 |
|---------|----------------------|------|
| **jrdb_odds_idx** | **+0.4214** | ⚠ `popularity` 単独 corr = -0.4242、 ★ ほぼ同等 = odds-based / LEAK 疑い ★ |
| **jrdb_paddock_idx** | **+0.3539** | ⚠ Sub-task 5-4 で delivery 17:00 JST 配信 leak 確定 |
| **jrdb_live_composite_idx** | **+0.2573** | ⚠ odds + padock 合成、 部分 leak |
| **jrdb_body_code** | +0.0121 | ✅ low signal、 safe |
| **jrdb_demeanor_code** | +0.0041 | ✅ low signal、 safe |

参考 (既存 V15 features):
- `popularity`: corr_target = **-0.4242** (LEAK で 除外済)
- `jrdb_idm`: corr_target = +0.1569
- `jrdb_composite_idx`: corr_target = +0.2124
- `jrdb_ze_idm_avg`: corr_target = +0.2719

★ `jrdb_odds_idx` corr +0.4214 ≈ \|popularity\| 0.4242 → 限りなく odds_log 同等の LEAK 疑い ★

### 4-3. per-year coverage

| year | rows | padock !=0 | odds_idx !=0 | sogo !=0 | batai !=0 |
|------|------|-----------|--------------|----------|-----------|
| 2015 | 49,610 | 32.6% | 34.5% | 99.8% | 100.0% |
| 2020 | 47,882 | 35.9% | 35.7% | 99.8% | 100.0% |
| 2023 | 47,274 | 38.9% | 36.2% | 99.8% | 100.0% |
| 2025 | 47,497 | 38.5% | 36.0% | 99.9% | 99.9% |

→ batai/kehai/sogo は 全期間 ほぼ 100% data あり、 padock/odds_idx は 30-40%。

---

## 5. 過去 100R shadow score 比較 (★ 重要警告 ★)

**実施せず**: 以下理由。

### 5-1. V15 model architecture の制約

| 項目 | 値 |
|------|----|
| V15 Pattern A model `num_feature()` | **145** |
| V15 Pattern A `features` list 長 | 145 |
| V15 Pattern B model `num_feature()` | **145** (同じ LGB object を share) |
| V15 Pattern B `features` list 長 | 150 (= 145 + 5 TYB live) |
| Pattern A vs Pattern B AUC | 共に 0.8939 (★ 同一 ★) |
| Pattern A model is Pattern B model | False (新 obj だが weights 完全同一、 `auc` 完全一致) |

**`tools/predict_core.py:2160-2163`** に **明示的 truncate** が存在:

```python
# LGB/XGBは学習時の特徴量数に合わせる（124列）
# pkl.gzのfeaturesリスト(129)にはPattern B TYB特徴量(5)が含まれるが
# LGB/XGBは先頭124列で学習されている
n_lgb_features = use_model.num_feature() if hasattr(use_model, 'num_feature') else X_full.shape[1]
X = X_full[:, :n_lgb_features]
```

→ V15 でも同じ slice、 TYB 5 columns は X 末尾に position するため **必ず truncate**。

### 5-2. 帰結

★ **V15 model を そのまま 使い続ける限り、 merge bug を 修正しても +AUC delta = 0** ★ (effectively zero)

shadow score 試算 を 行っても 結果は 全て 同一 (TYB 値が model に届かない)。 honest report のため 試算は skip。

### 5-3. ★ 重要 ★ 「修正後 効果あり」 シナリオ

merge bug 修正で uplift を得るには **V15 (もしくは V21 / V22) の retrain 必須**。 ただし:
- `jrdb_odds_idx`: corr +0.4214 → ★ LEAK 確定的 ★ 採用不可
- `jrdb_paddock_idx`: corr +0.3539 → P0-3 (5-4) で delivery 17:00 JST = race 後 配信 confirmed → live 不可、 retrospective backtest のみ
- `jrdb_live_composite_idx`: corr +0.2573 → odds 含む → 採用不可
- `jrdb_body_code` / `jrdb_demeanor_code`: corr ~0.01 → 採用しても 効果 ≈ 0

★ **5 features 中 採用候補 = 0** ★ (safe な body/demeanor は信号 極小、 高信号 3 features は LEAK)

---

## 6. 修正案 A-D 比較

| case | 内容 | cost | risk | 期待 +AUC | recommend |
|------|------|------|------|-----------|-----------|
| **A** | V15 cache rebuild + V15 retrain で TYB 5 features 統合 | 高 (2-3 日、 ensemble 4-model 再学習) | ★ 致命的 (`odds_idx` LEAK 採用すると LIVE 大破綻) ★ | **+0.000** (LEAK 除外後)、 **−0.05+** (LEAK 採用したら 致命的) | ❌ |
| **B** | `_build_v134_dataframe()` に TYB merge step 追加するだけ (model 不変) | 低 (1 日) | 低 (cache が新規 rebuild されないと反映なし、 仮に rebuild しても model が `[:, :145]` で truncate) | **+0.000** (model 不変なら effective zero) | ❌ |
| **C** | 両 format 受容 wrapper + TYB merge 追加 (B + format converter) | 中 (1-2 日) | B と同じ | **+0.000** | ❌ |
| **D** | ★ shadow features file (V21 専用、 V15 production 触らず) ★ | 中 (1-2 日) | ★ V15 production 完全不変、 影響ゼロ ★ | V21 retrain で safe な 2 features 採用 → +0.000 〜 +0.001 | ✅ |

### case D 詳細 (★ 推奨 ★)

1. **新規 file** `data/v21/_tyb_shadow_features.parquet` を 別途 build
2. `tools/v21/build_tyb_shadow.py` で `jrdb_tyb.csv` × cache の shadow merge
3. ★ 学習時の leak audit ★ : 各 feature 単独 add で 5CV AUC delta が ±0.001 以内なら safe (corr_target も併用判定)
4. V21 / V22 学習時に opt-in (V15 は 触らない)
5. live deployment は P0-3 (5-4) の delivery timing 問題 解決 が prerequisite

---

## 7. 5/18+ 実装 plan

### 7-1. 推奨 action items

| 優先 | task | 担当 (script) | 期限 |
|------|------|---------------|------|
| ★ 最優先 ★ | **このまま V15 を 触らない** (Phase 5 直行) | — | 即 |
| 高 | 既存 docs/TYB_LEAK_AUDIT_2026_05_16.md (Sub-task 5-4) の verdict を 再確認 | (read-only review) | 5/17 |
| 中 | case D shadow features file build (V21 学習 候補 用) | 新規 `tools/v21/build_tyb_shadow.py` | 5/18-5/19 |
| 中 | shadow 5CV AUC test (cache + shadow features、 V21 設計内で 評価) | 新規 `tools/v21/eval_tyb_shadow_cv.py` | 5/20 |
| 低 | `jrdb_body_code` / `jrdb_demeanor_code` 単独 採用 検討 (corr 0.01 → 効果 0 期待) | V21 学習時 1 fold ablation | 5/22+ |

### 7-2. 絶対遵守 (V15 production 不変保証)

- `keiba_model_v15_central*.pkl.gz` の **再 save 一切なし**
- `data/_v15_optuna_df_cache.pkl.gz` の **再 build 一切なし** (V15 retrain なし)
- `tools/predict_core.py` / `tools/daily_predict.py` / `tools/race_auto_notify.py` / `app.py` 一切 変更しない
- `train/train_v135_ft_transformer.py` (`build_v134_dataframe`) に TYB merge 追加しない (V15 path 再 trigger 防止)
- 新規 shadow file は `data/v21/_tyb_*` prefix で 分離

### 7-3. 回帰テスト追加項目 (V21 学習時)

- shadow features add 前後で V15 model 出力が **完全一致** (V15 path 不変 confirm)
- shadow features の corr_target が **`jrdb_body_code`/`jrdb_demeanor_code` のみ < 0.05** を satisfy
- WF 6-fold で **どの 1 fold も delta < -0.001** にならない (一個でも悪化したら NO-GO)
- LEAK audit: shadow merge 後の 5CV と V21 base 5CV の **train-test gap が < 0.05**

### 7-4. paper shadow 30R eval

- 5/24-6/14 の 3 週間 (12 開催日、 36 R 想定) で shadow score をログ
- bug 状態 (V15 production) vs shadow 修正版 で top1 / top3 hit rate 比較
- delta が ±2pt 以内 (= effective zero) を 期待、 これと違ったら 再調査

---

## 8. 重要 finding sub-summary

1. ★ **V15 model.num_feature()=145、 saved Pattern B features list=150、 5 feature gap が予期されて truncate される** ★ — `tools/predict_core.py:2162` で 明示 slice。 Pattern B は実質 Pattern A と完全同一。
2. ★ **`build_v134_dataframe()` の merge call が KYI/SED/CHA/JO/KTA/ZE/SR/KKA/OZ を呼ぶが TYB merge は 不在** ★ — bug の本体は **merge 関数 absence**。
3. ★ **shadow merge match rate 100%** ★ — `_build_nk_race_id_from_jv()` 既存 logic で完全 join 可能。 format 不一致は副因。
4. ★ **`jrdb_odds_idx` corr_target +0.4214 ≈ \|popularity\| 0.4242** ★ — odds-based、 LEAK 確定的。 採用したら V15 と同等 critical level の 事故。
5. ★ **5 TYB features 中 採用候補 = 0** ★ (LEAK 3 件 / 低信号 2 件)、 修正後 V15 retrain しても 効果 ≈ 0。
6. ★ **Sub-task 5-4 の P0-3 audit と integrate** ★ — 既に delivery 17:00 JST post-race 配信 確定済、 live deploy は publish_monitor 復旧 が prerequisite。

---

## 9. honest verdict (★ caveman summary ★)

- bug 真因: merge 関数なし (race_id 違いは second)
- 直しても効果 0 (model truncate)
- model retrain したら LEAK で死ぬ (odds_idx)
- 残り 2 safe features は信号ゼロ
- ★ 結論: V15 触るな、 V21 学習で shadow eval、 LIVE 投入 は delivery 解決後 ★

---

## 10. 出力ファイル

- `docs/TYB_MERGE_BUG_AUDIT_2026_05_16.md` (本文書)
- `tools/v21/_tyb_merge_audit.py` (audit script、 read-only)
- `data/v21/tyb_merge_audit.json` (生 metrics 全量)
