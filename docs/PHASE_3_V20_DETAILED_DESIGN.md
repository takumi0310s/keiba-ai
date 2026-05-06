# Phase 3 V20 統合モデル 詳細設計 (Session #36 E)

**作成**: 2026-05-07 深夜 (Session #36 E、就寝中マラソン)
**期間**: 6/9 (月) - 6/30 (月) の 3 週間
**目的**: JRA + NAR 統合 single model、共通 features 80% で運用効率化

---

## 1. 目的 + 動機

### 1.1 既存 model の問題

| model | features | AUC | 課題 |
|-------|---------|-----|------|
| V15 (JRA) | 150 | 0.8939 | 軸 top3 率 -16pt gap (BT 57% → 本番 41%) |
| V18/V19 (JRA、単/複) | 190 | 0.8954/0.8787 | distribution shift 27.7x、winner_top1 -13.3pt |
| NAR v4 | 22 | 0.8145 (0.8519 OOS) | 学習 data 1 年 stale (2024-03〜2025-05) |

→ 二重管理コスト、重複 features 多数、保守困難。

### 1.2 V20 の statement

**JRA + NAR 統合 single model**:
- 共通 features 80% (距離 / 馬場 / 性別 / 年齢 / 騎手 / 調教師 / 血統 / 過去成績 / 当日情報)
- 競馬場 specific features 20% (`is_nar` フラグ + JRA-only / NAR-only features)
- 学習 data: JRA 約 50 万 horses (V15 base) + NAR 約 5 万 horses (NAR v4 base) = 55 万
- target: 1 着率 (V18 と同) / 3 着内率 (V19 + V15 と同)

**期待効果**:
- 二重管理コスト削減 (model 2 → 1)
- NAR で学習した知見が JRA に転用 (汎化向上)
- features 80% 共通で保守工数削減

---

## 2. アーキテクチャ

### 2.1 4-model ensemble (V15.1 と継承)

```
LGB Booster (主、AUC ~0.88)
+ XGB Booster (副、AUC ~0.86)
+ FT-Transformer (4-layer Transformer、AUC ~0.85)
+ IntraRace Attention (race 内相対比較、AUC ~0.85)
↓
Grid Search 重み付き ensemble
最終 AUC target ≥ 0.88 (JRA subset、V15 0.8858 比)
最終 AUC target ≥ 0.83 (NAR subset、NAR v4 0.8145 比)
```

### 2.2 features 設計

#### 共通 features (推定 80 件)

```python
COMMON_FEATURES = {
    'basic': [
        'distance', 'surface_enc', 'sex_enc', 'age', 'weight_carry',
        'horse_num', 'bracket', 'num_horses', 'horse_num_ratio',
        'bracket_pos', 'carry_diff', 'is_nar',  # ← 切替フラグ
    ],
    'past_finish': [
        'prev_finish', 'prev2_finish', 'prev3_finish',
        'prev_last3f', 'prev2_last3f', 'prev_pass4', 'prev_prize',
        'avg_finish_3r', 'best_finish_3r', 'top3_count_3r', 'finish_trend',
    ],
    'jockey_trainer': [
        'jockey_wr_calc', 'jockey_course_wr_calc', 'jockey_surface_wr',
    ],
    'blood': [
        'sire_enc', 'bms_enc', 'sire_dist_wr', 'sire_surface_wr', 'bms_surface_wr',
    ],
    'horse_career': [
        'horse_career_races', 'horse_career_wr', 'horse_career_top3r',
        'horse_dist_top3r', 'horse_surface_top3r',
    ],
    'training': [
        'wood_best_4f_filled', 'has_wood_training',
        'sakaro_best_4f_filled', 'sakaro_best_3f_filled', 'has_sakaro_training',
        'training_time_filled', 'has_training',
    ],
    'today_info': [
        'horse_weight', 'odds_log', 'pop_rank', 'weight_change', 'weight_change_abs',
    ],
    'jrdb_basic': [
        'jrdb_idm', 'jrdb_jockey_idx', 'jrdb_info_idx', 'jrdb_sogo_idx',
        'paci_jockey_exp_wr', 'paci_jockey_exp_3rd', 'paci_ninki_idx',
    ],
    'derived': [
        'dist_change', 'dist_cat', 'age_sex', 'age_season',
        'horse_num_ratio', 'sire_dist', 'sire_surface',
    ],
}
# 合計約 80 件
```

#### JRA-only features (推定 50 件)

```python
JRA_ONLY = {
    'jrdb_extended': [
        'jrdb_dam_rensho_avg', 'jrdb_bms_rensho_avg',
        'jrdb_paci_*', 'jrdb_skb_*',  # 17 件
    ],
    'sr_srb_track_bias': [
        'sr_first3f_avg', 'sr_bias_homestr', 'sr_bias_4corner', 'sr_pace_up_pos',
        'srb_bias_*', 'srb_pace_up_pos',  # 11 件
    ],
    'premium_specialty': [
        'index_max_filled', 'index_avg5_filled', 'index_run1_filled',
        'time_1f_last_filled', 'training_intensity_enc',
    ],
}
# JRA-only 合計約 50 件
```

#### NAR-only features (推定 12 件)

```python
NAR_ONLY = {
    'nar_specific': [
        'is_chihou', 'course_jrac_code',  # NAR 場 code
        'nar_jockey_specific_wr',         # NAR 騎手特化勝率
        'rotation', 'kyakushitsu',         # JRDB はないが NAR では別形式
    ],
}
# NAR-only 合計約 12 件
```

→ 合計 142 features (共通 80 + JRA 50 + NAR 12)、 V15 150 と同等規模。

### 2.3 model logic

```python
def build_v20_features(df):
    """JRA / NAR 共通の features 構築"""
    is_nar = df['is_nar'].astype(int)
    # 共通 features は両方で同様計算
    df = build_common_features(df)
    # JRA only / NAR only features は条件付き
    df.loc[is_nar == 0] = build_jra_only_features(df.loc[is_nar == 0])
    df.loc[is_nar == 1] = build_nar_only_features(df.loc[is_nar == 1])
    # missing は 0 fallback (NAR data に JRA-only feature 不在 等)
    return df.fillna(0)


def predict_v20(df):
    """4-model ensemble"""
    p_lgb = v20_lgb.predict(df)
    p_xgb = v20_xgb.predict(df)
    p_ft = v20_ft_transformer.predict(df)
    p_ir = v20_intra_race_attention.predict(df)
    # Grid Search 最適重み
    p_final = 0.30*p_lgb + 0.25*p_xgb + 0.20*p_ft + 0.25*p_ir
    return p_final
```

---

## 3. 学習 plan (6/9-6/30)

### 3.1 学習 data 準備 (6/9-6/15、7 日)

#### JRA-VAN 1 ヶ月再契約 (¥2,090)
- 6/9 契約、 6/30 学習完了後 解約
- TARGET Frontier JV → CSV 抽出 (`tools/extract_jvdata.py`)
- 6/9-6/15 で 5 月分 + 6 月前半データ取得

#### data 統合
```python
jra_data = pd.read_csv('data/jra_races_full.csv')
nar_data = pd.read_csv('data/nar_all_races.csv')
jra_data['is_nar'] = 0
nar_data['is_nar'] = 1
combined = pd.concat([jra_data, nar_data])
```

#### sample 数 (推定)
- JRA: 50 万 horses (V15 base、 2015-2025 前期 まで)
- NAR: 5-10 万 horses (NAR v4 base + 2025-2026 backfill)
- 合計: 55-60 万 horses

### 3.2 features engineering (6/16-6/19、4 日)

```python
df = build_v20_features(combined)
# expanding window で各馬の career stats 計算 (リーク防止)
df = compute_expanding_horse_stats(df)
# JRA-only / NAR-only features の合成
df = build_jra_only_features(df)
df = build_nar_only_features(df)
```

リーク防止チェック:
- 全統計 features は expanding window
- sire / jockey encoding は fold ごとに train data のみで計算
- 4/29 リーク sib_*_wr は **完全削除** (V162_EXCLUDED 反映)

### 3.3 学習 (6/20-6/24、5 日)

```bash
python train/train_v20_jra_nar_ensemble.py \
    --data data/_v20_combined.pkl.gz \
    --features 142 \
    --ensemble lgb_xgb_ft_intra \
    --output keiba_model_v20.pkl.gz
```

学習時間 (推定):
- LGB: 1h
- XGB: 1.5h
- FT-Transformer: 4h (GPU 必要、Ryzen 7 + 16GB GPU)
- IntraRace Attention: 3h (GPU)
- Grid Search ensemble 重み: 30 min
- 合計: ~10 h × 5 日 = 学習試行 5 回まで可能

### 3.4 評価 (6/25-6/27、3 日)

```bash
# A/B test: V20 JRA subset vs V15 本番
python tools/eval_v20_vs_v15.py
# A/B test: V20 NAR subset vs NAR v4
python tools/eval_v20_vs_nar_v4.py
```

GO 条件:
- V20 JRA subset AUC ≥ 0.88 (V15 0.8858 維持)
- V20 NAR subset AUC ≥ 0.83 (NAR v4 0.8145 から改善)
- 両 model で winner_top1 ≥ 50% (V18 47.8% から改善)

### 3.5 production 統合 (6/28-6/30、3 日)

`tools/predict_v20.py` 新規:
```python
def predict_v20_for_race(race_id, is_nar):
    """JRA / NAR 共通で V20 を呼ぶ"""
    df = build_features_v20(race_id, is_nar=is_nar)
    p_ensemble = predict_v20(df)
    return p_ensemble
```

`predict_core.py` への統合は **慎重に** (既存 V15 動作影響)。 別 module で隔離 → 7 月以降 に統合判断。

---

## 4. paper trading + 本番投入 (7 月以降)

### 4.1 V20 paper trading (7/1-7/14)

V15 案B改 (JRA) + V20 paper (JRA + NAR) を並行運用、 2 週間で投資判断:
- V20 JRA subset paper ROI ≥ V15 ROI - 5pt → V20 採用
- 同 NAR subset paper ROI ≥ NAR v4 ROI → NAR 試行投入

### 4.2 V20 本番投入 (7/15+)

- 主: V20 JRA subset (案B改 12R 1勝、 700 円)
- 副: V20 NAR subset (NAR 試行 500 円)
- fallback: V15 (JRA fail 時) / NAR v4 (NAR fail 時) / 投票 skip

```python
# tools/predict_v20_orchestrator.py (将来)
result = predict_v20()
if result is None:
    # fallback chain
    result = predict_v15() if jra else predict_nar_v4()
```

---

## 5. リスク + 緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| V20 学習で個別 AUC 悪化 (JRA / NAR) | 中 | A/B test で確認、悪化なら個別 model 維持 |
| JRA-VAN 1 ヶ月再契約コスト ¥2,090 | 低 | 累計収支で回収 (6 月+ ROI 130% でも 1 ヶ月で +25K) |
| GPU リソース不足 | 中 | FT-Transformer / IntraRace Attention は subsample 学習 |
| feature shift 27.7x 同型問題 | 高 | sr/srb merge + 運用フィルタ + V20 で **学習時 features alignment** で根治 |
| sib_*_wr リーク再発 | 高 | V162_EXCLUDED を学習 script で明示的に exclude |

---

## 6. 撤退基準 (Phase 3 V20 失敗時)

V20 が GO 条件未達 (6/27 評価) なら:
- V15.1 SKB 採用 (Phase 3 既定 plan)
- V18/V19 sib 抜き再学習
- NAR v4 維持

→ V20 失敗でも V15.1 + 個別 V18/V19 + NAR v4 で 3 path 並行運用継続。

---

## 7. ファイル構成 (Phase 3 終了時)

```
keiba-ai/
├── keiba_model_v15_central{,_live}.pkl.gz    # V15 既存 (fallback)
├── keiba_model_v15_1.pkl.gz                  # V15.1 SKB 採用 (5/24-6/8)
├── keiba_model_v20.pkl.gz                    # V20 統合 (6/28-)
├── data/v18/models/v18_*_lgb.txt             # V18 sib 抜き再学習 (5/24+)
├── data/nar/models/keiba_model_nar_v4.pkl    # NAR v4 既存 (fallback)
└── tools/
    ├── predict_core.py                       # V15 既存 (絶対変更なし)
    ├── predict_v20.py                        # V20 新規 (6/28-)
    └── predict_v20_orchestrator.py           # V20 + fallback 統合 (7/15+)
```

---

## 8. 結論

V20 は 6/9-6/30 の 3 週間で構築可能、 JRA-VAN 1 ヶ月再契約 (¥2,090) で実現。
GO 条件達成で 7/15 以降 V20 統合運用、 失敗で V15.1 + V18/V19 + NAR v4 三 path 継続。
取り返し禁止ルール下、 段階的 + fallback 完備で安全。
