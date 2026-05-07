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

---

## 9. 学習 data 構造 詳細 (Session #37 拡張)

### 9.1 統合 data schema

```python
# data/_v20_combined.pkl.gz
{
    'df': pd.DataFrame (~55-60 万 horses),
    'features': List[str] (142 features),
    'target': {
        'is_win':  binary,  # 1 着 (NAR/JRA 共通)
        'is_top3': binary,  # 3 着以内 (NAR/JRA 共通)
    },
    'meta': {
        'jra_rows': int,    # 50 万
        'nar_rows': int,    # 5-10 万
        'years': '2015-2026',
        'cutoff_date': '2026-06-15',
    },
}
```

### 9.2 sample weight 戦略

統合学習でクラス不均衡を補正:

| group | rows | weight | 根拠 |
|-------|------|-------|------|
| JRA | 50 万 | 1.0 | base |
| NAR | 5-10 万 | 5.0 - 7.5 | rows 比逆数で 1:1 effective weight |

`LGBMClassifier(sample_weight=...)` で実装。 `is_nar=0` で 1.0、 `is_nar=1` で 5.0-7.5。

NAR weight は 5 / (NAR rows / JRA rows) で動的決定。

### 9.3 リーク厳禁チェック (8 features)

V15 既存 LEAK list 継承 + 4/29 sib リーク追加:

```python
LEAK_FEATURES_V20 = {
    # V15 既存
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs', 'weight_cat',
    'weight_cat_dist', 'cond_surface',
    # V162 EXCLUDED (4/29)
    'sib_top3_rate', 'sib_shinba_wr',
    # V17 LEAK_TIME_INVARIANT
    'zk_prev_anshin', 'skb_anshin', 'skb_aisho', 'skb_heavy_apt',
    'kka_bms_rensho_max', 'kka_bms_rensho_min',
}
# 計 18 features
```

**Pattern A** (リークフリー、評価用) は LEAK_FEATURES_V20 全除外、 124 features。
**Pattern B** (実運用、当日情報込み) は LEAK_FEATURES_V20 中 当日確定可能なものは含む、 142 features。

### 9.4 NAR data 補完戦略

NAR v4 の base data (2024-03〜2025-05) の他に:
- 2025-06〜2026-05 の 1 年分 backfill (NAR site scrape)
- 主要場: 川崎 / 大井 / 船橋 / 浦和 / 笠松 (合計 ~3 万 races)
- 取得 priority: 重賞 + 4 歳上 1 千万下 (高 odds 安定 path)

```python
nar_recent = scrape_nar_2025_2026()  # ~3 万 races, 4 万 horses
nar_existing = pd.read_csv('data/nar_all_races.csv')  # 既存
nar_combined = pd.concat([nar_existing, nar_recent])
```

---

## 10. 検証手順 詳細 (Session #37 拡張)

### 10.1 walk-forward validation 設計

| fold | train | test | 目的 |
|------|-------|------|------|
| WF-1 | 2015-2020 | 2021 | base 確認 |
| WF-2 | 2015-2021 | 2022 | 趨勢 |
| WF-3 | 2015-2022 | 2023 | 中期 |
| WF-4 | 2015-2023 | 2024 | 直近 |
| WF-5 | 2015-2024 | 2025 | 最新 OOS |
| WF-6 | 2015-2025 (前半) | 2025 (後半) | 短期 OOS |

### 10.2 metrics 計算 (per-fold)

```python
# JRA subset / NAR subset / 統合 で別計算
metrics = {
    'auc': roc_auc_score(y_test, p_test),
    'logloss': log_loss(y_test, p_test),
    'winner_top1': (top1_predicted == winner).mean(),  # 1 着 hit
    'top3_hit_rate': (top3_overlap >= 1).mean(),  # 複勝 of top3
    'mean_p18': p_test.mean(),
    'max_p18': p_test.max(),
    # 5/2-5/3 LIVE retro: shift_factor = mean_p18_BT / mean_p18_LIVE (target < 5x)
}
```

### 10.3 GO 条件 (per-fold WF-5 + LIVE retro)

| 条件 | target | 根拠 |
|------|--------|------|
| AUC JRA subset | ≥ 0.88 | V15 0.8939 維持 |
| AUC NAR subset | ≥ 0.83 | NAR v4 0.8145 改善 |
| winner_top1 (BT 2025 OOS) | ≥ 50% | V18 47.79% から改善 |
| LIVE retro winner_top1 (5/2-5/3 type) | ≥ 40% | V18 LIVE 34.5% から改善 |
| shift factor (BT/LIVE) | ≤ 5x | V18 11x から大幅改善 |
| sib リーク影響 (sib抜き比) | < 1pt | リーク疑い 完全消失 |

全 6 条件を 6 月後半 (6/27 evaluation) で同時検証、 4 つ以上 PASS で GO。

---

## 11. 実装 phase + schedule 詳細 (Session #37 拡張)

### Phase 3.0: 並行前倒し (5/24-6/8、Session #37 始動)

| Date | 作業 | 担当 model |
|------|------|----------|
| 5/24-5/27 | V18/V19 sib抜き 6-fold WF + retro 検証 | V18/V19 |
| 5/28-5/31 | V15.1 SKB の LGB+XGB+FT+IR 4-model 統合 | V15.1 |
| 6/1-6/4 | V15.1 BT 全条件 ROI 検証 | V15.1 |
| 6/5-6/8 | V20 architecture 確定 + JRA-VAN 再契約準備 | V20 設計 |

### Phase 3.1: V20 構築 (6/9-6/30、Session #36 既定 plan)

| Date | 作業 | 出力 |
|------|------|------|
| 6/9 | JRA-VAN 再契約 + TARGET extract | jra_races_full_2026_q2.csv |
| 6/10-6/15 | data 統合 + features engineering | _v20_combined.pkl.gz |
| 6/16-6/19 | LEAK チェック + sample_weight 設定 | _v20_train_df.pkl |
| 6/20-6/24 | 4-model 学習 + Grid ensemble | keiba_model_v20.pkl.gz |
| 6/25-6/27 | A/B test (vs V15 / vs NAR v4) | v20_ab_test.json |
| 6/28-6/30 | predict_v20.py + orchestrator 整備 | predict_v20.py + tools/ |

### Phase 3.2: paper trading + 本番投入 (7/1-)

| Date | 作業 | path |
|------|------|------|
| 7/1-7/14 | paper trading (V20 + V15 案B改 並行) | paper |
| 7/15+ | V20 本番投入 (JRA + NAR) | production |

---

## 12. A/B test rollout 詳細 (Session #37 拡張)

### 12.1 7/1-7/7: V20 影武者 deploy (paper only)

```bash
# tools/race_auto_notify.py に V20 並行 prediction 追加
# 投票は V15 案B改 のみ、 V20 は予測のみ記録
python tools/race_auto_notify.py --shadow-v20
```

蓄積 metric:
- 予測一致率 (V15 vs V20)
- V20 単独 winner_top1 / EV / 撤退判定
- 不一致 race の事後検証

### 12.2 7/8-7/14: V20 段階投入 (10% capital)

GO 条件 (paper 14 日):
- V20 winner_top1 ≥ 50% (V15 比 +3pt)
- V20 NAR winner_top1 ≥ 40%
- 不一致 race で V20 が V15 を上回る判定

満たせば V20 を 10% capital 投入 (200 円/レース)、 V15 700 円/レース 維持。

### 12.3 7/15-7/31: V20 本格投入 (50% capital)

paper 14 日 + 10% capital 7 日 で確認後、 段階移行:
- V20 主: JRA 案B改 + NAR 試行 (全体 50% capital, 350 円/レース)
- V15: 50% capital 維持 (350 円/レース)

8 月以降に V20 100% migration 判断。

### 12.4 撤退条件 (各 phase)

```python
# 累計 ROI で判定
roi_v20 = (wins * payouts) / invested
if roi_v20 < V15_ROI - 10pt:    # 10pt 悪化
    rollback_to_v15()
if roi_v20 < 80%:               # 絶対基準
    rollback_to_v15()
```

撤退時、 V15 案B改 100% に戻す、 V20 は再学習行きで paper 観察継続。

---

## 13. リスク + 緩和策 詳細 (Session #37 拡張)

### 13.1 学習段階リスク

| リスク | 影響 | 緩和 | 検出 |
|--------|------|------|------|
| sample_weight 過剰調整で NAR 学習過剰 | 中 | weight grid {3, 5, 7.5} で WF 比較 | NAR subset AUC 監視 |
| GPU OOM (FT-Transformer) | 高 | batch_size 段階削減 (256→128→64) | Loss NaN 検出 |
| feature alignment NaN (NAR-only / JRA-only) | 中 | fillna(0) + dummy column | merge 後 NaN rate |
| 学習時 LEAK 再発 (sib 復活) | 高 | LEAK_FEATURES_V20 を script で明示 assert | 学習開始 log で確認 |

### 13.2 評価段階リスク

| リスク | 影響 | 緩和 |
|--------|------|------|
| BT で AUC ≥ 0.88 だが LIVE で大幅劣化 | 高 | 5/2-5/3 LIVE retro で shift_factor < 5x 確認、 NG なら段階回帰 |
| NAR subset で sample 不足 (NAR v4 比退化) | 中 | NAR rows weight up + ND 評価 |
| feature shift 同型問題が V20 でも発生 | 高 | JRA + NAR 統合学習で shift 緩和を期待、 sib抜き効果と複合確認 |

### 13.3 production 段階リスク

| リスク | 影響 | 緩和 |
|--------|------|------|
| V20 予測 service 停止 (NAR data 取得 fail) | 中 | V15 fallback に確実切替 |
| V20 model file corruption | 高 | git LFS で V20 model 管理、 daily backup |
| paper trading 14 日で sample 不足 | 中 | 不足の場合 7 日延長、 capital 投入見送り |

---

## 14. JRA-VAN 再契約 detailed plan (Session #37 拡張)

### 14.1 再契約 期日 + workflow

```
6/8 (土) 21:00: 累計収支 + 撤退判定 確認
6/9 (日) 朝: JRA-VAN 1 ヶ月再契約 ¥2,090
6/9 (日) 昼: TARGET Frontier JV インストール / 設定
6/9 (日) 夜: jra_races_full.csv backfill (5月分 + 6月前半)
6/15 (土) 朝: data 完了確認、 data/_v20_combined.pkl.gz 構築
```

### 14.2 ROI 回収シミュレーション

V15 案B改 想定 ROI 140% / 月間投資 21,000 円 (100 races × 200 円/race) として:
- 6 月 累計利益 想定 +8,400 円 (ROI 140% × 21K)
- JRA-VAN ¥2,090 cost は 6 月内で 25% 回収
- V20 GO 後の 7 月以降は 連続 ROI 改善 で 1 ヶ月 完全回収

V20 失敗時でも JRA-VAN 1 ヶ月分は累計収支から控除可能 (撤退余裕 +63,530 円)。

### 14.3 解約 timing

6/30 V20 学習完了後、 7/1 から TARGET Frontier JV はバックアップ目的のみ:
- 7/1 解約 → 7/2-7/31 paper trading は scrape のみで運用
- ¥2,090 は 6/9-7/9 の 1 ヶ月のみ

---

## 15. ファイル構成 (Phase 3 終了時、 詳細)

```
keiba-ai/
├── # === V15 既存 (絶対不変) ===
├── keiba_model_v15_central{,_live}.pkl.gz
├── tools/predict_core.py
├── tools/daily_predict.py
│
├── # === V15.1 SKB (5/24-6/8 採用) ===
├── data/v15.1/
│   ├── v15_1_lgb_v37.txt       # Session #37 B 出力
│   ├── v15_1_xgb.json
│   ├── v15_1_ft_transformer.pt # Phase 3 末
│   └── v15_1_intra_race.pt
├── train/run_v15_1_lgb_xgb.py
├── train/v15_1_features.py
│
├── # === V18/V19 sib抜き (5/24+ 採用) ===
├── data/v18/v18v19_retraining/
│   ├── v18_lgb_no_sib_v1.txt   # Session #37 A 出力
│   ├── v19_lgb_no_sib_v1.txt
│   ├── v18_xgb_no_sib_v1.json  # Session #38
│   └── v18v19_4model_no_sib.pkl  # Phase 3 末
├── train/v18v19_no_sib/run_v18v19_no_sib_singlefold.py
│
├── # === V20 統合 (6/28+) ===
├── keiba_model_v20.pkl.gz
├── tools/predict_v20.py
├── tools/predict_v20_orchestrator.py  # V20 + fallback
├── train/train_v20_jra_nar_ensemble.py
│
└── # === paper trading + 評価 ===
└── tools/eval_v20_vs_v15.py
└── tools/eval_v20_vs_nar_v4.py
└── data/v20_paper_trading/  # 7/1-7/14
```

---

## 16. Session #37 → #38 連携

### 16.1 Session #37 完了 deliverable

- ✅ V18/V19 sib抜き single-fold LGB 学習 model 出力 (BT 2025 OOS 比較)
- ✅ V15.1 LGB+XGB 互換確認 / WF (2024 + 2025)
- ✅ V20 architecture 詳細設計 (本書 sections 9-15)
- ✅ V15 動作不変 final check
- ✅ 5 commits + push + Discord

### 16.2 Session #38 (5/13-5/15) 残作業

- [ ] V18/V19 sib抜き 6-fold WF (LGB+XGB)
- [ ] V18/V19 sib抜き LIVE retro (5/2-5/3, 4/26)
- [ ] V15.1 4-model ensemble (FT+IR 追加) WF
- [ ] V20 学習 data spec 確定 + JRA-VAN 再契約 timing 決定
- [ ] 5/16 GO/no-go 最終判断 (V18/V19 retro 結果ベース)

---

V20 は Phase 3 後半 (6/9-6/30) 集中構築、 Session #37 で前倒し並行 (V18/V19 sib抜き + V15.1 LGB+XGB) で土台作り完了。
GO 条件 4/6 PASS で 7/15 以降 V20 段階投入、 失敗で V15.1 + V18/V19 + NAR v4 三 path 継続。
取り返し禁止ルール下、 段階的 + fallback 完備で安全。
