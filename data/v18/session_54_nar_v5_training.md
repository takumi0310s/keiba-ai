# Session #54 C: NAR V5 学習結果

**作成**: 2026-05-09 (Session #54 C)
**tool**: tools/train_nar_v5.py
**model**: data/nar/models/keiba_model_nar_v5.pkl
**metrics**: data/v18/session_54_nar_v5_metrics.json

---

## 0. 学習設定

- 学習 data: `data/nar_all_races.csv` (54,159 rows、 2024-01-01 ~ 2025-05-14)
- 前処理後: 53,407 rows / 4,827 races
- 時系列 split: 80% train (42,725) / 20% test (10,682)
- ensemble: LGB + XGB
- 同 hyperparam (V4 と完全一致)

---

## 1. AUC 結果

| model | LGB AUC | XGB AUC | Ensemble AUC | n_features |
|-------|---------|---------|--------------|-----------|
| V4 (再現) | 0.8189 | 0.8183 | **0.8188** | 22 |
| V5 (37 features) | 0.8183 | 0.8183 | **0.8187** | 37 |
| **delta** | -0.0006 | 0.0000 | **-0.0001** | +15 |

注: V4 baseline AUC 0.8188 は 元 V4 model の 0.8145 と 0.004 差。 これは ハイパラ + early_stopping 効果 の 可能性、 あるいは 新しい test split に よる ノイズ。

---

## 2. 重要 発見 ★

### 2-1. last3f LEAK 検出 + 修正

初回学習 (last3f_filled = 当該レース last3f) で AUC 0.867 (+0.048!) を記録。
last3f は **post-race** (走破後 計測される 上がり 3F) → LEAK。

**修正**: `df.groupby("horse_name")["last3f"].shift(1)` で **前走 last3f** を 使用。
修正後 AUC 0.8187 (V4 と同等)。

→ NAR V5 に **last3f は post-race**、 prev_last3f を 必ず使う必要あり。

### 2-2. V4 22 features は 既に 高度 飽和

V5 で 15 新規 features を 追加したが Ensemble AUC -0.0001。
top 15 importance に 8 件の V5 新規 features が 入る が **既存 V4 features と冗長**:

| V5 新規 | importance | V4 既存 で 重複 |
|---------|-----------|--------------|
| horse_surface_top3r | 3729 | (V4 horse 系なし、 効果はあるが 弱い) |
| course_dist_wr | 2795 | course_enc + dist_cat で 部分カバー |
| jockey_course_wr | 2464 | jockey_wr で 部分カバー |
| trainer_wr | 2386 | (新規、 効果あり) |
| frame_course_dist_wr | 1688 | bracket + course_enc で 部分カバー |
| last3f_filled (prev) | 1623 | (新規、 効果あり) |
| horse_weight_change | 1489 | weight_cat で 部分カバー |
| rest_days_filled | 1183 | (新規) |

→ **追加 features は 個別 では 寄与あるが、 V4 の odds_log + pop_rank が 圧倒的支配**:
- odds_log: importance 104,357 (V5 全 importance の 70%+)
- pop_rank: 16,413 (15%)
- 残り 35 features 合計が ~15%

NAR は 中央 (V15) と違い **市場 (odds + pop_rank) が ほぼ 全 信号** を 持つ → 追加 features の 余地小。

---

## 3. 比較: 中央 V15 vs NAR V5

| model | 主 features | AUC | 飽和点 |
|-------|-----------|-----|--------|
| 中央 V15 | 145 features (expanding × cross 多数) | 0.8788 | 145 で飽和、 大規模 (TFJV) 必要 |
| NAR V4 | 22 features (odds + pop + 簡易) | 0.8188 | 22 で飽和、 odds_log 支配 |
| NAR V5 | 37 features (V4 + expanding 15) | 0.8187 | 同 |

**NAR の 飽和は 中央 と異なる**:
- 中央: features の **多様性** で 飽和
- NAR: **odds_log 単独** で 飽和 (市場 効率高)

→ NAR の さらなる 改善には **異質 source** (NAR JV 不対応で 困難) か **ensemble に FT-Transformer 追加** が 必要。

---

## 4. V5 model spec (保存済)

```
file: data/nar/models/keiba_model_nar_v5.pkl
version: nar_v5
n_features: 37
LGB AUC: 0.8183
XGB AUC: 0.8183
Ensemble AUC: 0.8187
ensemble_weights: LGB 0.500 / XGB 0.500
n_rows: 53,407
n_races: 4,827
trained_at: 2026-05-09
v4_baseline_auc: 0.8188
delta_vs_v4: -0.0001
```

---

## 5. 結論

✅ V5 学習 完了 (37 features)
✅ V5 AUC 0.8187、 V4 AUC 0.8188 → delta -0.0001 (改善なし)
✅ **重大発見**: last3f は post-race LEAK、 V5 で prev_last3f に修正
✅ **重大発見**: NAR は odds_log 単独で 70%+ importance、 features 追加 効果薄
✅ V5 model file 保存: `data/nar/models/keiba_model_nar_v5.pkl`

**結論**:
- V5 は **AUC 改善 達成不可** (audit 期待 +0.005-0.015 vs 実 -0.0001)
- 5/12 paper trade では V4 維持 推奨 (V5 は同等で 投入 メリット無し)

→ Session #54 D で 5/12 投入判断 実施
