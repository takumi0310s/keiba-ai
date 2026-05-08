# Session #55 A: V20 expanding 化 候補選定

**作成**: 2026-05-09 (Session #55 A、 dev/v20-expanding)
**目的**: V15 145 features を 3 cat 分類、 expanding 化候補抽出

---

## 1. 戦略 (V18/V19 sib_w5 +0.0689 成功 pattern)

Session #41 D で sib_top3_rate (lifetime) → sib_top3_rate_exp_w5 (window=5) で:
- corr_target 0.2939 (リーク含) → 0.1689 (full expanding) → **0.2010 (window=5)**
- LIVE retro winner_top1: no_sib 24.14% → **sib_w5 34.48%** (+10.34pt 完全回復)

→ **window 限定 expanding** で:
1. データリークを排除 (時系列 cumsum-current)
2. 直近の trend を捕捉 (古いレース重み低下)
3. 信号強化

V15 145 features の 同様の処理で +0.005-0.015 AUC 期待。

---

## 2. V15 145 features の 3 cat 分類

### Cat 1: point-in-time (不変、 expanding 化 不要、 ~80 件)

```
weight_carry, age, distance, course_enc, surface_enc, sex_enc,
num_horses_val, horse_num, bracket, sire_enc, bms_enc, ...
```

### Cat 2: history aggregation (★ expanding 化候補 28 件 ★)

| feature | 現状 | window 候補 |
|---------|------|-----------|
| jockey_wr_calc | lifetime | 30 / 90 / 365 (R 単位) |
| jockey_course_wr_calc | 同上 | 同 |
| jockey_surface_wr | 同上 | 同 |
| jockey_horse_wr | 同上 | 馬単位 累積 |
| jockey_horse_top3r | 同上 | 同 |
| trainer_top3_calc | 同上 | 30 / 90 / 365 |
| horse_career_races | 通算 | (期間限定 検討) |
| horse_career_wr | lifetime | window=3, 5, 10 |
| horse_career_top3r | 同上 | 同 |
| horse_dist_top3r | 同上 | window=3, 5 |
| horse_surface_top3r | 同上 | 同 |
| sire_surface_wr | lifetime | window=200, 500 産駒 |
| sire_dist_wr | 同上 | 同 |
| sire_shinba_top3r | 同上 | window=50, 200 |
| bms_surface_wr | 同上 | window=200, 500 |
| frame_course_dist_wr | lifetime | window=30, 100 |
| paci_jockey_exp_wr | (jrdb 内部) | (調整困難) |
| avg_finish_3r | 直近 3 R | (既に window=3) |
| top3_count_3r | 同上 | 同 |
| avg_last3f_3r | 同上 | 同 |
| index_avg5_filled | 直近 5 R | (既に window=5) |
| jrdb_ze_idm_avg | (前走 avg) | (前走限定、 expanding 不要) |
| jrdb_ze_ten_avg | 同上 | 同 |
| jrdb_ze_agari_avg | 同上 | 同 |
| jrdb_ze_furi_count | 同上 | 同 |
| jrdb_dam_rensho_avg | (jrdb 内部) | (調整困難) |
| jrdb_bms_rensho_avg | 同上 | 同 |
| wood_count_2w | (window=2 週) | 既に window |
| total_training_count | (短期間) | (既に短期) |

→ **本命候補 16 件** (既 window 化、 jrdb 内部 を除く):
   jockey_wr_calc, jockey_course_wr_calc, jockey_surface_wr,
   jockey_horse_wr, jockey_horse_top3r, trainer_top3_calc,
   horse_career_wr, horse_career_top3r,
   horse_dist_top3r, horse_surface_top3r,
   sire_surface_wr, sire_dist_wr, sire_shinba_top3r, bms_surface_wr,
   frame_course_dist_wr, horse_career_races

### Cat 3: leaderboard / ranking (動的 reweight 候補、 ~40 件)

```
prev_finish, prev2_finish, ... (前走 ranking、 expanding 化困難)
prev_last3f, prev_pass4, prev_prize (前走特定値、 expanding 化 N/A)
ev/odds 系 (動的)
```

→ Cat 3 は別 strategy (ranking_reweight、 Session #56-57 候補)。

---

## 3. expanding 化 6 features (PoC スコープ、 効果見込 大)

V18/V19 sib_w5 と同じ window=5 で:

| feature | 現状 | expanding 版 | 期待 |
|---------|------|-------------|------|
| jockey_wr_calc | lifetime | jockey_wr_calc_w30 (R 単位) | +0.001-0.003 |
| jockey_horse_top3r | lifetime | jockey_horse_top3r_w5 | +0.002-0.005 |
| trainer_top3_calc | lifetime | trainer_top3_w90 | +0.001-0.003 |
| horse_career_wr | lifetime | horse_career_wr_w5 | +0.001-0.003 |
| sire_surface_wr | lifetime | sire_surface_wr_w200 | +0.001-0.005 |
| frame_course_dist_wr | lifetime | frame_course_dist_wr_w30 | +0.001-0.002 |

合計期待: **+0.007-0.021 AUC** (V15 0.8939 baseline、 V20 + expanding で 0.901-0.915 想定)

---

## 4. 5/9 V15 投資保護

✅ V15 model md5: `842b9a5f...` 不変
✅ main 不変、 dev/v20-expanding 専用
✅ TFJV / jra_races_full は read-only

→ **5/9 朝 V15 完全保証**

---

## 5. 結論

✅ A1: V15 145 features 3 cat 分類
✅ A2: expanding 化候補 28 件 → 本命 6 件 (PoC スコープ)
✅ A3: 期待 AUC +0.007-0.021 (V18/V19 sib_w5 +0.069 の 1/3-1/10、 単一 features の効果見込み)

→ **B 領域で 6 features 実装、 C 領域で 学習効果測定**

---

**Session #55 A 完了 (dev/v20-expanding)**
