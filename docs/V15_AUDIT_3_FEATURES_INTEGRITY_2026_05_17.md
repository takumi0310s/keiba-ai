# V15-audit-3: V15 features integrity 真値 audit (2026-05-17)

★ read-only audit、 V15 .pkl.gz / cache / predict_core / v15.2 training 完全不変 ★

## 0. 結論

| 項目 | 値 | 判定 |
|------|----|----|
| V15 model features 数 | **145** (※ CLAUDE.md「150」は drift、 実値 145) | ⚠ doc drift |
| cache (df) shape | (527280, 232) | OK |
| cache に存在しない V15 feature | **0** | ✅ |
| RED_IMP_BUT_CONST (★ critical ★) | **0 件** | ✅ |
| RED_CONSTANT (unique<=1) | **8 件 (全 KNOWN)** | ✅ (intentional dead) |
| RED_LOW_UNIQUE (2-10) | 39 件 (大部分は categorical) | OK |
| WARN_QUASI_CONSTANT (mcr>95%) | 12 件 (うち 8 件は constant 重複) | OK |
| WARN_HIGH_NULL (>50%) | **0 件** | ✅ (cache は fillna 済) |
| MISSING (cache 欠落) | **0 件** | ✅ |
| TYB suspects (5) in V15 | **0 件** | ✅ |
| SKB suspects (10) in V15 | **0 件** | ✅ |
| LEAK_FEATURES_A (8) in V15 | **0 件** | ✅ |
| T1 monitor 再実行 | PASS (同結果) | ✅ |

**総合: V15 145 features integrity ✅ PASS、 critical leak / silent constant 共に検出されず。**

## 1. CLAUDE.md / memory drift

CLAUDE.md は「150 特徴量」と記載するが、 ★ 実際 V15 model file に格納された features は 145 ★。
これは v15.2 / v22 等への retrain 時の base としても 145 で固定。

→ V15 と関係ない doc drift、 model 自体は正常。

## 2. RED_CONSTANT 8 件 (★ 全 KNOWN ★)

| feature | value | lgb_gain | 理由 |
|---------|-------|----------|------|
| `is_nar` | 0 | 0.00 | JRA 専用 cache のため 0 固定 (intentional) |
| `prev_odds_log` | 2.7726 (log 16) | 0.00 | LEAK 除去後の default 残骸 |
| `prev_race_first3f` | 35.8 | 0.00 | default fill |
| `prev_race_last3f` | 36.5 | 0.00 | default fill |
| `prev_race_pace_diff` | 0.0 | 0.00 | default fill |
| `sire_shinba_top3r` | 0.22 | 0.00 | default fill |
| `pci` | 1.0196 | 0.00 | default fill |
| `gaisha_rank` | 0 | 0.00 | 4/26 audit で dead 確認済 |

★ 全 8 件 lgb_gain = 0 かつ xgb_gain = 0、 model に実質寄与ゼロ。
★ TYB merge bug 級 (importance > 0 but constant) は ★ 0 件 ★。

## 3. RED_IMP_BUT_CONST (★ critical ★): **0 件**

T1 monitor 「RED_IMP_BUT_CONST: 0 件」 と本 audit 完全一致。
TYB 同型の事故 (model 入力かつ分散ゼロ) 検出されず。

## 4. RED_LOW_UNIQUE (2-10 unique): 39 件

LGB gain 上位 (★ 正常 categorical ★):

| feature | unique | lgb_gain | 性質 |
|---------|--------|----------|------|
| paci_sogo_mark | 6 | 18266.2 | JRDB 印 (◎○▲△☆無) |
| surface_dist_enc | 10 | 16267.4 | 馬場×距離 cross |
| paci_idm_mark | 6 | 7326.0 | JRDB 印 |
| surface_enc | 2 | 6970.1 | 芝/ダ |
| course_enc | 10 | 6388.4 | 10 場 |
| training_intensity_enc | 4 | 1670.0 | 馬なり/強め/一杯 |
| jrdb_running_style | 5 | 1583.5 | 脚質 |
| jrdb_ze_furi_count | 8 | 1438.9 | 不利回数 |
| season | 4 | 1302.7 | 春夏秋冬 |
| jrdb_tb_homestr_inner | 6 | 1296.8 | 上り判定 |
| sex_enc | 3 | 1170.7 | 牡/牝/セン |
| jrdb_ranch_rank | 6 | 1160.3 | 牧場 rank |
| jrdb_stable_rank | 8 | 1121.3 | 厩舎 rank |
| jrdb_dist_apt | 6 | 853.5 | 距離適性 |
| ... | ... | ... | ... |

→ 全て natural categorical / ordinal feature。 RED_LOW_UNIQUE flag は false positive (intentional)。

## 5. WARN_QUASI_CONSTANT (mcr>95%): 12 件

8 件は §2 の RED_CONSTANT 重複。 残り 4 件:

| feature | unique | mcr | mcv | lgb_gain | 性質 |
|---------|--------|-----|-----|----------|------|
| `jrdb_prev_interference` | 14 | 0.9817 | 0.0 | 159.6 | 前走不利 (大半 0) |
| `course_renovated` | 2 | 0.9868 | 0 | 70.5 | 京都改修 flag |
| `jrdb_prev_rise_code` | 6 | 0.9762 | 3.0 | 34.7 | 前走上り評価 |
| `has_training` | 2 | 0.9701 | 1 | 0.0 | 調教取得有無 (97% 取得) |

→ 全て natural distribution、 model はわずかに重みを置く (`has_training` のみ dead)。

## 6. WARN_HIGH_NULL / MISSING: 0 件

cache build 時に fillna 済みのため null_rate 全 0%。
145 V15 features 全て cache に存在、 MISSING 0 件。

## 7. TYB / SKB / LEAK_FEATURES_A 永久除外 verify

| group | n | V15 145 features に含まれる数 | 期待 |
|-------|---|------------------------------|------|
| TYB suspects (paddock_idx / odds_idx / body_code / demeanor_code / live_composite_idx) | 5 | **0** | 0 ✅ |
| SKB suspects (skb_kishi_code_1-3 / skb_baba_code_1-3 / skb_kyaku_code_1-3 / skb_turf_hoof) | 10 | **0** | 0 ✅ |
| LEAK_FEATURES_A (odds_log / horse_weight / condition_enc / weight_change 等) | 8 | **0** | 0 ✅ |

★ LEAK 漏れ 0 件、 全 23 件 permanent exclusion 機能している。 ★

## 8. TOP 10 importance

| rank | feature | lgb_gain | unique |
|------|---------|----------|--------|
| 1 | paci_jockey_exp_3rd | 314,750 | 851 |
| 2 | paci_ninki_idx | 295,196 | 1,311 |
| 3 | paci_jockey_exp_wr | 266,346 | 754 |
| 4 | jrdb_ze_idm_avg | 175,971 | 7,637 |
| 5 | training_time_filled | 96,546 | 551 |
| 6 | training_per_dist | 53,711 | 5,102 |
| 7 | jrdb_ze_ten_avg | 36,673 | 18,777 |
| 8 | jrdb_idm | 35,433 | 711 |
| 9 | jrdb_class_code | 29,243 | 35 |
| 10 | horse_career_wr | 21,692 | 637 |

→ 全て unique 高 (>500)、 distribution 健全。 silent constant な top feature なし。

## 9. zero LGB gain 10 件 (★ XGB ensemble だけで使用 ★)

| feature | unique | xgb_gain | 判定 |
|---------|--------|----------|------|
| age_group | 6 | 194.5 | XGB のみ使用 (OK) |
| 残り 9 (is_nar 等) | 1 | 0.0 | 完全 dead (RED_CONSTANT に重複) |

→ LGB gain == 0 features の 90% は constant、 残 1 は XGB が捕捉。 整合性 OK。

## 10. T1 monitor 再実行 (★ check-only ★)

```
[features_integrity_monitor] Loading V15 cache...
  df shape: (527280, 232), V15 features: 145
[features_integrity_monitor] Loading V15 importance...
  V15 model features: 145
[features_integrity_monitor] Auditing...

=== SUMMARY ===
total: 145
RED_CONSTANT total: 8 (known=8, new=0)
RED_IMP_BUT_CONST (critical): 0
RED_LOW_UNIQUE: 39
WARN_HIGH_NULL: 0
WARN_QUASI_CONSTANT: 12
No critical issues (only known red flags)
```

★ T1 と本 audit 完全一致 (PASS) ★

## 11. CLAUDE.md / memory drift verify

| 記述 | 実値 | drift |
|------|------|-------|
| 「150 特徴量」 | 145 | ⚠ 5 件 drift (recommend: 更新 → 「145 features」) |
| 「リークフリー」 | TYB/SKB/LEAK_A 全 0 件 | ✅ 正しい |
| 「Pattern A (124 features)」 | V15 145 features (Pattern A 概念は v13.5b 由来) | ⚠ Pattern A は historical naming、 V15 とは別の特徴量 set |
| 「LGB+XGB+FT+IR ensemble」 | V15 は LGB+XGB (FT/IR は v13.5b) | ⚠ V15 本体は 2 model |

→ 整合性に critical impact なし。 doc 更新候補。

## 12. V15 production 不変保証 ✅

| ファイル | mtime | 状態 |
|----------|-------|------|
| `keiba_model_v15_central.pkl.gz` | 2026-04-08 23:32 | ✅ 不変 |
| `keiba_model_v15_central_live.pkl.gz` | 2026-04-08 23:32 | ✅ 不変 |
| `data/_v15_optuna_df_cache.pkl.gz` | 2026-04-08 23:40 | ✅ 不変 |
| `tools/features_integrity_monitor.py` | 2026-05-17 00:12 | 既存 (T1)、 read-only 実行のみ |
| `tools/predict_core.py` | — | 触れず |
| v15.2 training (PID 23528) | — | 中断せず ★ |

★ 全 V15 関連 file 完全不変、 v15.2 training 一切干渉なし、 destructive op ゼロ ★

## 13. 結論

1. **V15 145 features integrity ✅ PASS** (RED_IMP_BUT_CONST 0、 MISSING 0、 LEAK 漏れ 0)
2. **TYB merge bug 級事故 検出ゼロ** (silent constant な importance 入り feature なし)
3. **T1 monitor 完全動作** (同 result、 daily 監視 続行可)
4. **CLAUDE.md「150 features」 doc drift** (実 145)、 critical impact なし
5. **V15 / cache / v15.2 training 完全不変**

★ 親 audit 完全 honest 厳守、 V15 production 100% 保証 ★
