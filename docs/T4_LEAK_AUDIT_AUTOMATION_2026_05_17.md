# T4 leak audit automation (SKB / TYB padock_idx 級事故 永久防止)

**実施日**: 2026-05-17
**位置付け**: 学習前 leak detect の機械化 framework。 SKB (Session #38) / TYB padock_idx (Sub-task 6/11) / dam_top3r 全年集計 (Session #38) 級の事故が v15.2 以降で再発しないことを保証する gate logic。
**V15 production 影響**: ★ 0% (read-only analysis、 .pkl.gz / predict_core / app.py / daily_predict 一切 触れない) ★

---

## 0. 結論 (TL;DR)

| 項目 | 値 |
|------|----|
| V15 (145 features) leak audit | **0 件 PASS** ✅ (leak=0、 suspect=0、 unknown=0) |
| v15.2 候補 (22 features) 事前 audit | **1 件 SUSPECT** ⚠ (paci_info_idx, corr +0.4139) |
| 学習 gate ready | ✅ `tools/v15_2_train_gate.py` (exit 0/1/2) |
| test 結果 | **35/35 PASS** ✅ (`tests/T4_leak_audit_test.py`) |
| V15 production 影響 | **0** (read-only、 不変保証) |

---

## 1. 設計概要

### 1-1. 過去 leak 事故 (本 framework が想定する threat model)

| 事故 | session | 詳細 | 検出ポイント |
|------|---------|------|-------------|
| **V15.1 SKB POST-RACE LEAK** | Session #38 (4 月) | skb_kishi_code_3 単独 +480bp、 corr_target 0.137、 1着→364 / 10着→176 で monotonic | datatype=skb → safety=leak |
| **TYB padock_idx** | Sub-task 6/11 (5/16) | corr +0.3539、 TYB ZIP 17:00 JST publish = race 後配信 | datatype=tyb → safety=leak |
| **jrdb_odds_idx** | 5/16 | +0.4214 ≈ popularity 同等、 odds-based LEAK | (TYB 内、 leak block) |
| **dam_top3r 全年集計** | Session #38 | 静的 csv で全 train+test 集計 → 真の signal 0.169 / leak 0.125 | expanding 化必須、 本 framework は detect 範囲外 (FE 設計時 caution) |

### 1-2. 3 層 detect 戦略

```
layer 1: datatype 単位 release timing mapping (29 JRDB datatype)
         → POST-RACE / LIVE_only / safe / prev_only を機械的に判定
         → SKB / TYB / hjc / oz/ou/ot/ov/ow = 即 LEAK 判定

layer 2: corr_target audit (|corr| > 0.40 で HIGH_CORR_LEAK_SUSPECT flag)
         → datatype safe でも corr 異常 high なら audit 必須 (V15.1 SKB 教訓)
         → 既知 safe (paci_jockey_*, jrdb_idm 等) は KNOWN_SAFE_HIGH_CORR で除外

layer 3: prev_only ref 検出 (sed/sr/srb/ze/zk = current race LEAK、 prev 限定 safe)
         → name pattern (prev_*, jrdb_prev_*, jrdb_ze_*, jrdb_tb_*) で前走 ref と判定
         → そうでない sed/srb 由来 feature は PREV_ONLY warning 提示
```

---

## 2. JRDB 29 datatype release timing 完全表

Sub-task 5-1 / 6 / 11 / 18 の結果を **完全反映**。 V15 cache の 145 features は全てこの map に乗る。

| ID | timing | safety | desc | leak risk |
|----|--------|--------|------|----------|
| kyi | morning_06 | safe | 当日朝 06:00 基本 race meta + 馬指数 | low |
| bac | previous_thursday | safe | 前週木曜 番組公表 | low |
| cha | minus_3_days | safe | 中央木曜追切後 3-4 日前 確定 | low |
| kab | morning_06 | safe | 当日朝 kaisai-level | low |
| paci | morning_06 | safe | 前日夜 + 当日朝 sync | low |
| kta | morning_06 | safe | 場別調教師 | low |
| jo | morning_06 | safe | JO ファイル (CID/LS 指数) | low |
| joa | previous_thursday | safe | race meta v2 | low |
| ukc/ksa/kz/csa/cz/kkb/kza | static_master / weekly | safe | master / 週次集計 | low |
| cyb / cyb_v2 | live_minus_15 | **live_only** | 直前 (-15 min) | high (live 不可) |
| sed/sr/srb/ze/zk/kka/kka_v2/kka_features/kaa | post_race | **prev_only** | 前走 ref のみ safe、 current race は LEAK | medium |
| **skb** | post_race | **★ leak ★** | V15.1 SKB POST-RACE LEAK 確定 | **high (V15.1 NO-GO 原因)** |
| **tyb** | **post_race_17:00** | **★ leak ★** | TYB ZIP 17:00 JST publish (Sub-task 11) | **high (live 不可、 永久放棄)** |
| hjc | post_race | **★ leak ★** | 払戻系 | high |
| oz / ou / ot / ov / ow | post_race | **★ leak ★** | 賠率系 (JRDB) | high |

★ ★ ★ 重要注記 ★ ★ ★

**V15 cache の `oz_tansho_base_log` / `oz_fukusho_base_log` / `oz_base_pop_rank`** は **prefix が "oz_" だが ★ JRDB oz datatype (払戻系 LEAK) ではない ★**。
これらは `tools/build_odds_base_retro.py` が **jrdb_kyi.csv 基準オッズ (前日朝オッズ)** から計算した自前 FE で、 morning_06 release timing で **safe**。 本 framework は `OZ_FE_NAMES` で明示的に kyi datatype 扱いに上書きする。

---

## 3. V15 (145 features) leak audit 結果

### 3-1. 結果サマリ

```
=== T4 leak audit (V15 cache (145 features)) ===
features count: 145

=== summary ===
  total: 145
  OK: 145
RESULT: PASS (no leak / suspect)
```

- **leak detect: 0 件** ✅
- **HIGH_CORR_LEAK_SUSPECT: 0 件** ✅
- **UNKNOWN datatype: 0 件** ✅

→ ★ もし leak 検出されていたら V15 自体に致命的問題、 緊急修正必要。 0 件 PASS で V15 健全性確認 ★

### 3-2. high-corr (|corr|>0.40) features (6 件、 全て safe)

| feature | corr_target (実測) | datatype | safety | 採用根拠 |
|---------|-------------------:|----------|--------|----------|
| paci_jockey_exp_3rd | +0.4583 | paci | safe | 騎手 3 着内率 (morning_06 pre-race) |
| paci_jockey_exp_wr | +0.4560 | paci | safe | 騎手勝率 |
| paci_ninki_idx | +0.4477 | paci | safe | 人気指数 (odds-derived だが pre-race snapshot) |
| paci_jockey_mark | +0.4229 | paci | safe | 騎手印 |
| paci_sogo_mark | +0.4046 | paci | safe | 総合印 |
| jrdb_training_idx | +0.4010 | kyi | safe | 調教指数 (kyi 内 morning_06) |

→ 全て KNOWN_SAFE_HIGH_CORR list に登録済、 `corr_flag=ok_known_safe`、 verdict=OK。

### 3-3. 出力 JSON

`data/v15_2/v15_leak_audit_2026_05_17.json` に 145 features 全結果保存済。

---

## 4. v15.2 候補 (22 features) 事前 audit 結果

### 4-1. 結果サマリ

```
=== T4 leak audit (file: data/v15_2/features_v152_candidates.txt) ===
features count: 22

=== summary ===
  total: 22
  OK: 20
  PREV_ONLY: 1
  SUSPECT: 1
  ⚠ HIGH_CORR_LEAK_SUSPECT: ['paci_info_idx']
RESULT: FAIL (HIGH_CORR_LEAK_SUSPECT detected)
```

### 4-2. ★ 注意必須 features ★

| feature | corr_target | datatype | verdict | 推奨アクション |
|---------|------------:|----------|---------|--------------|
| **paci_info_idx** | **+0.4139** | paci | **SUSPECT** | ★ Sub-task 18 §3-2 注記の通り、 odds-related の可能性高 → 5/24+ 実 audit (monotonic + per-finish + train/test stability) で leak 確認必須。 LEAK なら永久除外 ★ |
| srb_bias_straight | +0.05 (推定) | srb | PREV_ONLY | srb は post_race、 前走 ref として参照されているか name 規則確認必要 (例: `prev_srb_*` / `srb_prev_bias_straight` 等) |

### 4-3. priority A+B 17 features の verdict

全 17 features (Sub-task 18 §5-1 採用想定) は **datatype-baseline OK**:

- breeder_dist_1 / breeder_dist_1_race_rank / breeder_track_1 → kka_v2 (静的 breeder、 OK)
- paci_gekiso_race_rank / paci_gekiso_idx / paci_lsidx_race_rank / paci_ls_idx_rank / paci_gekiso_rank → paci morning_06 safe
- kta_ichi_idx_pred / kta_ichi_pred_race_rank → kta morning_06 safe
- cha_chukan_idx_race_zscore / cha_chukan_time_idx / cha_oikiri_idx_trend / cha_shimai_time_3r_mean → cha 3-4 日前 safe
- kab_turf_baba_x_bracket / kab_straight_sa_x_horse_num_ratio / kab_renzoku_day → kab morning_06 safe

★ ただし feature 構築時に **expanding window 厳守** (dam_top3r 教訓)、 trend は per horse_id + date 順、 rank/zscore は groupby race_id ★

### 4-4. 出力 JSON

`data/v15_2/v152_candidates_leak_audit_2026_05_17.json`。

---

## 5. 学習 gate 仕様 (`tools/v15_2_train_gate.py`)

### 5-1. 使い方

```bash
# V15 cache 健全性 check
python tools/v15_2_train_gate.py --v15-cache

# v15.2 候補 audit (本番、 strict)
python tools/v15_2_train_gate.py --features data/v15_2/features_v152_candidates.txt

# v15.2 候補 audit (SUSPECT は warning 扱い、 5/24+ 実 audit 用)
python tools/v15_2_train_gate.py --features data/v15_2/features_v152_candidates.txt --allow-suspect
```

### 5-2. exit code 仕様

| exit | 意味 | trigger 条件 |
|:----:|------|------------|
| **0** | **PASS (学習許可)** | leak=0 + suspect=0 (allow_suspect なら suspect warning でも OK) + unknown=0 |
| **1** | **FAIL (学習禁止)** | leak ≥ 1 件 OR suspect ≥ 1 件 (allow_suspect なし) OR unknown ≥ 1 件 |
| **2** | **ERROR** | features file not found 等 |

### 5-3. 統合運用 例

```bash
# v15.2 学習スクリプト の冒頭 (5/24+ 親実装時)
python tools/v15_2_train_gate.py --features data/v15_2/features_v152_actual.txt || {
    echo "★ leak audit FAIL、 学習中止 ★"
    exit 1
}
python train/v15_2_master.py  # gate PASS 後のみ実行
```

---

## 6. test 結果 (35/35 PASS)

`tests/T4_leak_audit_test.py` 全 35 tests:

| カテゴリ | tests | 内容 |
|---------|------:|------|
| TestReleaseMapping | 6 | SKB/TYB/payout = leak、 safe datatype、 prev_only datatype、 cyb=live_only |
| TestIdentifyDatatype | 11 | jrdb_skb/tyb prefix、 oz_* = kyi (★払戻ではない注意★)、 jrdb_cha/jo 由来、 v15.2 prefix |
| TestAuditVerdict | 6 | SKB/TYB/payout LEAK 判定、 prev_ref OK、 breeder OK |
| TestHighCorrSuspect | 3 | paci_info_idx flag、 known_safe 除外、 low_corr OK |
| TestV15FullAudit | 3 | V15 leak=0、 suspect=0、 high_corr 全 safe |
| TestV152Candidates | 3 | v15.2 leak=0、 paci_info_idx suspect、 gate block |
| TestLeakInjection | 3 | SKB/TYB padock/HJC inject → LEAK detect |

```
$ python tests/T4_leak_audit_test.py
...
Ran 35 tests in 8.205s
OK
```

---

## 7. V15 production 不変保証 ✅

| 確認項目 | 状態 |
|---------|:----:|
| V15 .pkl.gz | ✅ 不変 (read-only audit のみ) |
| V15 cache (`data/_v15_optuna_df_cache.pkl.gz`) | ✅ 不変 |
| predict_core.py | ✅ 不変 |
| daily_predict.py / race_auto_notify.py / app.py | ✅ 不変 |
| cumulative_results.csv | ✅ 不変 |
| 既存テスト (regression_test_v15_final.py 等) | ✅ 不変 |
| 実 v15.2 学習実行 | ✅ なし (gate logic 設計のみ、 親 5/24+ で実 audit) |

---

## 8. 監視 plan (5/18+、 user 判断)

```
日次起動 (将来):
  Windows タスクスケジューラ で 朝 03:30 自動実行
  → python tools/leak_audit_automation.py --v15-cache --output data/v15_2/daily_audit_YYYYMMDD.json
  → leak 検出 (返り値 1) で Discord webhook alert
  → V15 cache の features が変更された場合 (新規 feature 追加・rename) に即時 detect
```

(★ 本 sub-task では「ready」 状態まで。 daily 自動起動の登録は user 確認後 ★)

---

## 9. fabrication 防止 (honest 留意)

### 9-1. release timing は実測引用

- TYB 17:00 JST = `docs/TYB_LEAK_AUDIT_2026_05_16.md` §1.1 (HTTP HEAD Last-Modified)
- SKB POST-RACE = Session #38 (CLAUDE.md 確定)
- 29 datatype mapping = `docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md` §1, §2-2

### 9-2. corr_target 値の根拠

- V15 cache 実測: 527,280 rows × 145 features、 finish ≤ 3 binary target
- v15.2 候補: EXTERNAL_CORR_PRIOR (Sub-task 18 §3 実測値、 jrdb_sed.csv 由来)
- paci_info_idx +0.4139 は Sub-task 18 §3-2 注記の通り **odds-related 疑惑あり** → 5/24+ 実 audit 必須

### 9-3. 「leak 確定」 と 「leak suspect」 の厳格区別

| 用語 | 意味 |
|------|------|
| **LEAK** (verdict) | datatype-level で release timing が POST-RACE 確定 (SKB / TYB / hjc / 払戻系)、 学習禁止 |
| **HIGH_CORR_LEAK_SUSPECT** (flag) | datatype-level は safe だが corr_target が異常 high (>0.40)、 manual audit 必須 |
| **PREV_ONLY** (verdict) | datatype は post_race だが前走 ref として参照可、 current race に使われていないか name 規則確認 |

★ suspect は **leak 確定 ではない**。 paci_info_idx は corr 高だけで NO-GO ではなく、 monotonic 等 detail audit 後の判定 ★

### 9-4. 不検出 risk (本 framework が detect できない 種類)

| 種類 | detect 不可な理由 | 対策 |
|------|------|------|
| **expanding window 違反 (dam_top3r 級)** | datatype は safe / corr も微妙、 集計 logic がリーク | FE 設計時 expanding 厳守 + LIVE retro で shift > 12x なら detect |
| **未知 datatype の新規 feature** | mapping に未登録 | unknown verdict で WARN、 mapping 更新必要 |
| **複合 leak (2 features の interaction で initial leak)** | 単独 corr では出ない | FE PoC で全 fold improvement 確認 |

---

## 10. 完了通知

```
T4 完了、 V15 leak = 0 件 PASS (145/145 OK)、
v15.2 候補 (22 features) 事前 audit: paci_info_idx (+0.4139) を HIGH_CORR_LEAK_SUSPECT として flag、
学習 gate ready (tools/v15_2_train_gate.py、 exit 0/1/2)、
test 35/35 PASS (tests/T4_leak_audit_test.py)、
V15 production 影響 0、 commit/push なし (親集中)、
実 v15.2 学習実行なし (gate 設計のみ)。
```

---

## Appendix A: ファイル一覧

| file | 用途 | 行数 |
|------|------|------|
| tools/leak_audit_automation.py | core audit module + CLI | ~470 |
| tools/v15_2_train_gate.py | v15.2 学習前 gate (exit 0/1/2) | ~140 |
| tests/T4_leak_audit_test.py | 35 tests (unittest) | ~290 |
| data/v15_2/features_v152_candidates.txt | v15.2 候補 22 features list | 22 |
| data/v15_2/v15_leak_audit_2026_05_17.json | V15 audit 結果 JSON | — |
| data/v15_2/v152_candidates_leak_audit_2026_05_17.json | v15.2 候補 audit 結果 JSON | — |
| docs/T4_LEAK_AUDIT_AUTOMATION_2026_05_17.md | 本 doc | — |

## Appendix B: 既存 docs 参照

- docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md (Sub-task 18 全体)
- docs/TYB_LEAK_AUDIT_2026_05_16.md (P0-3 / Sub-task 11、 TYB 永久放棄)
- CLAUDE.md Session #38 (V15.1 SKB POST-RACE LEAK 確定)

---

**END OF DOC**
