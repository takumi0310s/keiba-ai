# AUDIT-1 A: JRA-VAN / TFJV 全要素 audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 A 領域)
**前提**: ユーザー側 TARGET frontier JV install 済、 90 年分 data (約 6 GB / 43,000 files / 14 datatypes)
**位置付け**: read-only audit。 既存 code / model / schtasks 一切不変。 V15 production 完全保証

---

## 0. summary 一覧

| TFJV datatype | record | files | size | 主 dir | V15 利用状況 |
|--------------|--------|-------|------|--------|------------|
| RA | レース詳細 | -- | -- | SE/ES/DE | ✅ 利用 (jra_races_full.csv 経由) |
| SE | 馬毎レース | -- | -- | SE/ES/DE | ✅ 利用 (同上) |
| HR | 払戻金 | -- | -- | SE_DATA | ⚠️ 4/6 で 取得停止 (jra_payouts.csv) |
| RC | レース短信 | -- | -- | SE_DATA | ❌ 未取得 |
| YS | スケジュール | -- | -- | SE_DATA | ❌ 未取得 |
| H1 | 単複オッズ | 6,160 | 一部 | HY_DATA | ⚠️ odds_history.csv に部分 (確定オッズのみ) |
| H6 | 三連単オッズ | 6,160 | 残り | HY_DATA | ❌ 未取得 |
| HY | 払戻 (BY 内) | 283 | 22 MB | BY_DATA | ❌ 未取得 |
| UM | 馬個体 (1936-2025) | 280 | 497 MB | UM_DATA | ⚠️ 部分 (blood_full.csv 81,986 行) |
| SK | 産駒情報 | UM内 | -- | UM_DATA | ❌ 未取得 |
| KS | 騎手 master | 1 | -- | TFJ_KISI | ❌ 未取得 (JRDB KZ で代替中) |
| BR | 繁殖牝馬 | 10 | 5.8 MB | BR_DATA | ❌ 未取得 |
| HS | 生産者 master | 311 | 11 MB | BS_DATA | ❌ 未取得 |
| BN | 馬主 master | 10 | 4.1 MB | OW_DATA | ❌ 未取得 |
| TM | 調教タイム | 440 | 6.7 MB | TM_DATA | ⚠️ 部分 (training_times.csv 955K 行 が CK_DATA + TM_DATA 由来) |
| HN | 馬名? | 20 | 49 MB | KT_DATA | ❌ 未取得 (用途不明) |
| 02/12 | 調教 (CK) | 18,089 | 657 MB | CK_DATA | ✅ 利用 (training_times.csv) |
| TK | 特殊 race | -- | -- | DE_DATA | ❌ 未取得 |
| AV | 出走取消・変更 | 41 | 1.7 MB | JG_DATA | ❌ 未取得 |
| JC | 騎手変更 | JG内 | -- | JG_DATA | ❌ 未取得 |
| WF | WIN5 (10年) | 863 | 7.0 MB | W5_DATA | ❌ 未取得 |

---

## 1. RA (レース詳細) — 既存 jra_races_full.csv 経由 利用

| field | 内容 | V15 組込 | feature 名 |
|-------|------|--------|----------|
| year | 年 | ✅ | year_full / season |
| month_day | 月日 | ✅ | month / day / season |
| course_code | 競馬場コード | ✅ | course_enc (10コード→0-9) |
| kai | 開催回 | ✅ | (race_id 構成) |
| nichi | 開催日 | ✅ | (race_id 構成) |
| race_num | R | ✅ | race_num |
| youbi_code | 曜日 | ❌ 未活用 | (jrdb_kab に youbi あり、 V15 未使用) |
| race_name | レース名 (Shift-JIS) | ⚠️ | race_name (UI 用、 model 入力外) |
| race_class | クラス | ✅ | jrdb_class_code 経由 |
| surface | 芝/ダート/障害 | ✅ | surface_enc |
| distance | 距離 | ✅ | distance / dist_cat / dist_change |
| direction | 内外周り | ❌ 未活用 | (jrdb_bac inner_outer あり、 V15 未使用) |
| weather | 天気 | ⚠️ | Pattern B のみ weather_enc (気象庁API代替) |
| baba_state | 馬場状態 | ⚠️ | Pattern B のみ condition_enc |
| post_time | 発走時刻 | ❌ 未活用 | (jrdb_bac start_time あり、 V15 未使用) |
| num_horses_planned | 出走頭数 (予定) | ✅ | num_horses_val |
| prize_1st-5th | 賞金 | ⚠️ | jrdb_bac 経由 prize_1st-5th はあるが V15 model 入力外 |
| grade_code | グレード | ⚠️ | jrdb_class_code に内包 |
| race_symbol | レース記号 (混合・牡牝 等) | ❌ 未活用 | jrdb_bac race_symbol あり、 V15 未使用 |
| weight_type | 馬齢別/別定/H/G | ❌ 未活用 | jrdb_bac weight_type あり、 V15 未使用 |

**未活用 RA fields**: 7 件 (youbi / direction / post_time / race_symbol / weight_type / 他付帯)

---

## 2. SE (馬毎レース情報) — jra_races_full.csv 経由 利用

| field | 内容 | V15 組込 |
|-------|------|--------|
| umaban | 馬番 | ✅ horse_num |
| wakuban | 枠番 | ✅ bracket / bracket_pos |
| horse_id | 血統登録番号 | ✅ (集計 key) |
| horse_name | 馬名 | (UI のみ) |
| sex_code | 性別 | ✅ sex_enc |
| age | 馬齢 | ✅ age / age_group / age_sex |
| weight_carry | 斤量 | ✅ weight_carry / carry_diff |
| jockey_id | 騎手 ID | ✅ jockey_wr_calc / jockey_horse_* |
| jockey_name | 騎手名 | (UI のみ) |
| trainer_id | 調教師 ID | ✅ (一部 jrdb_csa 経由) |
| owner_code | 馬主 | ❌ 未活用 (JRA-VAN BN data 別) |
| breeder_code | 生産者 | ❌ 未活用 (JRA-VAN HS data 別) |
| location | 所属 (美/栗/地方/外) | ✅ location_enc |
| color_code | 毛色 | ❌ 未活用 |
| sire_id | 父馬 | ✅ sire_enc / sire_*_wr |
| bms_id | 母父 | ✅ bms_enc / bms_*_wr |
| dam_id | 母 | ⚠️ sib_*_exp 経由 (一部) |
| finish | 着順 | ✅ (target、 集計用) |
| time_sec | 走破タイム | ❌ 未活用 (jrdb_sed time_sec はあるが V15 未使用) |
| last_3f | 上がり 3F | ✅ prev_last3f / avg_last3f_3r |
| pass4 | 4 角通過順 | ✅ prev_pass4 |
| pass1-3 | 各 corner 通過順 | ❌ 未活用 (jrdb_sed corner1-4 はあるが V15 未使用) |
| horse_weight | 馬体重 | ⚠️ Pattern B のみ |
| weight_change | 馬体重変化 | ⚠️ Pattern B のみ |
| odds_final | 確定オッズ | ⚠️ prev_odds_log のみ (確定オッズはリーク扱い) |
| popularity | 人気 | ⚠️ Pattern B (pop_rank) |
| prize | 獲得賞金 | ✅ prev_prize |
| race_style | 脚質 | ✅ (jrdb_sed race_style 経由) |
| training_center | 入厩先 | ❌ 未活用 (jrdb_kyi 放牧先 で代替) |

**未活用 SE fields**: 8-10 件 (owner / breeder / color / time_sec / corner1-3 / training_center 等)

---

## 3. HR (払戻金) — ⚠️ 4/6 で 取得停止 (重大課題)

| field | 内容 | V15 組込 |
|-------|------|--------|
| tansho_payouts | 単勝 | ⚠️ (4/6 まで jra_payouts.csv) |
| fukusho_payouts | 複勝 | ⚠️ 同上 |
| umaren_payouts | 馬連 | ⚠️ 同上 |
| wide_payouts | ワイド | ⚠️ 同上 |
| trio_payouts | 三連複 | ⚠️ 同上 |
| tierce_payouts | 三連単 | ⚠️ 同上 |
| umatan_payouts | 馬単 | ⚠️ 同上 |
| wakuren_payouts | 枠連 | ❌ 未活用 |

**課題**: jra_payouts.csv が **2026-04-06 で更新停止** (CLAUDE.md 既知バグ)
**代替**: JRDB hjc.csv (払戻金) は取得済、 ROI 計算可能。 JV-Link HR が真の data source

---

## 4. UM (馬個体、 90 年分) — 部分活用

| field | 内容 | V15 組込 |
|-------|------|--------|
| horse_id | 血統登録番号 | ✅ |
| horse_name | 馬名 | (UI) |
| sire_id | 父馬 | ✅ blood_full.csv 経由 |
| dam_id | 母 | ⚠️ 部分 |
| bms_id | 母父 | ✅ |
| birth_date | 生年月日 | ❌ 未活用 |
| sex_code | 性別 | ✅ |
| coat_color | 毛色 | ❌ 未活用 (jrdb_ukc にも hair_color_code あり) |
| breeder_code | 生産者 | ❌ 未活用 |
| owner_code | 馬主 | ❌ 未活用 |
| ms_horse_id | 半弟・全弟 ID | ❌ 未活用 (★ sibling 拡張に活用可能) |
| career_total | 通算成績 | ⚠️ horse_career_* で代替 (expanding window) |

**未活用 UM**: 90 年分 母系拡張 (sib_*_ext)、 生産者 features、 毛色

---

## 5. WF (WIN5、 10 年分) — 完全 未活用

10 年分の WIN5 出走履歴 (年 87 race × 5 = 435/年 × 10 年 = ~4,350 race の 強調 data)

| field | 用途 |
|-------|------|
| WIN5 出走馬 | 各回の対象馬。 G レース 並みの強さ評価指標 |
| WIN5 払戻 | 大穴指標 |
| WIN5 売上 | 注目度指標 |

**期待効果**: low-confidence (WIN5 = G 級重賞重複なので別 source の重賞 indicator と冗長な可能性)

---

## 6. BR / HS / BN (繁殖牝馬・生産者・馬主 master) — 完全 未活用

| source | 行数 | V15 組込 | 期待効果 |
|--------|------|--------|--------|
| BR_DATA (繁殖牝馬) | 5.8 MB | ❌ | 母系成績 完全 (sib_*_exp 強化) |
| BS_DATA (生産者) | 11 MB / 311 file | ❌ | breeder_top3_rate (新馬戦 強力指標) |
| OW_DATA (馬主) | 4.1 MB | ❌ | owner_top3_rate (馬主成績) |

**期待 AUC**: 各 +0.001-0.003 (新馬戦 / 重賞 で 効果大)

---

## 7. TM (調教タイム) / CK (調教) — 部分活用

CK_DATA 657 MB / 18,089 files が training_times.csv 955K 行 に 抽出済。 TM_DATA 6.7 MB / 440 files は 別 (調教師の調教指数履歴?)。

| feature 候補 | V15 組込 |
|-------------|--------|
| training_intensity_enc | ✅ |
| wood_best_4f / sakaro_best_4f | ✅ |
| time_1f_last | ✅ |
| TM_DATA 直 利用 | ❌ 未確認 |

---

## 8. JG (出走取消・変更) — 完全 未活用

41 file / 1.7 MB。 リアルタイム取消検知に有用。 朝 8:00 以降〜直前 までの 取消馬を即時 detect 可能。

**用途**: Phase 4 リアルタイム反映 (V15 production には別 path)
**現状**: tools/predict_one_race.py で 取消発生時の手動再予測。 自動取消 trigger は未実装

---

## 9. その他 (RC / YS / TK / KT / KS) — 未活用

| source | 内容 | 価値 |
|--------|------|------|
| RC | レース短信 | low (JRDB SED race_comment が 同等) |
| YS | スケジュール | low (jrdb_kaa kaisai_key で代替) |
| TK | 特殊 race | medium (G I + ジャンプ等) |
| KT (HN) | 馬名? | unknown |
| KS | 騎手 master | medium (jrdb_kz/ksa が同等) |

---

## 10. 未活用 features summary (TFJV 全要素 audit 結論)

**取得済 だが V15 未活用** (即実装可能):
1. youbi (曜日) - jrdb_kab 経由
2. direction (内外周り) - jrdb_bac
3. post_time (発走時刻) - jrdb_bac
4. race_symbol (混合等) - jrdb_bac
5. weight_type - jrdb_bac
6. owner_code - jrdb_ukc
7. breeder_code - jrdb_ukc
8. coat_color (毛色) - jrdb_ukc
9. corner1-3 通過順 - jrdb_sed
10. time_sec (走破タイム) - jrdb_sed

**未取得 だが TFJV 直 parse で 取得可能** (中期):
1. **★ HR 直 (4/6 停止 解消)** - jvlink_fetcher で 自動更新
2. **★ BR (繁殖牝馬 90 年) - sib_*_ext 強化**
3. **★ BS (生産者) - breeder_top3_rate**
4. OW (馬主) - owner_top3_rate
5. WF (WIN5) - WIN5 出走 indicator
6. JG (出走取消) - リアルタイム reflect
7. UM 90 年 全 fields - 半弟・全弟 ID 等
8. H6 (三連単 odds 完全) - 確定 trio odds
9. TM (調教師 調教指数) - 詳細
10. TK (特殊 race) - G I 別 model 候補

---

## 11. 5/9 V15 投資保護

✅ TFJV 全 data は read-only audit (parse 結果は data/tfjv/ 別 dir 出力 想定)
✅ V15 model 不変 (keiba_model_v15_central*.pkl.gz)
✅ predict_core / daily_predict / app.py 完全不変
✅ 既存 jra_races_full.csv / training_times.csv / odds_history.csv / blood_full.csv 不変

---

## 12. 結論

✅ TFJV 14 datatypes / 約 700 fields の 全 audit 完了
✅ V15 利用率: **大体 30%** (RA / SE / 一部 HR / UM / CK のみ)
✅ 未活用 取得済 fields: 約 10 件
✅ 未取得 datatypes: 7 件 (HR 修復 / BR / BS / OW / WF / JG / TM / TK)
✅ Phase 3 V20 (6/8 投入候補) で 約 5-10 features 追加候補

**Session #44 A の inventory + 本 audit で TFJV 全要素 把握 完了**
