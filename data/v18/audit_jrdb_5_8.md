# AUDIT-1 C: JRDB 全要素 audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 C 領域)
**前提**: JRDB Advance 加入済 (月額 2,880 円)、 26 datatypes (一部取得)
**位置付け**: read-only audit。 既存 download_jrdb / parse_jrdb_* / model 不変

---

## 0. summary (datatype 別)

### 0.1 取得済 (10+ datatypes)

| datatype | 行数 | V15 組込 | 主要 fields |
|---------|------|--------|------------|
| KYI (paci) | 290,981 / 548,607 | ✅ JRDB_FEATURES_PRE_RACE 22 件 (主軸) | IDM、 騎手指数、 調教指数、 厩舎指数、 総合指数、 激走指数、 ペース予想、 位置予想、 重適性、 蹄、 放牧先ランク、 厩舎ランク、 入厩 etc |
| KAB (場別) | 3,279 | ⚠️ 部分 (turf_baba_code 経由 condition_enc) | 場・天候・芝/ダ馬場差・直線馬場差 |
| SED (成績) | 547,774 | ✅ JRDB_FEATURES_PREV_RACE 8 件 (前走成績) | 着順、 distance、 time_sec、 IDM、 上がり、 ペース、 corner1-4、 race_style |
| TYB (直前) | 548,113 | ✅ Pattern B 5 件 (paddock/odds/composite/body/demeanor) | パドック指数、 オッズ指数、 直前 IDM、 馬体、 気配 |
| CYB (調教分析) | 548,607 | ⚠️ 部分 (train_arrow / train_eval 等) | 調教種別、 コース、 馬場、 印、 量、 変化、 評 |
| SKB (成績拡張) | 547,101 | ❌ **POST-RACE LEAK 確定 (Session #38 で V15.1 で 全 10 features 完全除外)** | kishi_code_1-3, baba_code_1-8, kyaku_code_1-3, padock/kyaku/baba/race comment, turf_hoof, run_stage 等 |
| SRB (成績レース、 ハロン・バイアス) | 21,592 | ❌ 未組込 (★ 高優先) | furlong_times, corner1-4_order, pace_up_pos, bias_1corner-straight |
| ZE (前走) | 537,387 | ⚠️ 部分 (jrdb_prev_* 経由) | SED 同等 (前走分) |
| ZK (前走拡張) | 530,100 | ❌ 未組込 | SKB 同等 (前走、 kishi/baba/kyaku codes) |
| JO (情報) | 301,719 | ❌ 未組込 | 想定オッズ、 予想オッズ、 CID 指数、 LS 指数、 EM、 外厩 BB |
| KZ (騎手 master) | -- | ❌ 未組込 | 騎手 leading / turf_wr / 着回数 (年/前年/通算) |
| CZ (調教師 master) | -- | ❌ 未組込 | 調教師 leading / turf_wr / 着回数 |
| UKC (馬基本) | -- | ⚠️ 部分 (sire/bms 経由) | 父名・母名・母父名・誕生日・father_code・bms_code |
| KKA (競走馬拡張) | -- | ❌ 未組込 | JRA 成績、 交流成績、 距離・トラック・重・休養・class・季節・枠・坂 別 |
| BAC (場番組) | -- | ⚠️ 部分 | race_date、 race_type、 race_cond、 grade、 race_name、 prize_1st-5th |
| HJC (払戻金) | -- | ⚠️ ROI 計算用 | tansho/fukusho/wakuren/umaren/wide/umatan/trio/tierce 払戻 |
| OZ (基準オッズ) | -- | ❌ 未組込 | 単勝/複勝 オッズ tabular |
| OW (ワイド) | -- | ❌ 未組込 | wide_min/max/median |
| OU (馬連) | -- | ❌ 未組込 | umaren_count/min/max/median/p10 |
| OT (三連複) | -- | ❌ 未組込 | trio_count/min/max/median/p10 |
| OV (三連単) | -- | ❌ 未組込 | tierce_count/min/max/median/p10 |
| KAA (場別拡張) | -- | ❌ 未組込 | KAB の昔 ver |
| KTA (出走馬詳細) | -- | ❌ 未組込 | jockey_name、 trainer_name、 prize、 condition_class、 turf_apt、 dirt_apt 等 |
| CSA (調教師拡張) | -- | ❌ 未組込 | KZ の昔 ver |
| KSA (騎手拡張) | -- | ❌ 未組込 | CZ の昔 ver |
| JOA (場別情報) | -- | ⚠️ 部分 | track 情報、 distance、 surface、 inner_outer 等 |
| CHA (調教詳細) | -- | ❌ 未組込 | oikiri_date、 oikiri_count、 oikiri_course、 oikiri_shurui、 oikiri_aite、 oikiri_rank、 ten_time、 chukan_time、 shimai_time、 awase_*、 oikiri_idx |

### 0.2 未取得 (公式に存在するが download_jrdb で 未取得)

| datatype | 内容 | 価値 | 工数 |
|---------|------|------|------|
| **CH (調教本追切)** | 重賞前 本追切 詳細 | medium | 1h |
| (上記 CHA で代替されている可能性 high) | -- | -- | -- |

→ JRDB の 26 datatypes は ほぼ download_jrdb.py で 取得カバー済 (一部別名 alias)

---

## 1. KYI (取得済 主軸) - 70 fields

```
場コード,年,回,日,R,馬番,血統登録番号,馬名,IDM,騎手指数,情報指数,予備1-3,
総合指数,脚質,距離適性,上昇度,ローテーション,基準オッズ,基準人気順位,
基準複勝オッズ,基準複勝人気順位,人気指数,調教指数,厩舎指数,調教矢印コード,
厩舎評価コード,騎手期待連対率,激走指数,蹄コード,重適性コード,クラスコード,
総合印,IDM印,情報印,騎手印,厩舎印,調教印,激走印,芝適性コード,ダ適性コード,
騎手コード,調教師コード,テン指数予想,ペース指数予想,上がり指数予想,位置指数予想,
ペース予想,取消フラグ,性別コード,激走順位,LS指数順位,テン指数順位,ペース指数順位,
上がり指数順位,位置指数順位,騎手期待単勝率,騎手期待3着内率,輸送区分,降級フラグ,
激走タイプ,入厩何走目,入厩年月日,入厩何日前,放牧先,放牧先ランク,厩舎ランク,
jra_race_id,nk_race_id
```

V15 組込 (22 件):
- 数値指数: IDM, 調教指数, 厩舎指数, 情報指数, 総合指数, 激走指数 (6)
- 予想指数: テン/ペース/上がり/位置 指数予想 (4)
- コード: クラス、 上昇度 (rise_code)、 重適性、 蹄、 放牧先ランク、 厩舎ランク (6)
- その他: 入厩何日前、 入厩何走目、 調教矢印、 厩舎評価、 脚質、 距離適性 (6)

V15 **未組込 KYI fields** (Top):
1. **基準オッズ / 基準人気順位** (2) - 朝オッズ近似 だが 確定 odds 並みに 強力リーク source 注意
2. 基準複勝オッズ / 基準複勝人気順位 (2)
3. 人気指数 (paci_ninki_idx は v14.1 で 採用済、 KYI 経由は重複)
4. **総合印 / IDM印 / 情報印 / 騎手印 / 厩舎印 / 調教印 / 激走印** (7) - paci の sogo_mark/idm_mark/jockey_mark/train_mark は v15 master で 採用済 (4/7)
5. 騎手期待連対率 / 単勝率 / 3 着内率 (3) - v14.1 で 採用済
6. 激走タイプ / 激走順位 / LS 指数順位 / テン/ペース/上がり/位置 順位 (7)
7. 騎手コード / 調教師コード (2) - jockey_id / trainer_id 別系
8. 取消フラグ (1) - リアルタイム取消反映で 重要 (現在 V15 未組込)

---

## 2. SED (前走成績) - 48 fields

```
race_id,umaban,horse_name,finish,abnormal,distance,surface_code,num_horses,time_sec,
weight_carry,odds_final,popularity,idm,soten,baba_sa,ten_idx,agari_idx,pace_idx,
race_pace_idx,pace,deokure,ichi_dori,furi,mae_furi,naka_furi,ato_furi,race_idx,
course_dori,josho_code,batai_code,kehai_code,race_pace,uma_pace,first_3f,last_3f,
corner1,corner2,corner3,corner4,horse_weight,weight_diff,race_style,jockey_code,
trainer_code,blood_num,yyyymmdd,race_name,grade,class_code
```

V15 組込 (8 件、 jrdb_prev_*):
- jrdb_prev_idm, jrdb_prev_track_bias (baba_sa 経由), jrdb_prev_interference (furi 経由), jrdb_prev_late_start (deokure 経由), jrdb_prev_ten_idx, jrdb_prev_agari_idx, jrdb_prev_pace_idx, jrdb_prev_rise_code

**SED 未組込 fields** (Top):
1. **time_sec** (走破タイム) - 速度指標、 距離別 norm 化必要
2. **first_3f / last_3f** - 前走前半・後半 3F
3. **corner1-4** (corner 通過順) - 道中位置
4. **race_pace / uma_pace** (個馬・全体ペース)
5. **mae_furi / naka_furi / ato_furi** (前・中・後 不利)
6. **josho_code / batai_code / kehai_code** (上昇・馬体・気配 code)
7. **race_idx** (レース指数)
8. **course_dori / ichi_dori** (コース取り、 位置取り)

→ ほぼ V15 で カバー済 (SED の 8 件で)、 残り は 細かい補強候補

---

## 3. TYB (直前) - 26 fields

```
race_id,umaban,idm,jockey_idx,info_idx,odds_idx,padock_idx,sogo_idx,
bagu_change,ashimoto,cancel_flag,tansho_odds,fukusho_odds,horse_weight,
weight_diff,weight_carry,odds_mark,padock_mark,sogo_mark,batai_code,kehai_code,
jockey_code,jockey_name,baba_code,weather_code,start_time
```

V15 (Pattern B) 組込 (5 件):
- jrdb_paddock_idx, jrdb_odds_idx, jrdb_live_composite_idx, jrdb_body_code (batai_code), jrdb_demeanor_code (kehai_code)

**TYB 未組込 fields**:
1. **bagu_change** (馬具変更) - 中
2. **ashimoto** (馬装) - 中
3. **cancel_flag** (取消フラグ) - リアルタイム取消反映で 重要
4. **tansho_odds / fukusho_odds** (直前確定オッズ) - リーク扱い OK
5. **odds_mark / padock_mark / sogo_mark** (印 直前版) - 既 paci の Tier B mark で 部分 採用

---

## 4. CYB (調教分析) - 13 fields

```
umaban,train_type,train_course_type,train_baba,train_mark,train_amount,
train_change,train_comment,comment_year,comment_date,train_eval,train_course,race_id
```

V15 組込 (部分):
- jrdb_training_arrow (KYI 由来)
- jrdb_paddock_idx (TYB 由来)
- training_intensity_enc (netkeiba 由来)

**CYB 未組込 fields**:
1. **train_type** (調教種別)
2. **train_course_type** / **train_course** (コース種別)
3. **train_baba** (調教馬場)
4. **train_mark** (調教印 ◎○▲△×)
5. **train_amount** (調教量)
6. **train_change** (調教変化)
7. **train_eval** (調教評)

→ 期待 +0.001-0.003 (5-7 features)

---

## 5. SKB (成績拡張) - 32 fields ★ POST-RACE LEAK 確定

```
race_id,umaban,blood_num,yyyymmdd,kishi_code_1-6,baba_code_1-8,kyaku_code_1-6,
padock_comment,kyaku_comment,baba_comment,race_comment,turf_hoof,run_stage,
anshin,heavy_apt,aisho,niwa
```

**Session #38 確定**: SKB POST-RACE LEAK
- skb_kishi_code_3 単独で feature_importance +480bp、 corr_target 0.137
- 1着馬 0-rate 15% / 敗者 49% で monotonic
- finish と直接相関を持ち、 V15.1 採用 NO-GO
- V20 で **全 10 features 完全除外** (V20_LEAK_FEATURES)

---

## 6. SRB (成績レース、 ハロン・バイアス) ★ 高優先 未組込

```
race_id,furlong_times,corner1_order,corner2_order,corner3_order,corner4_order,
pace_up_pos,bias_1corner,bias_2corner,bias_backstr,bias_3corner,bias_4corner,
bias_straight,race_comment
```

**価値**: ★★★ 最高優先 (Session 1 で audit 済み、 取得済だが V15 未組込)

| field | 内容 | 期待 |
|-------|------|------|
| furlong_times | レース 1F ごと タイム | コース癖 / 距離別 で +0.002 |
| corner1-4_order | 各 corner 通過順 (集計) | レースペース指標 |
| pace_up_pos | ペース上げ位置 | 中 |
| bias_1corner-straight | 各位置 馬場バイアス | ★ track_bias の真値、 +0.003-0.005 |

→ **SRB は最優先取得 (Sprint 4 候補)**、 期待 AUC +0.003-0.005

---

## 7. ZE / ZK (前走拡張)

ZE: SED とほぼ同 fields (前走分)。 V15 jrdb_prev_* と内容重複
ZK: SKB と同 fields (前走分、 リーク risk 同様)

---

## 8. JO (情報) - 18 fields

```
umaban,blood_num,horse_name,soten_odds,yoso_odds,cid_soten_idx,cid_sara_idx,
cid_idx,cid,ls_idx,ls_eval,em,gaisha_bb,gaisha_bb_wr,gaisha_bb_rensho,
breeder_bb,breeder_bb_wr,breeder_bb_rensho,race_id
```

**価値**: medium-high (未組込)

| field | 内容 | 期待 AUC |
|-------|------|---------|
| soten_odds / yoso_odds | 想定 / 予想オッズ | +0.001 (朝オッズ近似) |
| cid_idx / ls_idx | CID 指数 / LS 指数 | +0.001-0.003 |
| em | 馬印度 (M) | +0.0003 |
| gaisha_bb / gaisha_bb_wr / gaisha_bb_rensho | 外厩 BB 系 (3) | +0.001-0.002 |
| breeder_bb / breeder_bb_wr / breeder_bb_rensho | 生産者 BB 系 (3) | +0.001-0.002 |

→ **JO 高優先**、 期待 AUC +0.005

---

## 9. KZ / CZ / KSA / CSA (騎手 / 調教師 master)

**KZ (騎手 master) 主要 fields** (44 cols):
- year_leading, year_turf_wr, year_special_w
- last_leading, last_turf_wr, last_special_w
- year_turf_1st-out, year_dirt_1st-out, last_turf_1st-out, last_dirt_1st-out
- total_turf_1st-out, total_dirt_1st-out

V15 組込: jockey_wr_calc / jockey_course_wr / jockey_surface_wr (内製 expanding)
**KZ 直 利用**: ❌ 未組込
**価値**: medium (内製 expanding と冗長な可能性、 +0.0005-0.001)

**CZ (調教師 master)**: 同様、 V15 では trainer 集計 features 限定的

---

## 10. UKC (馬基本)

```
blood_num,horse_name,sex_code,hair_color_code,keito_code,father_name,mother_name,
bms_name,birthday,father_birth_year,mother_birth_year,bms_birth_year,owner_name,
owner_code,breeder_name,birthplace,register_del_flag,data_date,father_code,bms_code
```

V15 組込: father_code (sire_enc), bms_code (bms_enc)、 sex_code (sex_enc)

**UKC 未組込 fields**:
1. **hair_color_code** (毛色) - low (+0.0001)
2. **keito_code** (系統) - medium (+0.001-0.002、 父系統 別 適性)
3. **father_birth_year / mother_birth_year / bms_birth_year** - low (種牡馬世代)
4. **owner_code** - medium (馬主成績 +0.001-0.002)
5. **breeder_name** - medium (生産者成績 +0.001-0.002)
6. **birthplace** - low

---

## 11. KKA (競走馬拡張)

```
race_id,umaban,
jra_seiseki_1-out,koryu_seiseki_1-out,kyori_seiseki_1-out,track_seiseki_1-out,
heavy_seiseki_1-out,rest_seiseki_1-out,class_seiseki_1-out,season_seiseki_1-out,
waku_seiseki_1-out,saka_seiseki_1-out,speed_seiseki_1-out,
dam_rensho_max/min/avg, bms_rensho_max/min/avg
```

**KKA 未組込 fields** (V15 横展開で 強力候補):
- jra_seiseki / koryu_seiseki: 中央/交流 着順 (1/2/3/out 4 値)
- **kyori_seiseki**: 距離 別 着順 (V15 horse_dist_top3r と冗長 だが 直接 値 利用可)
- **track_seiseki**: トラック 別 (左右内外)
- **heavy_seiseki**: 重 別 (V15 重 適性 と 補完)
- **rest_seiseki**: 休養 別
- **class_seiseki**: クラス 別
- **season_seiseki**: 季節 別
- **waku_seiseki**: 枠 別 (V15 frame_course_dist_wr と補完)
- **saka_seiseki**: 坂 別
- **speed_seiseki**: 速度 別
- **dam_rensho_*** / **bms_rensho_***: 母産駒 / 母父産駒 連勝 (sib_*_exp と補完)

→ KKA 12 group × 4 値 = 48 features、 期待 +0.003-0.008 (一部冗長)

---

## 12. CHA (調教詳細)

```
umaban,race_id,youbi,oikiri_date,oikiri_count,oikiri_course,oikiri_shurui,
oikiri_aite,oikiri_rank,oikiri_naiyou,ten_time,chukan_time,shimai_time,
ten_time_idx,chukan_time_idx,shimai_time_idx,oikiri_idx,awase_result,
awase_shurui,awase_nenrei,awase_class
```

**CHA 未組込 fields**:
- oikiri_date, oikiri_count - low
- **oikiri_course / oikiri_shurui** - 中
- **oikiri_aite** (併走相手) - 中
- **oikiri_rank** (調教 ランク A/B/C/D) - 中 (+0.001、 netkeiba training rank と同等)
- **ten_time / chukan_time / shimai_time / *_idx** - 各 6 - 中 (+0.001-0.002)
- **oikiri_idx** - 中 (+0.001)
- **awase_result / awase_shurui / awase_nenrei / awase_class** (併走結果) - 中

→ CHA 全 9 + features、 期待 +0.002-0.005

---

## 13. OZ / OW / OU / OT / OV (基準オッズ)

| datatype | fields | V15 組込 | 価値 |
|---------|--------|--------|------|
| OZ | tansho_01-18, fukusho_01-18, umaren_min/max | ❌ | 朝オッズ計算可、 +0.001-0.002 |
| OW | wide_min/max/median | ❌ | 賭け方推奨に使用可 |
| OU | umaren_count/min/max/median/p10 | ❌ | 配当分布、 ROI 期待値計算 |
| OT | trio_count/min/max/median/p10 | ❌ | 三連複 配当 期待値 |
| OV | tierce_count/min/max/median/p10 | ❌ | 三連単 配当 期待値 |

→ オッズ 系 5 datatypes は **配当期待値計算** で 強力。 model 入力としては 確定オッズ並みに リーク risk 注意

---

## 14. 未組込 features summary (JRDB)

**取得済 だが V15 未活用** (Top 15):
1. ★ **SRB bias_1corner-straight** (馬場バイアス) - +0.003-0.005
2. ★ **JO cid_idx / ls_idx** - +0.002-0.003
3. **KKA kyori_seiseki / track_seiseki / heavy_seiseki** - +0.002-0.005
4. ★ **CHA oikiri_rank / oikiri_idx / 3 time** - +0.002-0.005
5. **JO gaisha_bb / breeder_bb 6 件** - +0.002-0.004
6. **CYB train_mark / train_eval / train_amount 5 件** - +0.001-0.003
7. **TYB bagu_change / ashimoto / cancel_flag** - +0.001-0.002
8. **UKC keito_code / owner_code / breeder_name** - +0.002-0.005
9. **SED time_sec / first_3f / last_3f (前走 詳細 時計)** - +0.001-0.003
10. **OU/OT 配当分布** (ROI 期待値計算用) - 別目的
11. KZ/CZ leading 系 (騎手・調教師 leading) - +0.0005-0.001
12. KYI 印 7 件 (sogo/idm/jockey/train が paci 経由 採用済、 残り 3 件) - +0.0003-0.001
13. KAB 直線馬場差 / 場別 weather 詳細 - +0.001
14. JOA inner_outer / track 詳細 - +0.0003-0.001
15. KTA condition_class - +0.0005

**未取得**: ほぼなし (download_jrdb で 26 datatypes ほぼ全取得)

---

## 15. 5/9 V15 投資保護

✅ JRDB 全 CSV 全 read-only。 V15 model 不変
✅ download_jrdb / parse_jrdb_* / jrdb_features.py 完全不変
✅ daily_jrdb_kyi 6:00 タスク 不変、 SCRAPER-GUARD 動作不変

---

## 16. 結論

✅ JRDB 26 datatypes / 約 500 fields の 全 audit 完了
✅ V15 利用率: 取得済 fields の **約 25%** (KYI/SED/TYB 主軸 35 features のみ)
✅ 即実装候補: ★ **SRB / JO / KKA / CHA** 中心 で 期待 +0.010-0.020 AUC
✅ POST-RACE LEAK: SKB 確定 (V20 で 完全除外)、 ZK も同様 risk

**Sprint 4 候補 (短期)**: SRB bias 6 件 + JO cid/ls/em + CHA oikiri 3 件 = 期待 +0.005-0.010
