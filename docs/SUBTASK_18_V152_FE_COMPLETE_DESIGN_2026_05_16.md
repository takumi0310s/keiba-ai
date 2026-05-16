# Sub-task 18: v15.2 FE 完全設計 (TYB 放棄後の真の path)

**作成日**: 2026-05-16 (Session 親 sub-task 18)
**位置付け**: P0-4 TYB live fetch 永久放棄確定 (Sub-task 11) を受け、 v15.2 を **JRDB cha + sr + paci + kab + kta + kka_v2 主軸** で再設計
**先行 doc**: `docs/V152_FE_DESIGN_2026_05_16.md` (Sub-task 5-4、 TYB 主軸版) → 本 doc で **完全置換**
**V15 production 影響**: ★ 0% (predict_core / daily_predict / race_auto_notify / app.py / V15 .pkl.gz 不変) ★

---

## 0. 結論 (TL;DR)

| 項目 | 値 |
|------|----|
| **採用候補 N (priority A+B+C)** | **22 個** (A: 8 / B: 9 / C: 5) |
| **推奨採用 (A+B)** | **17 個** |
| **想定 +AUC delta (実測前 assumption)** | **+0.004 〜 +0.010** (V15 0.8939 → 0.898〜0.904) |
| **想定 +ROI delta** | **+2 〜 +6pt** (戦略⑦込み 140% → 142-146%) |
| **データ source (主軸 6 種)** | cha (追切) / sr (ラップ) / paci (展開) / kab (場別馬場) / kta (場別調教師) / kka_v2 (breeder 集計) |
| **永久除外** | tyb (P0-4 verdict) / skb (POST-RACE LEAK 確定) / hjc / oz/ou/ot/ov/ow (払戻系) |
| **着手時期** | 5/24+ (P0-3 PASS 不要、 cha/sr/paci/kab/kta は release timing 安全) |
| **実装期間** | 2-3 日 |
| **採用判定基準** | §5-3、 5 項目 ALL PASS が条件 (V15.1/V18/V19 NO-GO 教訓) |

**主軸 path 切替の理由**:
1. **TYB content 5 features = 採用候補 0** 確定 (Sub-task 6 / 11、 LEAK 3 + 信号 0 が 2)
2. **TYB delivery POST-RACE** 確定 (17:00 JST publish = 当日 race 後)
3. **cha (追切) / sr (ラップ) / paci (展開) は release timing 安全** (cha=3 日前、 sr/paci=前日まで、 kab=当日朝)
4. **V15 で paci 既に部分採用** (4 features TIER B)、 cha 既に 3 features 採用 → 「未使用 column」 中心に補強

---

## 1. JRDB 29 data type 棚卸し (全件 実測)

`data/jrdb_*.csv` に存在する **29 file**、 各 row/col、 V15 利用状況、 release timing。

| ID | file | rows | cols | V15 利用 | release timing | 主軸候補 | leak risk |
|----|------|-----:|-----:|----------|----------------|:--------:|-----------|
| **kyi** | jrdb_kyi.csv | 292,490 | 70 | ★main★ (jrdb_idm / class_code / heavy_apt 等 18 features) | **当日朝 06:00 確定** (KYI バッチ) | (既存 main) | low (pre-race) |
| **bac** | jrdb_bac.csv | 39,161 | 22 | △ (race meta、 distance/surface 既存と重複) | 前週木曜 (番組公表時) | (race meta) | low |
| **sed** | jrdb_sed.csv | 548,780 | 49 | ★main★ (jrdb_prev_* 7 features = 前走集計) | レース後 (前走として利用) | (既存 main) | low (prev-race) |
| **cha** | jrdb_cha.csv | 302,742 | 21 | ☆部分 (oikiri_idx / ten_time_idx / shimai_time_idx = 3 features) | **3-4 日前 確定** (中央木曜追切) | **★ priority A+B 主軸 ★** | low |
| **sr** | jrdb_sr.csv | 39,136 | 32 | ✗ 未使用 (race-level) | レース後 (前走 race 集計として利用) | △ (前走 race 集計のみ) | medium (current race=leak) |
| **kab** | jrdb_kab.csv | 3,296 | 55 | ✗ 未使用 (kaisai-level) | **当日朝 06:00** (KAB ファイル) | ★ priority B 候補 (race 横断統計) | low |
| **kta** | jrdb_kta.csv | 298,551 | 34 | ☆部分 (kta_idm / kta_ten_pred / kta_agari_pred = 3 features) | 当日朝 (KTA = 場別調教師) | △ (既存 3 + 拡張余地小) | low |
| **ksa** | jrdb_ksa.csv | 336 | 35 | ✗ 未使用 (騎手 master) | 静的 master (年次更新) | △ (騎手 metadata のみ) | low |
| **kz** | jrdb_kz.csv | 1,278 | 45 | ✗ 未使用 (騎手 v2 master) | 静的 master | (ksa と重複) | — |
| **paci** | jrdb_paci.csv | 549,604 | 63 | ☆部分 (paci_*_mark 4 + jockey_exp 3 + ninki_idx 等 7 = 計 7 features) | **当日朝 06:00** (PACI = 前日夜更新+朝 sync) | **★ priority A 主軸 (rank/zscore features) ★** | low (pre-race) |
| **cyb** | jrdb_cyb.csv | 512 | 32 | ✗ 未使用 (調教分析 v1、 size 小) | (低 fill) | ✗ (data 不足、 512 rows) | — |
| **cyb_v2** | jrdb_cyb_v2.csv | 1,874 | (差替) | ✗ 未使用 | (低 fill) | ✗ (data 不足) | — |
| **hjc** | jrdb_hjc.csv | 21,591 | 49 | ✗ 未使用 (払戻) | レース後配信 | **★ POST-RACE 永久除外 ★** | high |
| **skb** | jrdb_skb.csv | 547,100 | 33 | ★永久除外★ (Session #38 POST-RACE LEAK 確定) | レース後 (成績拡張) | **★ 永久除外 ★** | high |
| **tyb** | jrdb_tyb.csv | 550,115 | 26 | ✗ truncate (V15 model = 145 features、 5 TYB は merge bug で constant) | 17:00 JST publish (Sub-task 6/7/11) | **★ 永久除外 (Sub-task 11 verdict) ★** | high (delivery POST-RACE) |
| **ukc** | jrdb_ukc.csv | 36,939 | 20 | ☆部分 (sire / bms / breeder 静的属性) | 静的 master | △ (既存集計済) | low |
| **ze** | jrdb_ze.csv | 537,386 | 49 | ★main★ (jrdb_ze_idm_avg / ten_avg / agari_avg / furi_count = 4 features) | レース後 (過去 5 走集計として利用) | (既存 main) | low (prev-races) |
| **zk** | jrdb_zk.csv | 530,099 | 34 | ✗ 未使用 | レース後 (前走着差等の補助) | △ (sed と重複) | medium |
| **srb** | jrdb_srb.csv | 21,591 | 14 | ☆部分 (jrdb_tb_homestr_inner = 1 feature) | レース後 (track bias、 前走として利用) | △ (拡張余地小) | medium |
| **kka** | jrdb_kka.csv | 546,925 | 52 | ✗ 未使用 (旧 schema、 全 0%) | — | ✗ (kka_v2 に置換) | — |
| **kka_v2** | jrdb_kka_v2.csv | 548,606 | (拡張) | ☆部分 (dam_rensho_avg / bms_rensho_avg = 2 features) | レース後集計だが breeder は 静的 | **★ priority B 主軸 (breeder_*) ★** | low |
| **kka_features** | jrdb_kka_features.csv | 548,606 | 多 (356 MB) | ✗ 未使用 | (未確認、 巨大) | △ (要 column 詳細監査) | — |
| **joa** | jrdb_joa.csv | 8,013 | 21 | ✗ 未使用 (race meta v2) | 番組発表時 | △ (bac と重複) | low |
| **jo** | jrdb_jo.csv | 302,742 | 19 | ★main★ (jrdb_cid_idx / jrdb_ls_idx = 2 features) | 当日朝 (CID/LS 指数) | (既存 main、 拡張余地 small) | low |
| **csa** | jrdb_csa.csv | 362 | 33 | ✗ 未使用 (調教師 master) | 静的 | △ (master のみ) | — |
| **kaa** | jrdb_kaa.csv | 1,805 | 28 | ✗ 未使用 (kaisai 旧 schema) | (kab に置換) | ✗ | — |
| **cz** | jrdb_cz.csv | 1,344 | 43 | ✗ 未使用 (調教師 v2 master) | 静的 | (csa と重複) | — |
| **ot / ou / ov / ow / oz** | (払戻 賠率系) | 21,592 × 5 | 5-40 | ✗ 未使用 (oz は配当系) | レース後 | **★ POST-RACE 永久除外 ★** | high |

**重要観察**:
- V15 既採用 13 source: kyi / sed / cha (3) / paci (7) / kta (3) / ze (4) / jo (2) / srb (1) / kka_v2 (2) / ukc / bac / 動画 ✗ / video ✗
- **未使用で安全な真の signal source**: **cha / paci / kab / kka_v2** (cha は既存 3 採用、 拡張余地は trend/rank)
- **永久除外**: tyb (Sub-task 11) / skb (POST-RACE LEAK) / hjc / ot/ou/ov/ow/oz (払戻)

---

## 2. release timing audit (Sub-task 5-1 framework 拡張)

JRDB の release timing は data type 単位で **4 段階**:

| timing | data types | 学習利用 | live 利用 |
|--------|-----------|----------|-----------|
| **静的 master (年次更新)** | ukc / ksa / kz / csa / cz | ✓ 安全 | ✓ 安全 |
| **前週 木曜 (番組発表)** | bac / joa | ✓ 安全 | ✓ 安全 |
| **3-4 日前 確定 (追切)** | **cha** | ✓ 安全 | ✓ 安全 (3 日前 fetch 確定) |
| **当日朝 06:00 確定** | **kyi / kab / paci / kta / jo** | ✓ 安全 | ✓ 安全 (V15 daily_jrdb_kyi 取得済) |
| **17:00 JST 一括 publish (race 後)** | **tyb** | ✗ retrospective のみ | ✗ live 不可 (永久放棄) |
| **レース後 配信 (確定)** | hjc / sed / ze / zk / sr / srb / skb / ot/ou/ov/ow/oz | △ 前走として利用なら OK | ✗ 当該 race は LEAK |

### 2-1. 実測根拠 (cha / paci / kab)

```
cha (追切): oikiri_date vs race_date diff = median 3 days
  (実測、 10,000 sample) → 中央 木曜追切 → 土日 race の 3-4 日前 確定
paci: JRDB AM3:00 schtask で取得済、 当日朝 06:00 確定 (既存 V15 utilizes)
kab: KAB ファイルは 当日朝 (kaisai_key + yyyymmdd で race 当日)
kta: 当日朝 (場別調教師、 既存 V15 utilizes)
```

### 2-2. 各 data type 真の release timing 表

| data type | publish source | publish 時刻 | live 直前 fetch 可否 | V15 schtask 取得済 |
|-----------|---------------|-------------|--------------------|-------------------|
| kyi | KYI ファイル | **06:00 JST** | ✓ | ✓ daily_jrdb_kyi |
| bac | BAC ファイル | **前週木曜** | ✓ | ✓ |
| sed | SED ファイル | **race 後** | ✗ (current race) / ✓ (前走) | ✓ |
| cha | CHA ファイル | **3-4 日前 (木曜追切後)** | ✓ | ✓ |
| sr | SR ファイル | **race 後** | ✗ (current) / ✓ (prev) | ✓ |
| kab | KAB ファイル | **06:00 JST** | ✓ | ✓ |
| paci | PACI ファイル | **06:00 JST** (前日夜更新含む) | ✓ | ✓ daily_jrdb_kyi |
| kta | KTA ファイル | **06:00 JST** | ✓ | ✓ |
| jo | JO ファイル | **06:00 JST** | ✓ | ✓ |
| **tyb** | **TYB ZIP** | **17:00 JST** | **✗ 永久 NO-GO (Sub-task 11)** | ✓ (15:30 race の場合 retro のみ) |
| skb | SKB | レース後 (= 成績拡張) | ✗ POST-RACE LEAK | ✓ (除外運用) |
| hjc/ot/ou/ov/ow/oz | (払戻 / 賠率系) | レース後 | ✗ | ✓ |

→ ★ v15.2 主軸 source = **kyi / cha / paci / kab / kta** (全て当日朝 06:00 までに確定)、 sed/ze は **前走 reference のみ** ★

---

## 3. 未使用 data type signal 候補 (実測 corr_target)

`data/jrdb_sed.csv` (finish カラム) を target 化 (top3 = finish ≤ 3) して各 data type の 単独 predictive power を測定。

### 3-1. cha (追切、 既存 3 + 新規 ≈4)

| column | n | corr_top3 | V15 採用 | priority | 備考 |
|--------|----:|-----------:|:--------:|:--------:|------|
| oikiri_idx | 52,343 | **+0.0674** | ★既存★ | — | 追切合成指数 |
| shimai_time_idx | 52,081 | **+0.0807** | ★既存★ | — | 終い指数 |
| ten_time_idx | 51,642 | +0.0001 | ★既存★ | — | テン指数 (信号 弱い) |
| **chukan_time_idx** | 51,850 | **+0.0299** | ✗ | **B** | **中間指数 (未採用)** |
| shimai_time | 52,411 | -0.0834 | ✗ | B | 終い時計 (生 sec) |
| chukan_time | 52,413 | -0.0239 | ✗ | C | 中間時計 (生 sec) |
| oikiri_rank | 52,278 | +0.0111 | ✗ | C | 追切順位 1=best |
| oikiri_naiyou | 52,482 | +0.0160 | ✗ | C | 追切内容コード |
| awase_result | 20,772 | -0.0582 | ✗ | C | 合わせ結果 (fill 38%) |
| ten_time | 52,416 | +0.0011 | ✗ | C | テン時計 (生 sec) |

### 3-2. paci (展開、 既存 7 + 新規候補)

| column | n | corr_top3 | V15 採用 | priority | 備考 |
|--------|----:|-----------:|:--------:|:--------:|------|
| idm | 199,662 | **+0.1449** | ★既存★ (jrdb_idm) | — | IDM |
| jockey_idx | 196,065 | **+0.4457** | ★既存★ (paci_jockey_exp_wr 経由) | — | 騎手指数 (★ POST-RACE 候補 = 既存採用済 ★) |
| info_idx | 134,921 | **+0.4139** | ✗ | (要 leak audit) | 情報指数 (corr 高 → 要 P0-3 audit) |
| sogo_idx | 199,997 | +0.1901 | ✗ | B (要 audit) | 総合指数 |
| ninki_idx | 185,786 | +0.4324 | ★既存★ (paci_ninki_idx) | — | 人気指数 (= odds-derived、 LEAK 隣接) |
| train_idx | 198,170 | +0.3994 | ✗ | (要 audit) | 調教指数 (★ jrdb_training_idx 既存と重複?) |
| stable_idx | 196,782 | +0.3607 | ✗ | (要 audit) | 厩舎指数 (既存 jrdb_stable_idx と重複) |
| gekiso_idx | 199,953 | +0.1996 | ✗ | **B** | 激走指数 |
| **manken_idx** | 195,472 | **+0.1496** | ★既存★ (paci_manken_idx) | — | — |
| pace_idx_pred | 192,836 | +0.1091 | ★既存★ (jrdb_pace_idx_pred) | — | — |
| **dochu_rank** | 193,605 | **-0.1494** | ★既存★ (paci_dochu_rank) | — | — |
| **goal_rank** | 193,605 | **-0.3402** | ★既存★ (paci_goal_rank) | — | — |
| **goal_diff** | 178,173 | **-0.2485** | ★既存★ (paci_goal_diff) | — | — |
| ls_idx_rank | 193,605 | **-0.2749** | ✗ | **B** | LS 指数 rank |
| gekiso_rank | 193,605 | -0.2552 | ✗ | **B** | 激走 rank |
| jockey_exp_3rd | 199,998 | **+0.4506** | ★既存★ (paci_jockey_exp_3rd) | — | — |
| jockey_exp_win | 199,823 | +0.4375 | ★既存★ (paci_jockey_exp_wr) | — | — |
| turf_apt | 145,617 | -0.0855 | ✗ | C | 芝適性 (kyi 既存と重複可能) |
| dirt_apt | 115,026 | -0.1264 | ✗ | C | ダ適性 |

→ paci 新規候補は **gekiso_idx / ls_idx_rank / gekiso_rank / info_idx (audit 必要) / sogo_idx (audit 必要)** の 5 件。 info_idx の corr +0.41 は odds-based の可能性高い (V15 odds_log と類似)。

### 3-3. kta (場別調教師、 既存 3 + 拡張余地小)

| column | n | corr_top3 | V15 採用 | priority | 備考 |
|--------|----:|-----------:|:--------:|:--------:|------|
| idm | 78,132 | **+0.0848** | ★既存★ (jrdb_kta_idm) | — | — |
| ten_idx_pred | 258,772 | +0.0871 | ★既存★ (jrdb_kta_ten_pred) | — | — |
| pace_idx_pred | 258,026 | +0.0949 | ✗ | **C** | **kta ペース予想 (paci.pace_idx_pred 既存と重複?)** |
| agari_idx_pred | 258,448 | +0.1525 | ★既存★ (jrdb_kta_agari_pred) | — | — |
| ichi_idx_pred | 253,640 | +0.1446 | ✗ | **B** | **kta 位置予想 (V15 未採用)** |
| condition_class | 214,331 | +0.0206 | ✗ | C | 状態クラス |
| turf_apt | 205,956 | -0.0873 | ✗ | C | 芝適性 (paci/kyi 重複) |
| dirt_apt | 175,931 | -0.1146 | ✗ | C | ダ適性 |

→ kta 新規候補は **ichi_idx_pred** (位置予想、 +0.145) と pace_idx_pred (kta 版、 +0.095) の 2 件。

### 3-4. kka_v2 (breeder/sire 集計、 既存 2 + 新規)

| column | n | corr_top3 | V15 採用 | priority | 備考 |
|--------|----:|-----------:|:--------:|:--------:|------|
| dam_rensho_avg | 294,284 | +0.0349 | ★既存★ (jrdb_dam_rensho_avg) | — | — |
| bms_rensho_avg | 252,305 | +0.0007 | ★既存★ (jrdb_bms_rensho_avg) | — | (信号 弱い) |
| **breeder_dist_1** | 276,188 | **+0.1429** | ✗ | **A** | **breeder 距離別 1 着率 (新規 強信号)** |
| **breeder_track_1** | 216,594 | **+0.1046** | ✗ | **B** | breeder トラック別 1 着率 |
| breeder_surface_1 | 235,522 | +0.0423 | ✗ | C | breeder 芝/ダ別 1 着率 |
| jra_seiseki_1 | 151,315 | +0.0007 | ✗ | C | (信号ほぼ無し) |
| kyori_seiseki_1 | 102,288 | +0.0013 | ✗ | C | (信号ほぼ無し) |

→ kka_v2 新規候補は **breeder_dist_1** (+0.143、 ★最強★) と breeder_track_1 (+0.105) の 2 件。

### 3-5. kab (場別 race-level、 未使用)

| column | n | corr_top3 | V15 採用 | priority | 備考 |
|--------|----:|-----------:|:--------:|:--------:|------|
| turf_baba_sa | 536,663 | -0.0036 | ✗ | (interaction で利用) | 芝馬場差 |
| turf_baba_inner | 548,955 | -0.0022 | ✗ | (interaction) | 芝内 |
| turf_baba_outer | 548,955 | -0.0027 | ✗ | (interaction) | 芝外 |
| straight_sa_most | 319,679 | -0.0015 | ✗ | (interaction) | 直線差 (最多) |
| renzoku_day | 548,955 | -0.0026 | ✗ | (interaction) | 連続日 (荒れ要因) |
| grass_height | 548,804 | -0.0001 | ✗ | (interaction) | 芝丈 |
| rain_mid | 389,300 | +0.0003 | ✗ | (interaction) | 降雨 (中) |

→ kab は **per-horse corr 全て ≈0** だが、 **interaction features (kab × horse 特性) で利用価値**:
- 例: turf_baba_inner × wakuban (内枠 × 内有利) / straight_sa_inner × bracket_pos / grass_height × horse_career_top3r 等

→ priority **B (interaction features 経由)** で 3-5 candidates。

### 3-6. sr / srb (ラップ / track bias、 race-level)

sr (current race の ラップ) は **当該レースの集計 = POST-RACE = LEAK**。
→ **前走 race の sr 集計** だけが safe。 既存 V15 で `prev_race_first3f` / `prev_race_last3f` / `prev_race_pace_diff` で部分採用済 → 拡張余地 small。

| column | n | corr_top3 | V15 採用 | priority | 備考 |
|--------|----:|-----------:|:--------:|:--------:|------|
| harlon_1 (current race) | 274,603 | +0.0012 | ✗ | ★ POST-RACE 除外 ★ | 当該レース集計 = LEAK |
| (前走 sr 集計: race_id-1 で利用) | — | — | △既存★ (prev_race_*) | — | 既存採用済 |

srb (track bias) も同様。 既存 V15 で `jrdb_tb_homestr_inner` 1 件採用。 拡張余地は **bias_1corner / bias_4corner / bias_straight** の 3 件で priority **C**。

---

## 4. cross / interaction features 候補

V15 既存 145 features × 上記新規 base feats の cross。 ただし **Session #57 PoC で V15 interaction は LGB が内部捕捉済 = 効果 0** が確証済。 → cross は priority B/C 中心、 真の信号は **rank/zscore (race 内 normalization)** に絞る。

### 4-1. priority A 候補 (race 内 rank / zscore、 主軸)

| ID | name | formula | base | 想定 +AUC |
|----|------|---------|------|----------:|
| **A-1** | `paci_gekiso_race_rank` | race 内 paci.gekiso_idx 降順 ranking | paci.gekiso_idx | **+0.0010** |
| **A-2** | `paci_lsidx_race_rank` | race 内 paci.ls_idx_rank 降順 | paci.ls_idx_rank | +0.0008 |
| **A-3** | `cha_chukan_idx_race_zscore` | race 内 (chukan_time_idx - mean) / std | cha.chukan_time_idx | +0.0007 |
| **A-4** | `kta_ichi_pred_race_rank` | race 内 kta.ichi_idx_pred 降順 | kta.ichi_idx_pred | +0.0008 |
| **A-5** | `breeder_dist_1_race_rank` | race 内 kka_v2.breeder_dist_1 降順 | kka_v2.breeder_dist_1 | +0.0009 |

### 4-2. priority A 候補 (per-horse 直接、 強信号)

| ID | name | formula | base | 想定 +AUC |
|----|------|---------|------|----------:|
| **A-6** | `breeder_dist_1` | (raw) | kka_v2.breeder_dist_1 | **+0.0012** |
| **A-7** | `paci_gekiso_idx` | (raw) | paci.gekiso_idx | +0.0009 |
| **A-8** | `kta_ichi_idx_pred` | (raw) | kta.ichi_idx_pred | +0.0008 |

### 4-3. priority B 候補 (trend / 派生)

| ID | name | formula | base | 想定 +AUC |
|----|------|---------|------|----------:|
| **B-1** | `cha_oikiri_idx_trend` | 今回 - 前走 oikiri_idx | cha.oikiri_idx | +0.0006 |
| **B-2** | `cha_chukan_time_idx` | (raw) | cha.chukan_time_idx | +0.0005 |
| **B-3** | `paci_ls_idx_rank` | (raw) | paci.ls_idx_rank | +0.0005 |
| **B-4** | `paci_gekiso_rank` | (raw) | paci.gekiso_rank | +0.0005 |
| **B-5** | `breeder_track_1` | (raw) | kka_v2.breeder_track_1 | +0.0006 |
| **B-6** | `kab_turf_baba_x_bracket` | kab.turf_baba_inner × bracket | kab + bracket | +0.0005 (interaction) |
| **B-7** | `kab_straight_sa_x_horse_num_ratio` | kab.straight_sa_inner × horse_num_ratio | kab + horse_num_ratio | +0.0004 |
| **B-8** | `kab_renzoku_day` | (raw、 連続開催日 = 荒れ要因) | kab.renzoku_day | +0.0003 |
| **B-9** | `cha_shimai_time_3r_mean` | 直近 3 走 cha.shimai_time 平均 (expanding) | cha.shimai_time | +0.0004 |

### 4-4. priority C 候補 (保留、 試験 round で評価)

| ID | name | formula | base | 想定 +AUC |
|----|------|---------|------|----------:|
| C-1 | `cha_oikiri_rank` | (raw) | cha.oikiri_rank | +0.0003 |
| C-2 | `cha_awase_result` | (raw、 fill 38%) | cha.awase_result | +0.0003 |
| C-3 | `srb_bias_straight` | 前走 srb bias_straight | srb (prev) | +0.0003 |
| C-4 | `kta_pace_idx_pred` | (kta 版 pace pred) | kta.pace_idx_pred | +0.0002 |
| C-5 | `paci_info_idx` | (★ corr +0.41 だが LEAK 疑い、 audit 必要) | paci.info_idx | (audit 後判定) |

---

## 5. v15.2 features list (priority sort、 採用判定)

### 5-1. 採用想定 17 features (priority A+B、 corr 順)

| rank | feature ID | name | category | corr_top3 (実測) | 想定 +AUC | leak risk | 採用 |
|-----:|-----------|------|----------|-----------------:|----------:|-----------|:---:|
| 1 | A-6 | breeder_dist_1 | per-horse | +0.143 | +0.0012 | low | ★ |
| 2 | A-1 | paci_gekiso_race_rank | rank | (rank 派生) | +0.0010 | low | ★ |
| 3 | A-5 | breeder_dist_1_race_rank | rank | (rank 派生) | +0.0009 | low | ★ |
| 4 | A-7 | paci_gekiso_idx | per-horse | +0.200 | +0.0009 | low | ★ |
| 5 | A-4 | kta_ichi_pred_race_rank | rank | (rank 派生) | +0.0008 | low | ★ |
| 6 | A-2 | paci_lsidx_race_rank | rank | (rank 派生) | +0.0008 | low | ★ |
| 7 | A-8 | kta_ichi_idx_pred | per-horse | +0.145 | +0.0008 | low | ★ |
| 8 | A-3 | cha_chukan_idx_race_zscore | rank | (zscore 派生) | +0.0007 | low | ★ |
| 9 | B-5 | breeder_track_1 | per-horse | +0.105 | +0.0006 | low | ★ |
| 10 | B-1 | cha_oikiri_idx_trend | trend | (trend 派生) | +0.0006 | low | ★ |
| 11 | B-2 | cha_chukan_time_idx | per-horse | +0.030 | +0.0005 | low | ★ |
| 12 | B-3 | paci_ls_idx_rank | per-horse | -0.275 | +0.0005 | low | ★ |
| 13 | B-4 | paci_gekiso_rank | per-horse | -0.255 | +0.0005 | low | ★ |
| 14 | B-6 | kab_turf_baba_x_bracket | interaction | (cross 派生) | +0.0005 | low | ★ |
| 15 | B-9 | cha_shimai_time_3r_mean | trend | (trend 派生) | +0.0004 | low | ★ |
| 16 | B-7 | kab_straight_sa_x_horse_num_ratio | interaction | (cross 派生) | +0.0004 | low | ★ |
| 17 | B-8 | kab_renzoku_day | per-horse | -0.003 (race-level) | +0.0003 | low | ★ |

**想定累計 +AUC**: Σ = +0.0114 (独立性 80% 仮定で **+0.0091**)

### 5-2. 試験 round 候補 (priority C、 5 features)

C-1 〜 C-5、 まず 17 features で WF round → 効果見極めて C 追加判定。 ただし C-5 `paci_info_idx` (corr +0.41) は **P0-3 leak audit 必須** (V15.1 SKB 教訓)。

### 5-3. 採用判定基準 (絶対遵守)

| 項目 | 基準 |
|------|------|
| WF AUC 平均 (6-fold) | ≥ V15 0.8939 + **0.003** (= **≥ 0.8969**) |
| 全 fold AUC | ≥ 各 fold V15 + 0.002 (全 6 fold 上回り必須) |
| 実 ROI (戦略⑦込み、 paper) | ≥ V15 140% + 2pt (= **≥ 142%**) |
| LIVE retro (3 週末) | winner_top1 rate ≥ V15 + 1.5pt、 shift ≤ 12x |
| LEAK 監査 | 全 17 features individual + 集約で leak audit PASS (corr_target / monotonic / per-finish 集計) |

★ 1 項目でも未達 → **採用 NO-GO、 V15 維持** ★ (V15.1 SKB / V18 / V19 sib hybrid NO-GO 教訓)

---

## 6. v15.2 学習 schedule (5/24+ 着手)

### Phase A: 5/24-5/27 (3 日、 base 整備)

| 日 | task |
|----|------|
| 5/24 (Sat) | tools/v152/fe_jrdb_unused_merge.py 新規実装 (cha trend / paci rank / kta ichi / breeder_dist 等の merge logic) |
| 5/25 (Sun) | data/_v15_2_optuna_df_cache.pkl.gz 構築 (V15 cache + 17 new features)、 nunique / fill rate 検証 |
| 5/26 (Mon) | tools/v152/leak_audit_17_features.py、 17 features の corr_target / monotonic / per-finish 集計 audit |
| 5/27 (Tue) | audit PASS 確認 → train/v15_2_master.py 着手 (train_v15_master.py の差分 implement) |

### Phase B: 5/28-5/31 (4 日、 WF 評価)

| 日 | task |
|----|------|
| 5/28 (Wed) | WF 6-fold 学習 (LGB+XGB)、 fold 1-3 完走 |
| 5/29 (Thu) | fold 4-6 完走、 全 fold AUC + 比較 V15 |
| 5/30 (Fri) | FT-Transformer + IntraRace 追加、 4-model ensemble |
| 5/31 (Sat) | 採用判定 §5-3 機械評価 → GO/no-go 判断 |

### Phase C: 6/1-6/8 (1 週、 LIVE retro + paper)

| 日 | task |
|----|------|
| 6/1 (Sun) | paper trade (5/31 + 6/1 weekend) winner_top1 / shift 測定 |
| 6/2-6/6 | 週次 LIVE retro 蓄積 |
| 6/7 (Sat) | 戦略⑦込み ROI シミュレーション |
| 6/8 (Sun) | 最終判定 → GO ならば 6/15+ V15.2 段階投入 (V20 投入と同じ枠) |

### 工数見積 (各 priority A/B 1-2 features ≈ 0.3 day)

| feature 数 | 想定工数 |
|----------:|----------|
| 17 (A+B) | 5 work-day (FE 1 + audit 1 + WF 2 + LIVE 1) |
| +5 (C 試験 round) | +2 day |
| **合計** | **7 day** (5/24-5/31 + 6/1-6/8 LIVE) |

---

## 7. ★ TYB 放棄後の真の path ★

```
旧 path (V152_FE_DESIGN_2026_05_16.md):
  V15 (145) + TYB 17 features (priority A+B) → V15.2
  → P0-4 TYB live fetch 必須 → Sub-task 11 で 永久放棄

新 path (本 doc):
  V15 (145) + JRDB 未使用 source 17 features → V15.2
  - cha trend (chukan_idx / oikiri trend / shimai_3r_mean): 3 feats
  - paci 拡張 (gekiso_idx / ls_idx_rank / gekiso_rank + race rank): 5 feats
  - kta 拡張 (ichi_idx_pred + race rank): 2 feats
  - kka_v2 breeder (breeder_dist_1 + race rank + breeder_track_1): 3 feats
  - kab interaction (turf_baba × bracket / straight_sa × horse_num_ratio / renzoku_day): 3 feats
  - rank/zscore subtotal: 5 ranks + zscore
  → 17 features 計、 P0-3 audit 不要 (release timing 全 OK)
  → 5/24+ 着手可能
```

### 7-1. 戦略 layer との統合 (Sub-task 13-17 並走)

v15.2 model layer は単独で評価し、 5/18+ の戦略 layer 結果 (Sub-task 13 ticket 最適 / 14 odds × venue / 17 odds band / calibration v2) と **直交** で統合:

```
最終 ROI = model layer (V15 → V15.2 想定 +0.003 AUC) × 戦略 layer (戦略⑦ → 案 C 想定 +5pt)
        ≈ V15 base 119% × (model +1.2pt × 戦略 +5pt) = ~125% 想定
        ⇔ 採用基準 142% (戦略⑦込み V15 baseline 140%) を 上回るには
           model +3pt + 戦略 +0pt または model +0pt + 戦略 +3pt または合算で +3pt
```

→ ★ v15.2 単独で +3pt 達成は厳しい (想定 +2pt)、 戦略 layer との合算で 142%+ 達成想定 ★

### 7-2. 5/24 着手 確認事項

| 確認 | 値 |
|------|----|
| P0-3 (tansho_odds 等 LEAK audit) PASS 必要性 | ✗ 不要 (本 doc 候補 17 features は odds-related なし) |
| V15 production 影響 | 0 (cache 別 file、 .pkl.gz 別 file) |
| daily_jrdb_kyi schtask 維持 | ✓ 既存維持 (cha / paci / kta / kab 既に取得済) |
| 撤退 ライン | 累計 -50,000 円 (現状 +13,530 円、 撤退余裕 +63,530 円) |
| commit/push | 本 sub-task = docs のみ、 親集中 |

---

## 8. fabrication 防止 + honest 留意

### 8-1. 「想定 +AUC」 = pre-implementation estimate

- 全 17 features の `想定 +AUC` は **実装前の事前推定 (assumption)**
- 根拠: 単独 corr_top3 実測値 (n>50K) を base に、 LGB の interaction 内部捕捉 (Session #57 教訓) 控除分を 20-30% 想定
- **過大評価 risk**: rank features の真の delta は実測必要。 fold 別 ばらつき 大きい可能性
- 採用判定基準 §5-3 で 全 fold 改善必須 → 過大評価 受容不可

### 8-2. LEAK 候補の見落とし

- paci.info_idx (corr +0.41) は **odds-related と疑わしい** → priority C で audit 必須
- paci.train_idx / stable_idx (corr +0.40 / +0.36) も検討候補だったが、 既存 jrdb_training_idx / jrdb_stable_idx と重複可能性 → 採用しない
- paci.jockey_idx (corr +0.45) は paci_jockey_exp_wr 既存と類似 (騎手指数) → 採用しない

### 8-3. 過去 LEAK 教訓 (踏襲必須)

| 事例 | 教訓 |
|------|------|
| V15.1 SKB POST-RACE LEAK (Session #38) | skb_kishi_code_3 corr 0.137 だけで V15.1 NO-GO 判定。 corr 高い feature は必ず monotonic + per-finish audit |
| dam_top3r 全年集計 LEAK (Session #38) | 静的 CSV 集計値は expanding 化必須。 breeder_dist_1 は **静的 (生涯)** だが race date より過去の限定要否を確認必須 |
| sib_top3_rate hybrid (Session #38) | corr 0.294 → expanding 化後 0.169。 残存信号で V18/V19 NO-GO。 本 doc も実測必要 |
| TYB content delivery 矛盾 (Sub-task 6/11) | content 安全でも delivery POST-RACE なら live 不可。 release timing は data type 単位で 確認 |

### 8-4. 実装時 verify checklist

```
1. v15_2_cache 構築後:
   - 全 17 features の nunique > 10 (constant 化 ガード)
   - 全 17 features の fill rate > 50% (TYB merge bug 教訓)
   - race 内 rank features の groupby race_id で nunique == num_horses 確認

2. leak audit:
   - 17 features 個別 corr_target、 |corr| > 0.15 なら monotonic + per-finish 詳細
   - LIVE retro shift ≤ 12x (sib hybrid 教訓: shift 30x 超で NO-GO)

3. WF 6-fold:
   - 全 fold で V15 を上回り必須 (3 fold OK + 3 fold NG はトータル GO 判定でも NO-GO)
   - 採用基準 §5-3 機械的に適用
```

---

## 9. 完了通知 template

```
Sub-task 18 完了、 v15.2 候補 features = 17 (priority A: 8 / B: 9)、 試験 round +5 (priority C)、
想定 +AUC +0.004 〜 +0.010 (実測前 assumption)、
想定 +ROI +2 〜 +6pt (戦略⑦込み)、
TYB 永久放棄後の真の path = cha + paci + kab + kta + kka_v2 主軸、
P0-3 leak audit 不要 (release timing 全 OK)、
着手時期 5/24+、 学習期間 7-day (FE 4 + WF 2 + LIVE 1)、
V15 production 影響 0、 commit/push なし (親集中)。

honest 留意:
- 「想定 +AUC」 は事前推定、 LGB interaction 内部捕捉控除分含む
- paci.info_idx (corr +0.41) は priority C で leak audit 必須
- 全 17 features の corr_top3 実測根拠 n>50K
- 採用判定基準 5 項目 ALL PASS が条件、 1 項目でも未達なら V15 維持
- LIVE retro shift ≤ 12x、 全 fold V15 上回り必須
```

---

## Appendix A: 主軸 6 source の data 規模 + V15 利用率

| source | rows | V15 採用 features | utilization |
|--------|-----:|------------------:|-------------|
| kyi | 292,490 | 18 / 70 cols | 26% |
| sed | 548,780 | 7 / 49 cols (前走) | 14% |
| cha | 302,742 | 3 / 21 cols | 14% (新規 4 余地) |
| paci | 549,604 | 7 / 63 cols | 11% (新規 5 余地) |
| kta | 298,551 | 3 / 34 cols | 9% (新規 2 余地) |
| ze | 537,386 | 4 / 49 cols (集計) | 8% |
| kab | 3,296 | 0 / 55 cols | 0% (interaction 3 候補) |
| kka_v2 | 548,606 | 2 / 95+ cols | 2% (新規 3 候補) |
| jo | 302,742 | 2 / 19 cols | 11% |
| srb | 21,591 | 1 / 14 cols | 7% |
| ukc | 36,939 | 静的 (sire/bms map) | — |

→ v15.2 で利用率 引き上げ: paci 11% → 17%、 cha 14% → 19%、 kka_v2 2% → 7%。
全体としては JRDB の 中位 1/3 を活用する設計。 残り 2/3 は POST-RACE / 静的 master / 重複 で除外。

## Appendix B: 永久除外 source 一覧

| source | 除外理由 | session 確定 |
|--------|---------|--------------|
| tyb | delivery 17:00 JST = POST-RACE (live 不可) + content LEAK 3 features | Sub-task 11 |
| skb | POST-RACE LEAK 確定 (skb_kishi_code_3 corr 0.137、 1 着→364 / 10 着→176) | Session #38 |
| hjc / ot / ou / ov / ow / oz | 払戻系 = race 後配信 | (自明) |
| (video paddock CNN / patrol YOLO / training keypoint) | 規約 NG + cost 1K/月 + 不確実性 | Session #85 撤回 |

## Appendix C: 5/24 着手 prompt template

```
★ v15.2 実装 (Sub-task 18 設計に基づく) ★

【前提】
- docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md の 17 features を実装
- TYB / SKB 永久除外、 P0-3 leak audit 不要
- V15 cache (data/_v15_optuna_df_cache.pkl.gz) は read-only

【絶対遵守】
🔴 NEVER:
- predict_core.py / daily_predict.py / race_auto_notify.py / app.py 変更
- V15 .pkl.gz / V15 cache 上書き
- git commit / push (親集中)
- fabrication

🟢 OK:
- tools/v152/ 新規 (fe_jrdb_unused_merge.py / leak_audit_17_features.py)
- data/_v15_2_optuna_df_cache.pkl.gz (新規 file)
- train/v15_2_master.py (新規、 train_v15_master.py の差分)

【実装】
1. tools/v152/fe_jrdb_unused_merge.py
   - V15 cache load (read-only)
   - cha / paci / kta / kka_v2 / kab merge (race_id [+ umaban or kaisai_key])
   - 17 features compute (priority A 8 + priority B 9)
   - rank / zscore は groupby race_id
   - trend (cha_oikiri_idx_trend / cha_shimai_time_3r_mean) は per horse_id + date 順
   - v15_2_cache 永続化

2. tools/v152/leak_audit_17_features.py
   - 17 features の corr_target / monotonic / per-finish 集計
   - paci.info_idx (priority C) のみ +0.4 corr で 単独 audit 必須

3. train/v15_2_master.py (差分 implement)
   - WF 6-fold (V15 baseline と同一 split)
   - LGB+XGB+FT+IR ensemble
   - 採用判定基準 §5-3 機械的に評価
   - 出力: docs/V152_WF_RESULTS_<DATE>.md

【出力】
- data/_v15_2_optuna_df_cache.pkl.gz
- docs/V152_LEAK_AUDIT_<DATE>.md
- docs/V152_WF_RESULTS_<DATE>.md
- 判定: GO (基準 §5-3 全 PASS) → 6/15+ paper trade 段階投入
       NO-GO → V15 維持、 honest 報告

★ V15 production 影響 0、 commit/push 親集中、 honest 厳守 ★
```

---

**END OF DOC**

