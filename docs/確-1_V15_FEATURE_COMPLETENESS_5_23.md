# 確-1: V15 特徴量取得漏れ Audit 2026-05-23

作成: 2026-05-23 (Session #91)

## 0. 目的

5/23 当日 DailyPredict (08:00-08:39) において、V15 production モデル (145 booster features) が
何個の特徴量を実データで埋め、何個をデフォルト値で補完したかを確認する。

---

## 1. V15 モデル 特徴量構成 (真値)

### 1.1 pkl 構造
| 項目 | 値 |
|------|----|
| ファイル | `keiba_model_v15_central_live.pkl.gz` (Pattern B) |
| pkl['features'] | **150** features |
| LGB booster 入力 | **145** features (Column_0 〜 Column_144) |
| XGB booster 入力 | 145 features |
| 5件 truncate | features[145-149] = TYB 直前指数 (Pattern B only、booster 未入力) |

### 1.2 booster に入らない 5 features (Pattern B 専用)
```
146. jrdb_paddock_idx      <- TYB 直前パドック指数
147. jrdb_odds_idx         <- TYB 直前オッズ指数
148. jrdb_live_composite_idx <- TYB 直前総合指数
149. jrdb_body_code        <- TYB 馬体コード
150. jrdb_demeanor_code    <- TYB 気配コード
```
★ これら 5 件は booster 未入力のため、5/23 欠損かどうかは **予測精度に影響なし**。
(TYB ファイル自体は 5/23 分なし: `jrdb_tyb_v2.csv` max = 202609020712 ≠ 5/23 race IDs)

---

## 2. V15 145 Feature の データソース分類

### 2.1 データソース別 feature 数

| データソース | feature 数 | 5/23 データ有無 |
|------------|-----------|----------------|
| netkeiba (出馬表/成績) | 30 | OK |
| feature_lookups.pkl | 29 | OK (事前計算キャッシュ) |
| derived (計算式) | 14 | OK |
| netkeiba_premium (調教タイム) | 10 | OK (部分) |
| JRDB_KYI | 19 | **OK** (549行/36レース確認) |
| JRDB_SED (blood_num基準) | 6 | **OK** (89.4%取得) |
| JRDB_CHA | 3 | **OK** (549行/36レース確認) |
| JRDB_JO | 2 | **OK** (549行/36レース確認) |
| JRDB_ZE (blood_num基準) | 4 | OK (blood_num経由) |
| JRDB_speed_index | 4 | OK (netkeiba_speed_index.csv) |
| cumulative_history | 3 | OK (馬体重履歴) |
| derived_geo / derived_static | 4 | OK |
| **JRDB_KTA** | **3** | **MISSING → default** |
| **JRDB_SR** | **1** | **MISSING → default** |
| **JRDB_KKA** | **2** | **MISSING → default** |
| **JRDB_OZ** | **3** | **MISSING → default** |
| **JRDB_OZ (odds change)** | **3** | **MISSING → default** |
| **JRDB_PACI** | **12** | **PACI fallback (stale)** |

### 2.2 Summary
| 区分 | 件数 | 割合 |
|------|------|------|
| AVAILABLE (実データ) | 121 | **83.4%** |
| DEFAULT VALUE (欠損) | **24** | **16.6%** |

---

## 3. JRDB ファイル 5/23 データ有無

5/23 レース race_id:
- 京都: `202608030901` 〜 `202608030912`
- 新潟: `202604010701` 〜 `202604010712`
- 東京: `202605020901` 〜 `202605020912`

| ファイル | 5/23 rows | 状態 |
|--------|-----------|------|
| `jrdb_kyi.csv` | 549 (36レース×~15.3頭) | OK |
| `jrdb_cha.csv` | 549 | OK |
| `jrdb_jo.csv` | 549 | OK |
| `jrdb_sed.csv` | N/A (blood_num基準) | OK (89.4%取得) |
| `jrdb_sed_v2.csv` | N/A (blood_num基準) | OK |
| `jrdb_ze.csv` | N/A (blood_num基準) | OK |
| `jrdb_kta.csv` | **0** | **MISSING** |
| `jrdb_sr.csv` | **0** | **MISSING** |
| `jrdb_srb.csv` | 0 (booster未使用) | — |
| `jrdb_kka.csv` | **0** | **MISSING** |
| `jrdb_oz.csv` | **0** | **MISSING** |
| `jrdb_paci.csv` | **0** (馬名 fallback) | **STALE** |
| `jrdb_tyb_v2.csv` | 0 (booster外) | — |

---

## 4. DEFAULT 値で埋まった 24 features 詳細

### 4.1 JRDB_KTA (3 features) — デフォルト固定
| feature | default値 | 意味 |
|---------|-----------|------|
| `jrdb_kta_idm` | 13.0 | KTA IDM予想 |
| `jrdb_kta_ten_pred` | -14.0 | KTA テン指数予想 |
| `jrdb_kta_agari_pred` | -11.0 | KTA 上がり指数予想 |

### 4.2 JRDB_SR (1 feature) — デフォルト固定
| feature | default値 | 意味 |
|---------|-----------|------|
| `jrdb_tb_homestr_inner` | 2.0 | 直線内外バイアス |

### 4.3 JRDB_KKA (2 features) — デフォルト固定
| feature | default値 | 意味 |
|---------|-----------|------|
| `jrdb_dam_rensho_avg` | 1600.0 | 母馬平均連勝指数 |
| `jrdb_bms_rensho_avg` | 1600.0 | BMS平均連勝指数 |

### 4.4 JRDB_OZ (3 features) — デフォルト固定
| feature | default値 | 意味 |
|---------|-----------|------|
| `oz_tansho_base_log` | 2.3 | 基準単勝オッズ log |
| `oz_fukusho_base_log` | 0.7 | 基準複勝オッズ log |
| `oz_base_pop_rank` | 8 | 基準人気順位 |

### 4.5 JRDB_OZ 派生 (3 features) — デフォルト固定
| feature | default値 | 意味 |
|---------|-----------|------|
| `odds_change_rate` | 0.0 | オッズ変動率 |
| `pop_rank_change` | 0 | 人気順変動 |
| `odds_sharp_drop` | 0 | 急落フラグ |

### 4.6 JRDB_PACI (12 features) — **馬名フォールバック使用 (stale)**
| feature | default値 (非hit時) | 備考 |
|---------|---------------------|------|
| `paci_manken_idx` | 36.0 | 5/23当日データなし → 過去PACI最新値 |
| `paci_goal_rank` | 8.0 | 同上 |
| `paci_dochu_rank` | 8.0 | 同上 |
| `paci_goal_diff` | 12.0 | 同上 |
| `paci_jockey_exp_wr` | 14.5 | 同上 |
| `paci_jockey_exp_3rd` | 21.9 | 同上 |
| `paci_ninki_idx` | 159.0 | 同上 |
| `gaisha_rank` | 0 | 同上 |
| `paci_sogo_mark` | 0 | 同上 |
| `paci_idm_mark` | 0 | 同上 |
| `paci_jockey_mark` | 0 | 同上 |
| `paci_train_mark` | 0 | 同上 |

**PACI 状況詳細:**
- jrdb_paci.csv に 5/23 race_id は 0 件 (5/23 当日分はなし)
- **馬名フォールバック**: 過去 PACI から最新値を取得 → 平均 **7.5頭/レース (50.3% coverage)**
- hit した馬: 過去レース値 (stale、典型的に数週〜数ヶ月前の PACI)
- miss した馬 (49.7%): default 値 (上表)
- pci 特徴量: 馬名フォールバック経由で一部取得 (PACI fallback 内の gaisha_rank も同様)

---

## 5. SED 前走特徴量 取得率

log から集計:
```
SED prev-race: 472/528 horses = 89.4%
(34レース × 平均15.5頭、血統登録番号経由)
```
- 取得できなかった 10.6% の馬: jrdb_prev_* 全て default値
  - `jrdb_prev_idm` = 50.0
  - `jrdb_prev_pace_idx` = 50.0
  - `jrdb_prev_rise_code` = 3
  - 他 3 features も各 default

---

## 6. KAB ファイルについて

`jrdb_kab.csv` は存在するが、predict_core.py の `merge_jrdb_predict_features()` で **参照されていない**。
V15 145 features リストに KAB 由来の特徴量は含まれない。
→ **KAB は V15 予測に影響なし**。(5/23 morning audit で "KAB NG" と記録されていた件は無関係)

---

## 7. 5/23 当日 Log サマリー

```
開始: 2026-05-23 08:00:03
終了: 2026-05-23 08:39:11
対象: 34レース (36検出 - 障害2除外)
成功: 34/34 (失敗0)
モデル: keiba_model_v15_central_live.pkl.gz (Pattern B, 150特徴量)
```

**JRDB 別取得状況:**
- `[JRDB] SED前走特徴量: XX/YY馬取得` → 全34レースで出力、89.4% coverage
- `[JRDB] PACI馬名フォールバックでXX/YY馬取得` → 全34レースで出力、50.3% coverage
- KTA/SR/KKA/OZ に関するエラーメッセージ: **なし** (ファイルに5/23データが存在しないため、コードが直接 0件と判定してデフォルト補完)

---

## 8. 精度への影響評価

### 8.1 欠損 features の重要度分類

| group | features | 重要度推定 |
|-------|----------|-----------|
| PACI (12 features) | paci_* / gaisha_rank | 中 (PACI fallback で 50.3% は stale 値) |
| OZ + change (6 features) | oz_*/odds_* | 中〜高 (基準オッズ・変動がゼロ固定) |
| KTA (3 features) | jrdb_kta_* | 低〜中 |
| KKA (2 features) | jrdb_dam/bms_rensho_avg | 低 |
| SR (1 feature) | jrdb_tb_homestr_inner | 低 (2.0固定) |

### 8.2 OZ 欠損の影響
- `oz_tansho_base_log = 2.3` は ln(1+10) ≈ 単勝オッズ10倍相当の全馬均一値
- `oz_base_pop_rank = 8` は全馬 8番人気相当の全馬均一値
- **全馬同一値のため、ranks 間の相対的差分がゼロ → OZ 系特徴量が無効化**

### 8.3 PACI 欠損の影響
- 50.3% の馬は stale PACI 値 (過去レースから取得)
- 49.7% の馬は default 値 (全馬同一)
- stale 値 vs default 値の混在 → **馬間の比較に一貫性なし**

### 8.4 結論
- 欠損 24 features のうち OZ 系 6 件と PACI 12 件が実質的に **無効化 or stale**
- これは **18 features / 145 = 12.4%** が信頼性低下
- KYI (19) + SED (89.4%) + CHA (3) + JO (2) の主要 JRDB 指数は **正常取得**
- **5/23 予測は一部欠損だが、主要指数は取得できており部分欠損**

---

## 9. 既知課題リスト

| # | 欠損ソース | 欠損理由 | 対策案 |
|---|-----------|---------|-------|
| 1 | JRDB_OZ | 5/23 分が jrdb_oz.csv に未収録 | JV-Link O1 経由での補完 (Phase 3 6/9+) |
| 2 | JRDB_KTA | 5/23 分が jrdb_kta.csv に未収録 | 同上 |
| 3 | JRDB_SR | 5/23 分が jrdb_sr.csv に未収録 | 同上 |
| 4 | JRDB_KKA | 5/23 分が jrdb_kka.csv に未収録 | 同上 |
| 5 | JRDB_PACI | 5/23 当日分なし、fallback=stale | PACI は weekly 更新のみ (許容範囲内) |

---

## 10. Verdict

| 項目 | 値 |
|------|----|
| 145 features 実データ rate | **121/145 = 83.4%** |
| 欠損 features (default値) | **24 features = 16.6%** |
| KAB 影響 | **なし** (V15 features に KAB 不使用) |
| 5/23 予測精度 | **一部欠損 (OZ/PACI/KTA/SR/KKA がデフォルト補完)** |
| 主要 JRDB 指数 (KYI/SED/CHA/JO) | **正常取得** |
| 予測そのものは成功 | **34/34 レース 完了** |

**5/23 予測 = 主要特徴量は取得成功、OZ/PACI/KTA 系 24 features はデフォルト値。真精度ではなく一部欠損運用。**
