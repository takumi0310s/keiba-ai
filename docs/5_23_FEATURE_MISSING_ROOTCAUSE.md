# 2026-05-23 Feature Missing Root Cause Audit

作成: 2026-05-23 (Session #92)
対象: V15 本番予測 / 08:00 DailyPredict / 34レース

## 前提: 5/23 レース race_id 一覧

| 会場 | 会場コード | kai | day | 5/23 race_id 例 |
|------|-----------|-----|-----|----------------|
| 京都 | 08 | 03 | 09 | 202608030901 〜 202608030912 |
| 新潟 | 04 | 01 | 07 | 202604010702 〜 202604010712 |
| 東京 | 05 | 02 | 09 | 202605020901 〜 202605020912 |

> 注意: JRDB 会場コード 08 = 京都, 09 = 阪神 (通称と逆なので注意)

---

## Section 1: OZ 6 features — 根本原因

### OZ features (6件)
```
oz_tansho_base_log   # 基準単勝オッズ log変換
oz_fukusho_base_log  # 基準複勝オッズ log変換
oz_base_pop_rank     # 基準人気順位
odds_change_rate     # リアルタイムvs基準オッズ変化率
pop_rank_change      # 人気順位変化
odds_sharp_drop      # 急落フラグ (realtime <= base * 0.8)
```

### データソース

| ファイル | 最終更新 | 最新日付 |
|---------|---------|---------|
| `data/jrdb_oz.csv` | 2026-03-31 12:04 | OZ260329 (3/29) |
| `data/jrdb/raw/Oz/` (ZIP) | 2026-03-31 | OZ260329.zip まで |
| `data/jrdb/extracted/Oz/` | 2026-05-02 | OZ260503.txt まで |

### OZ feature 3件 (oz_tansho/fukusho/base_pop) の実際の動作

**`jrdb_features.py` の KYI フォールバック (lines 1059-1085) が機能している:**

1. OZ CSV で race_id 検索 → 0件 (3/29 以降データなし)
2. `oz_tansho_base_log` が horses_df に存在しない → KYI フォールバック発動
3. `data/jrdb_kyi.csv` を再読み込み → 5/23 レースの `基準オッズ`/`基準複勝オッズ`/`基準人気順位` を取得
4. 各馬の値が有効 (例: 4.6, 49.2, 4.0 など非一様)

**判定: oz_tansho_base_log / oz_fukusho_base_log / oz_base_pop_rank = KYI フォールバックで実値提供済み → 実害軽微**

### OZ feature 3件 (change系) の実際の動作

`odds_change_rate`, `pop_rank_change`, `odds_sharp_drop` はリアルタイムオッズとの比較が必要。
- `predict_core.py` line ~2013: `_odds_feats_zero` 条件チェック
- 08:00 AM 時点: パリミュチュエル投票がまだ開いていない (開始 ~09:00)
- → realtime odds = 0 → change 計算不可 → デフォルト (0.0, 0, 0)

**判定: TIMING — 08:00 AM では構造的に取得不可。race_auto_notify の直前予測では利用可能。**

### OZ 根本原因まとめ

| 問題 | 原因 | 判定 |
|------|------|------|
| `jrdb_oz.csv` が 3/29 以降なし | `download_parse_jrdb_batch2.py --types oz` が daily schedule に未登録 | **BUG** |
| 5/5〜5/23 の OZ raw ZIP なし | 同上 (5/2-5/3 の extracted ファイルは存在するが CSV 未再構築) | **BUG** |
| base odds 3件: 実害なし | KYI フォールバック (`基準オッズ` 列) が正常動作 | 軽微 |
| change 3件: デフォルト 0 | リアルタイムオッズ TIMING 問題 | **TIMING** |

### OZ 修正

```bash
# 手動実行: OZ 最新ファイル取得 + CSV 再構築
python tools/download_parse_jrdb_batch2.py --types oz

# 自動化: tools/daily_jrdb_kyi.bat に追加
python tools/download_parse_jrdb_batch2.py --types oz >> %LOGFILE% 2>&1
```

---

## Section 2: PACI 12 features — 根本原因

### PACI features (12件)
```
paci_sogo_mark, paci_idm_mark, paci_jockey_mark, paci_train_mark  # マーク系 (4件)
paci_manken_idx, paci_goal_rank, paci_dochu_rank, paci_goal_diff  # 展開予想 (4件)
paci_jockey_exp_wr, paci_jockey_exp_3rd, paci_ninki_idx          # 騎手・人気指数 (3件)
gaisha_rank                                                         # 外厩ランク (1件)
```

### データソース

| ファイル | 最終更新 | 最新日付 |
|---------|---------|---------|
| `data/jrdb_paci.csv` | 2026-05-12 06:11 | PACI260510 (5/10) まで |
| `data/jrdb/raw/Paci/` | 2026-05-12 | PACI260510.zip まで |

### 5/23 の実動作

**ログより:**
```
[JRDB] PACI馬名フォールバックでX/N馬取得
```
- 京都レース: 6〜12/14〜16 頭取得 (50〜80%)
- 新潟レース: 1〜10/14〜16 頭取得 (7〜67%, 最低は新馬戦・未経験馬)
- 東京レース: 5〜11/12〜18 頭取得 (40〜70%)

馬名フォールバック = 各馬の最新 PACI レコードを horse_name で検索 (過去データ、stale)

### PACI coverage 詳細

| 会場 | kai | 5/23 day | PACI 最新 day | 欠損 days |
|------|-----|---------|--------------|---------|
| 京都 (08) | 03 | 09 | 06 | day07, 08, 09 (5/17, 5/23, 5/24) |
| 新潟 (04) | 01 | 07 | 02 | day03-07 (5/9〜5/25) |
| 東京 (05) | 02 | 09 | 06 | day07, 08, 09 (5/17, 5/23, 5/24) |

### PACI 根本原因

**`tools/scrape_jrdb_paci.py` が daily/weekly スケジュールに未登録。**
- 最終手動実行推定: 2026-05-12 (PACI260510 対応)
- 5/17 以降の PACI データ未取得
- 5/17, 5/23, 5/24 の当日 PACI データなし → 全予測で horse-name fallback (stale)

### PACI 修正

```bash
# 手動実行: 最新 PACI 取得
python tools/scrape_jrdb_paci.py

# 自動化: tools/daily_jrdb_kyi.bat に追加 (毎レース日朝)
python tools/scrape_jrdb_paci.py --since $(date +%Y%m%d -d '7 days ago') >> %LOGFILE% 2>&1
```

### PACI chronic 判定: **CHRONIC BUG**

4/26, 5/9, 5/10, 5/16, 5/17 のすべての予測でも同様に PACI race_id 未マッチ → horse-name fallback。

---

## Section 3: KTA / KKA / SR — 根本原因

### KTA 3 features (jrdb_kta_idm, jrdb_kta_ten_pred, jrdb_kta_agari_pred)

**JRDB KTA = 登録地データ (調教師・騎手の地区成績指数)**

#### coverage (5/23 時点)

| 会場 | 5/23 race_id | KTA 最新 race_id | gap |
|------|-------------|-----------------|-----|
| 京都 (08) | 202608030901 | 202608020612 (kai02 day06) | kai03 全域なし |
| 新潟 (04) | 202604010702 | NONE (0件) | 全域なし |
| 東京 (05) | 202605020901 | 202605010812 (kai01 day08) | kai02 全域なし |

#### 根本原因

KTA は `download_parse_jrdb_batch2.py --types kta` で取得されている (daily 実行)。
しかし 2026 年の KTA は yearly ZIP (`KTA_2026.zip`) ではなく individual files として提供される。

**問題: 新潟 (04) の individual KTA ファイルが JRDB サーバ上で存在しない / 提供されていない。**
また京都 kai03, 東京 kai02 の KTA ファイルは JRDB server に 5/23 時点で未提供の可能性。

**判定: CHRONIC BY_DESIGN (新潟) + TIMING (京都/東京 最新 kai)**

### KKA 2 features (jrdb_dam_rensho_avg, jrdb_bms_rensho_avg)

**JRDB KKA = 繁殖馬登録データ (母・BMS 連勝平均)**

#### coverage (5/23 時点)

| 会場 | 5/23 race_id | KKA 最新 race_id | gap |
|------|-------------|-----------------|-----|
| 京都 (08) | 202608030901 | 202608030412 (kai03 day04) | day05-09 なし |
| 新潟 (04) | 202604010702 | 202604010212 (kai01 day02) | day03-07 なし |
| 東京 (05) | 202605020901 | 202605020412 (kai02 day04) | day05-09 なし |

KKA は daily_jrdb_kyi.bat で毎朝取得 (`download_parse_jrdb_extra.py --types kka`)。
今日の 06:01 に更新されたが、JRDB サーバ上の KKA は 5/23 分を含まない (day04 まで)。

**判定: CHRONIC TIMING (KKA は JRDB 配信が通常 1〜2 週間遅延)**

### SR 1-5 features (jrdb_tb_homestr_inner + sr_*)

**JRDB SR = 前日馬場バイアス (ラップ・コーナーバイアス)**

- `jrdb_sr.csv` 最終更新: 2026-05-02 14:30
- SR は `parse_jrdb_extended.py --types srb` で SRB ファイルから生成
- SRB extracted: SRB260509 (5/9) まで存在するが、`jrdb_sr.csv` に未反映
- `parse_jrdb_extended.py` が daily schedule に**未登録**

#### 5/23 coverage

| 会場 | 5/23 race_id | SR 最新 race_id |
|------|-------------|----------------|
| 京都 (08) | 202608030901 | 202608030304 (kai03 day03 最終R) |
| 新潟 (04) | 202604010702 | 202604010104 (kai01 day01 最終R) |
| 東京 (05) | 202605020901 | 202605020304 (kai02 day03 最終R) |

SR は直前の開催日 (day07-08, day05-06 など) のバイアスデータを使うため、
5/23 直前の 5/22 (金) 時点でも SRB 再解析すれば 5/17-18 (day07-08) データが利用可能。

**判定: BUG (parse_jrdb_extended 未スケジュール) + BY_DESIGN (当日 SR は post-race)**

#### SR 修正

```bash
# 手動実行: SRB → jrdb_sr.csv 再構築
python tools/parse_jrdb_extended.py --types srb

# 自動化: tools/daily_jrdb_kyi.bat に追加
python tools/parse_jrdb_extended.py --types srb >> %LOGFILE% 2>&1
```

---

## Section 4: TIMING vs BUG 判定一覧

| データソース | features | 判定 | 理由 |
|------------|---------|------|------|
| OZ base (3件) | oz_tansho/fukusho/base_pop | **軽微 BUG → KYI fallback で補完済** | OZ CSV 未更新だが KYI fallback 正常動作 |
| OZ change (3件) | change_rate/pop_rank_change/sharp_drop | **TIMING** | 08:00 AM はパリミュ未開 |
| PACI (12件) | paci_* | **CHRONIC BUG** | scrape_jrdb_paci.py 未スケジュール |
| KTA (3件) | jrdb_kta_* | **CHRONIC BY_DESIGN + TIMING** | 新潟ゼロ、Kyoto/Tokyo lag |
| KKA (2件) | jrdb_dam/bms_rensho_avg | **CHRONIC TIMING** | JRDB サーバ配信 1-2 週遅延 |
| SR (1-5件) | jrdb_tb_homestr_inner + sr_* | **BUG** | parse_jrdb_extended 未スケジュール |

---

## Section 5: 修正オプション (data path のみ、model/logic 変更なし)

### 最高優先: 即効修正 (BUG 分類)

```bash
# 1. OZ batch2 を daily_jrdb_kyi.bat に追加
# → tools/daily_jrdb_kyi.bat 末尾に追加:
python tools\download_parse_jrdb_batch2.py --types oz >> %LOGFILE% 2>&1

# 2. SR 再構築を daily_jrdb_kyi.bat に追加
python tools\parse_jrdb_extended.py --types srb >> %LOGFILE% 2>&1

# 3. PACI を weekly scrape に追加 (金曜夜または土曜 AM)
python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1
```

### 次善: stale PACI の fill rate 改善

現在の馬名フォールバックは最新 PACI データを使うため、
`scrape_jrdb_paci.py` を毎週金曜夜に実行するだけで 5/24 以降は当日データを利用可能。

### KTA / KKA

- **KTA 新潟**: JRDB サーバ側の問題の可能性。手動で確認が必要。
- **KKA lag**: JRDB の配信ポリシー上どうしようもない。デフォルト値 (1600.0) で運用継続。

---

## Section 6: 過去予測への影響 (chronic 確認)

### 4/26 予測 (確認済み)

| feature | 状況 |
|---------|------|
| OZ | race_id 202608030201 等: **0件** → KYI fallback で対応 |
| PACI | horse-name fallback 動作 |
| KTA | 同様に chronic miss |
| KKA | 同様に chronic miss |

### 結論: **5/23 特有の問題ではない。少なくとも 4/26 (V15 稼働開始) 以降すべての予測で同一状況。**

V15 モデル学習時も PACI/KTA/KKA の coverage は同様のパターンだった可能性が高い
(V15 feature list に PACI/KTA/KKA が含まれていれば、訓練時も部分欠損→デフォルト値の状態)。

---

## Section 7: 5/23 予測品質の総合評価

### 正常に機能したデータ

| データ | fill rate | 備考 |
|--------|---------|------|
| KYI (JRDB 主力指数) | **100%** | 5/23 06:00 に正常更新 |
| SED (前走成績) | **73〜100%** | blood_num フォールバック |
| PACI (horse-name fb) | **7〜80%** | stale だが非ゼロ |
| OZ base odds (KYI fb) | **100%** | KYI 基準オッズで補完 |
| CHA (追切) | **100%** | 正常更新 |
| KKA | **0%** | デフォルト 1600.0 |

### 欠損した機能

| 機能 | 影響 | severity |
|------|------|---------|
| OZ change features (3件) | オッズ急変感知なし | 中 (レース直前予測では利用可能) |
| KTA idm/pace/agari (3件) | JRDB 予想指数なし | 中 (ただし KYI 指数で代替) |
| SR track bias (5件) | バイアス情報なし | 低 (直前雨などで変化しやすい) |

### 全体評価

KYI (22 features) は 100% 正常、SED (6-8 features) も 73-100%。
欠損の 24 features は主に補助指数 (PACI/KTA/SR) と OZ change。
V15 モデルは KYI + SED + odds_log (Pattern B) が主力なので、
**5/23 予測品質は許容範囲。ただし最適ではない。**

### 修正後の期待効果

- `batch2 --types oz` 追加 → OZ base features: KYI fallback → 実 OZ 値 (差異小)
- `scrape_jrdb_paci.py` 週次追加 → PACI fill rate 100% → stale 解消
- `parse_jrdb_extended --types srb` 追加 → SR が前日バイアスを正確に反映

---

## 付録: daily_jrdb_kyi.bat 修正案

```bat
@echo off
...
python tools\scrape_jrdb.py --type KYI --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type SED --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type TYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type CYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type JOA --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type KAB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\download_parse_jrdb_batch2.py --types kta cha oz >> %LOGFILE% 2>&1   ← oz 追加
python tools\download_parse_jrdb_extra.py --types kka jo >> %LOGFILE% 2>&1
python tools\parse_jrdb_extended.py --types srb >> %LOGFILE% 2>&1                  ← srb 追加
python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1                                 ← paci 追加 (土日のみ推奨)
```
