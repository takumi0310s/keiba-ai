# Phase 12 audit C: 17 features parse 可能性評価

実行: 2026-05-13 PM、 Opus 4.7、 read-only

## C1. 17 features 全 list (Phase 12 originally)

### グループ 1: オッズ拡張 4 件 (O1/O2/O5)

| feature | 内容 | source record |
|---------|-----|--------------|
| o1_change_3h_v18 | 単勝オッズ 3h 前 → 直前 変化 | O1 (単複オッズ、 時系列複数) |
| o1_change_30m_v18 | 単勝オッズ 30m 前 → 直前 変化 | O1 |
| o2_winrate_v18 | 馬連オッズ 由来 馬の implied winrate | O2 (馬連オッズ) |
| o5_change_v18 | 三連複オッズ 変化 | O5 (三連複オッズ) |

### グループ 2: 番組情報 3 件 (RA/BT)

| feature | 内容 | source record |
|---------|-----|--------------|
| race_class_v18 | レースクラス (新馬/未勝利/1勝/2勝/3勝/OP/L/G3/G2/G1) | RA (offset 212+) |
| race_grade_v18 | 重賞 grade (G1/G2/G3/G/L) | RA (grade code) |
| race_distance_class_v18 | 距離区分 (短/マイル/中/長) | RA (distance + 計算) |

### グループ 3: ハロン+天候 6 件 (SE/WE/WH)

| feature | 内容 | source record |
|---------|-----|--------------|
| se_pace_v18 | SE ペース (前半 3F / 後半 3F) | SE (offset 後段) |
| se_lap_3f_v18 | SE 上がり 3F | SE (offset 後段) |
| we_temperature_v18 | WE 気温 | WE (天候マスタ) |
| wh_track_condition_v18 | WH 馬場状態 | WH (馬場マスタ) |
| we_wind_v18 | WE 風 | WE |
| wh_rainfall_v18 | WH 雨量 | WH |

### グループ 4: 血統拡張 4 件 (UM/SK/BR)

| feature | 内容 | source record |
|---------|-----|--------------|
| um_sire_winrate_v18 | 父勝率 (TFJV UM record 父名 → SE history 集計) | UM + SE |
| um_broodmare_winrate_v18 | 母父勝率 (UM 母父 → SE history) | UM + SE |
| sk_pedigree_class_v18 | 血統クラス | SK (系統マスタ) |
| br_inbreeding_score_v18 | インブリード score 5代 | BR (血統5代) |

## C2-C3. 各 features の TFJV JSON 内 location + 難易度

### 簡単 (即 parse 可能、 30 分-1h) — 4 features

| feature | 場所 | 実装 |
|---------|-----|------|
| **race_name 修正** (★ bug fix ★) | RA offset 28 → 32 | parser L62 修正 + re-extract |
| race_class_v18 | RA offset 212-225 範囲 (各 race の 競走種別 code / 競走条件 code) | RA offsets 追加 + 番組 code → class マッピング辞書 |
| race_grade_v18 | RA offset 75-79 (重賞 code、 G1=A/G2=B/G3=C 等) | RA offsets 追加 + grade dict |
| race_distance_class_v18 | RA offset 692-696 (距離) 既 parse 候補 | RA offsets 追加 + 距離 binning |

**実装工数**: 1-2h
- tools/tfjv_parser.py RA_FIELD_OFFSETS 拡張 (15-20 fields 追加)
- RA_2026.csv re-extract (10 sec、 1194 records)
- per-race JSON 再生成 (10 sec、 288 R)
- features 4 件 即真値化 OK

### 中 (一部複雑、 2-3h) — 6 features

| feature | 場所 | 難易度 理由 |
|---------|-----|----------|
| se_pace_v18 | SE offset 後段 (各馬の通過順位 + 前半/後半 タイム) | SE offsets 拡張、 但し race 単位集計が必要 (12-18 頭 / race) |
| se_lap_3f_v18 | SE offset 後段 (上がり 3F field、 1/10秒) | 同上、 SE field 拡張 |
| we_temperature_v18 | WE record (TFJV 天候マスタ、 WE_DATA か WC_DATA) | WE parser 新規実装、 spec ref 必要 |
| wh_track_condition_v18 | WH record (馬場マスタ) | WH parser 新規実装 |
| we_wind_v18 | WE record (風速 / 方向) | 同上 |
| wh_rainfall_v18 | WH record (含水率 / 雨量) | 同上 |

**実装工数**: 2-3h
- SE parser 拡張 (offset 100-200 範囲、 通過順 / タイム / 上がり)
- WE_FIELD_OFFSETS, WH_FIELD_OFFSETS 新規定義 (PARSERS dict 追加)
- WE/WH binary file C:/TFJV 内 location 確認
- per-race JSON への WE/WH section 追加

### 難 (大規模 refactor、 1-2 day) — 7 features

| feature | 場所 | 難易度 |
|---------|-----|-------|
| o1_change_3h_v18 | O1 records (1 R に 複数 時刻 records、 例 -180m/-30m/-5m) | 時系列 取得 + diff 計算、 race_id × time series merge |
| o1_change_30m_v18 | 同上 | 時系列 取得 |
| o2_winrate_v18 | O2 records (馬連 N×(N-1)/2 通り、 各馬の implied prob 計算) | 馬連 → tansho 推定 (matrix decomp、 算法) |
| o5_change_v18 | O5 records (三連複、 6000-12000 通り / race) | data size 巨大、 時系列 集約 |
| um_sire_winrate_v18 | UM record 父名 + SE history 集計 (expanding) | 親-子 hierarchy 構築 + 各 sire の cumsum |
| um_broodmare_winrate_v18 | UM 母父名 + SE history | 同上 |
| sk_pedigree_class_v18 | SK record (系統マスタ、 spec 別) | SK parser 新規 + 系統 dict 構築 |
| br_inbreeding_score_v18 | BR record + 5代 血統 algorithm | 5代 walk algorithm (重複 ancestor 検出 + ratio score) |

**実装工数**: 1-2 day (8h-16h)
- O1/O2/O5 parser 拡張 + 時系列 merge (data size 巨大)
- UM-SE 親子 history merge (既存 sire_*_expanding と統合可能)
- SK / BR parser 新規 (spec ref 必要)
- 各 features Bayesian smoothing + expanding window

## C4. 即 parse 可能 features 識別 (5/14-5/16 で先に真値化)

### 5/14 (水) PM 着手可能 (1-2h)

★ **4 features 即真値化** ★:
1. ~~race_name~~ (bug 修正、 即化前提)
2. race_class_v18
3. race_grade_v18
4. race_distance_class_v18

→ 5/14 内に **4 features 真値化 + RA offset 確実** → 5/15 以降の 中 features 着手

### 5/15-5/16 着手可能 (2-3h)

★ **6 features 真値化** (中) ★:
5. se_pace_v18
6. se_lap_3f_v18
7. we_temperature_v18
8. wh_track_condition_v18
9. we_wind_v18
10. wh_rainfall_v18

→ 5/16 までに **合計 10 features 真値化**、 V20 真の学習 一部前倒し 可能

### 5/24+ (難、 集中作業)

★ **残 7 features** (難):
11. o1_change_3h_v18
12. o1_change_30m_v18
13. o2_winrate_v18
14. o5_change_v18
15. um_sire_winrate_v18
16. um_broodmare_winrate_v18
17. sk_pedigree_class_v18 (or br_inbreeding_score_v18)

→ 5/24+ JV-Link COM 環境 + 32-bit venv + credentials 設定後、 1-2 day 集中で 全 17 features 真値化

## 結論

- **★ 簡単 4 / 中 6 / 難 7 ★** = 合計 17 features
- 5/14-5/16 で **10 features 真値化 可能** (簡単 4 + 中 6、 4-5h、 通常週末作業 規模)
- 残 7 features は 5/24+ JV-Link 32-bit venv 環境 + credentials 設定後 集中作業

詳細 工数評価 + 5/24+ 計画影響は phase12_audit_workload.md。
