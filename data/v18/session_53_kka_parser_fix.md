# Session #53 B: KKA parser 修復実装 + 動作検証

**実施**: 2026-05-09
**branch**: dev/sprint6-kka
**実装**: `tools/jrdb_kka_parser_v2.py` 新規 (既存 parser は不変)
**出力**: `data/jrdb_kka_v2.csv` (gitignore 済、 196 MB)

---

## 1. fix の核心

### Before (`tools/download_parse_jrdb_extra.py:336`)

```python
def _parse_level_12(val_str):
    v = val_str.strip()                  # ← 12-byte field を最初に strip
    if len(v) < 12:                      # ← 内側空白で 12 chars 未満になり常に True
        return None, None, None, None    # ← 全 record で None
    ...
```

### After (`tools/jrdb_kka_parser_v2.py:_parse_level_12_v2`)

```python
def _parse_level_12_v2(val_str: str):
    """12 byte の ZZ9*4 を 3-byte ずつ切って int 化 (strip は _safe_int 内部)。"""
    if len(val_str) < 12:
        return None, None, None, None
    return (
        _safe_int(val_str[0:3]),   # 1 着  (例: "  0" → 0)
        _safe_int(val_str[3:6]),   # 2 着  (例: "  1" → 1)
        _safe_int(val_str[6:9]),   # 3 着
        _safe_int(val_str[9:12]),  # 着外
    )
```

→ field を slice する前に strip しない。 `_safe_int` が内部で個別の slice を strip + int 変換。

---

## 2. 動作検証

### smoke test (1 file, KKA200201.txt)

```
KKA files found: 1228
Parsing: 1 files
  Parsed: 348 rows (0 errors)
--- Coverage (% non-null) ---
  any seiseki_*: 100.0%
  jra_seiseki_1: 94.3%
  kyori_seiseki_1: 94.3%
  track_seiseki_1: 94.3%
```

### 全 file 実行 (1228 files)

```
KKA files found: 1228
Parsing: 1228 files
  Parsed: 548,606 rows (0 errors)
--- Coverage (% non-null) ---
  any seiseki_*: 100.0%
  jra_seiseki_1: 90.4%
  kyori_seiseki_1: 90.4%
  track_seiseki_1: 90.4%
--- Year coverage (rows) ---
    2015:  49,992
    2016:  50,076
    2017:  49,299
    2018:  48,618
    2019:  47,574
    2020:  48,282
    2021:  47,821
    2022:  47,220
    2023:  47,672
    2024:  47,181
    2025:  47,884
    2026:  16,987
Saved: data/jrdb_kka_v2.csv (548,606 rows, 100 cols)
```

---

## 3. coverage 結果

| metric | v1 (broken) | v2 (fixed) | 達成 |
|---|---:|---:|:---:|
| **jra_seiseki_1 non-null** | **0.0%** | **90.4%** | ✅ 80%+ 目標 達成 |
| any seiseki_* non-null | 0.0% | 100.0% | ✅ |
| total rows | (count OK だが値全 NaN) | 548,606 | ✅ |
| 期間 coverage | 2015-2026 | 2015-2026 (12 年) | ✅ |
| parse errors | 0 | 0 | ✅ |

### 残り 9.6% non-null について

- `jra_seiseki_1 == None` のケース = JRA 中央成績がまだ無い (新馬・地方転入直後など)
- `_safe_int` が空文字 / "-" / 非数値を None で返している正常動作
- spec の 「該当成績無しは空白」 に合致

→ **真に取得できなかった** rate ではなく、 **データ上 値が存在しない** rate。

---

## 4. 出力 schema (100 columns)

```
race_id, umaban,
jra_seiseki_{1,2,3,out},        # JRA 中央成績
koryu_seiseki_{1,2,3,out},      # 交流成績
other_seiseki_{1,2,3,out},      # その他成績
turf_dirt_2_{1,2,3,out},        # 芝ダ 2 着成績
turf_dirt_2_dist_{1,2,3,out},   # 芝ダ 2 着距離別
track_seiseki_{1,2,3,out},      # トラック (右回り/左回り) 別
rotation_sei_{1,2,3,out},       # ローテーション
kyori_seiseki_{1,2,3,out},      # 距離別
saka_seiseki_{1,2,3,out},       # 坂別
heavy_seiseki_{1,2,3,out},      # 重馬場
rest_seiseki_{1,2,3,out},       # 休み明け
class_seiseki_{1,2,3,out},      # クラス別
speed_seiseki_{1,2,3,out},      # S ペース
slow_seiseki_{1,2,3,out},       # N ペース
mid_seiseki_{1,2,3,out},        # T ペース
season_seiseki_{1,2,3,out},     # 季節別 (春/夏/秋/冬)
waku_seiseki_{1,2,3,out},       # 枠別 (内/中/外)
breeder_dist_{1,2,3,out},       # 産駒距離
breeder_track_{1,2,3,out},
breeder_adjust_{1,2,3,out},
breeder_baba_{1,2,3,out},
breeder_blanker_{1,2,3,out},
breeder_surface_{1,2,3,out},
dam_rensho_max, dam_rensho_min, dam_rensho_avg,
bms_rensho_max, bms_rensho_min, bms_rensho_avg,
```

→ 23 個の level 集計 x 4 (1着/2着/3着/着外) = 92 numeric features
→ + 6 単一 numeric (連勝率系)
→ = 計 **98 features 候補**

---

## 5. 既存 csv との整合性

- 既存 `data/jrdb_kka.csv` (broken) は **不変** (上書きせず)
- 新 `data/jrdb_kka_v2.csv` は別 file
- `tools/download_parse_jrdb_extra.py` は **読み取りのみ参照**、 変更なし
- → existing 運用 (V15 daily_predict 等) は完全不変

---

## 6. 次の action (Section C)

新 features を expanding window で集計し、 V15 baseline との AUC contribution を測定。 リーク厳禁 (CLAUDE.md 教訓: dam_top3r / SKB POST-RACE LEAK / sib_top3_rate hybrid)。
