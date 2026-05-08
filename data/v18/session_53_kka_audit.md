# Session #53 A: JRDB KKA parser audit (root cause 特定)

**実施**: 2026-05-08
**branch**: dev/sprint6-kka (origin/main 6c0680ad から分岐)
**目的**: jrdb_kka.csv の seiseki_* 全 NaN (0%) の root cause 特定

---

## 1. KKA 概要 (JRDB 公式 spec より)

- **datatype**: KKA = 競走馬拡張 (出走馬の集計成績)
- **file 名**: `KKA{YYMMDD}.txt` (例: `KKA200201.txt` → 2020-02-01 開催分)
- **format**: 固定長テキスト、 322 byte/record + 2 byte CRLF = 324 byte
- **encoding**: Shift-JIS / CP932
- **更新頻度**: 月木 19:00、 金土 20:00
- **位置**: ローカル `data/jrdb/extracted/Paci/KKA*.txt`

### record 構造 (主要 field)

| offset | length | field | 内容 |
|---:|---:|---|---|
| 1-2 | 2 | basho_code | 場コード |
| 3-4 | 2 | year | 年 (2 桁) |
| 5 | 1 | kai | 開催回 |
| 6 | 1 | nichi | 開催日 |
| 7-8 | 2 | race_num | R |
| 9-10 | 2 | umaban | 馬番 |
| 11-22 | 12 | jra_seiseki | JRA 成績 (ZZ9*4 = 4 区分 x 3 桁: 1着/2着/3着/着外) |
| 23-34 | 12 | koryu_seiseki | 交流成績 (同上) |
| 35-46 | 12 | other_seiseki | その他成績 |
| 47-58 | 12 | turf_dirt_2着 | 芝ダ 2 着成績 |
| ... | ... | ... | (各 12 byte で同じ ZZ9*4 構造) |
| 287-289 | 3 | dam_rensho_max | 母産駒最連勝率 |
| 290-292 | 3 | dam_rensho_min | 母産駒最低連勝率 |
| 293-296 | 4 | dam_rensho_avg | 母産駒平均連勝距離 |
| 297-299 | 3 | bms_rensho_max | 母父産駒最連勝率 |
| 300-302 | 3 | bms_rensho_min | 母父産駒最低連勝率 |
| 303-306 | 4 | bms_rensho_avg | 母父産駒平均連勝距離 |
| 307-322 | 16 | reserve | 予備 |

### `ZZ9*4` (12 byte) format の特徴

各 3 byte が 1 つの数値で **右詰め空白 padding**:

```
"  0  1  1  2"    ← 4 つの数値 (0, 1, 1, 2)
 │  │  │  │
 │  │  │  └─ 着外
 │  │  └──── 3着
 │  └─────── 2着
 └────────── 1着
```

→ **field 内に空白が含まれる** ため、 全体を `.strip()` すると先頭の空白だけ除去されて長さが減る。

---

## 2. 既存 parser の現状

**file**: `tools/download_parse_jrdb_extra.py` (lines 286-389)

### 既存 logic

```python
def _parse_level_12(val_str):
    """12バイトのレベルデータ(ZZ9*4)をパース → (1着, 2着, 3着, 着外)"""
    v = val_str.strip()                  # ← BUG: 先に strip すると 12 chars 未満になる
    if len(v) < 12:
        return None, None, None, None    # ← 全 record で None になる
    try:
        w = _safe_int(v[0:3])
        ...
```

### 出力 csv (`data/jrdb_kka.csv`) 現状

- size: 48 MB
- columns: 52
- header: race_id, umaban, jra_seiseki_1/2/3/out, koryu_seiseki_*, kyori_seiseki_*, track_seiseki_*, heavy_seiseki_*, rest_seiseki_*, class_seiseki_*, ...
- row 1 (race_id=201508010102, umaban=2): **全 seiseki_* が空文字 (NaN)** ← 確認済

---

## 3. ROOT CAUSE (再現済)

**bug 箇所**: `tools/download_parse_jrdb_extra.py:336`

```python
v = val_str.strip()
```

**再現テスト結果** (`KKA200201.txt` の最初の line):

```
Line length: 322
Raw field at offset 11-22: b'  0  1  1  2'
Decoded:                   '  0  1  1  2'   (12 chars)
After .strip():             '0  1  1  2'    (10 chars)
len < 12 → True → return (None, None, None, None)

→ jra_seiseki_1: None
→ jra_seiseki_2: None
→ jra_seiseki_3: None
→ jra_seiseki_out: None
```

**原因まとめ**:
- `ZZ9*4` 形式は **3 byte ごとに右詰め空白 padding された 4 つの数値**
- field の **先頭が空白で始まる** ケースが多い (1 桁数値の場合 "  0" のように padding される)
- `val_str.strip()` で先頭空白を除去すると 12 chars 未満になり、 length check で reject
- → 結果として **全 record の全 ZZ9*4 field が None** になる
- → 17 個の level field x 4 (1着/2着/3着/着外) = **68 columns 全部が NaN**

### 影響範囲

```
seiseki 系 columns (全 NaN):
  jra_seiseki_1/2/3/out
  koryu_seiseki_1/2/3/out
  kyori_seiseki_1/2/3/out
  track_seiseki_1/2/3/out
  heavy_seiseki_1/2/3/out
  rest_seiseki_1/2/3/out
  class_seiseki_1/2/3/out
  (他、 同様の field 多数)
```

### 影響を受けない columns

- race_id ✅ (parse_kka_line の `_build_race_id` で正常生成)
- umaban ✅ (`_safe_int(row['umaban'])` で正常)
- dam_rensho_max/min/avg ✅ (3-4 byte の単一数値、 `_safe_int` 直)
- bms_rensho_max/min/avg ✅

---

## 4. 修復方針

`_parse_level_12` を **strip 前に 3-byte ずつ slice する** 形に変更:

```python
def _parse_level_12_v2(val_str):
    """12 byte の ZZ9*4 を 3-byte ずつ切って int 化 (strip は内部 _safe_int で実施)"""
    if len(val_str) < 12:
        return None, None, None, None
    return (
        _safe_int(val_str[0:3]),   # 1着
        _safe_int(val_str[3:6]),   # 2着
        _safe_int(val_str[6:9]),   # 3着
        _safe_int(val_str[9:12]),  # 着外
    )
```

`_safe_int` は内部で `s.strip()` を呼ぶので、 各 3-byte slice の空白は問題なし。

→ Section B で実装。

---

## 5. 確認した事実

- ✅ KKA file は ローカルに **存在し、 normal にダウンロード済** (`data/jrdb/extracted/Paci/KKA*.txt`、 多数)
- ✅ encoding は CP932 で正常 decode 可能
- ✅ record 長 322 byte は spec と一致
- ✅ race_id / umaban / dam_rensho / bms_rensho は **正常 parse**
- ❌ **17 個の ZZ9*4 level field 全部が NaN** ← parser bug が原因 (file は正常)
- ✅ root cause 特定完了 (1 行の bug: `v = val_str.strip()`)
- ✅ 修復方針確立

→ Section B で fix 実装。
