# Phase 12 audit B: race_name offset bug 根本確認

実行: 2026-05-13 PM、 Opus 4.7、 read-only

## B1-B3. 関連 file + parser logic 確認

### file list

| file | 役割 |
|------|------|
| `tools/tfjv_parser.py` | TFJV binary .DAT → CSV 抽出 (RA/SE/HR offset 定義) |
| `tools/jvlink_backfill_phase12_poc.py` | 抽出 CSV → per-race JSON 集約 (read-only re-shape) |
| `tools/predict_core_v18_phase12.py` | (存在確認のみ、 後段 用) |
| `data/tfjv/RA_2026.csv` | 既 parse 済 RA records (1194 行) |
| `data/jvlink/2026/{04,05}/*.json` | 288 件 per-race JSON 出力 |

### parser logic (現状)

**tools/tfjv_parser.py L48-64 (RA_FIELD_OFFSETS)**:
```python
RA_FIELD_OFFSETS = {
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "year":         (3,  4),
    "month_day":    (7,  4),
    "year_create":  (11, 4),
    "create_md":    (15, 4),
    "course_code":  (19, 2),
    "kai":          (21, 2),
    "nichi":        (23, 2),
    "race_num":     (25, 2),
    "youbi_code":   (27, 1),
    "race_name":    (28, 60),   # ★ bug: 4 byte 不足、 28 → 32 が正解 ★
}
```

**parse_record() L142-159**:
```python
def parse_record(raw: bytes, schema: dict, encoding: str = "shift_jis") -> dict:
    out = {}
    for name, (start, length) in schema.items():
        if length == -1:
            chunk = raw[start:]
        else:
            chunk = raw[start:start+length]
        try:
            if name in ("horse_name", "race_name") or name.endswith("_kanji"):
                val = chunk.decode(encoding, errors="replace").rstrip("\x00 ")
            else:
                val = chunk.decode("ascii", errors="replace")
        except Exception:
            val = chunk.hex()
        out[name] = val.strip()
    return out
```

logic 自体は OK (Shift-JIS decode + null/space strip)。 問題は **offset spec の 4 byte 不足**。

## B4. 真の bug 識別

### 4 候補 評価

| 仮説 | 評価 | 結論 |
|------|------|------|
| offset 計算 logic 誤り | NO | 数式 + slice は正常、 (start, length) tuple 計算 OK |
| regex pattern 誤り | NO | regex 使用なし、 fixed-length slice のみ |
| encoding 誤り | NO | Shift-JIS は spec 通り、 horse_name (offset 40-76 of SE) は実 値正常 (例: "セルシャーム") |
| **JSON schema (offset spec) 想定誤り** | **YES** | RA spec で youbi_code(1) の後に **特別競走番号(4 bytes ASCII)** が抜けている |

### 真の bug

**JV-Data RA record layout** (JRA-VAN spec from https://jra-van.jp/dlb/manual/recordlayout/):

| offset | length | field | current parser |
|-------|-------|-------|----------------|
| 0 | 2 | レコード種別ID | ✅ record_type |
| 2 | 1 | データ区分 | ✅ data_kbn |
| 3 | 8 | データ作成年月日 | ✅ year(4)+month_day(4) |
| 11 | 8 | 開催年月日 | ✅ year_create(4)+create_md(4) ※ 名前は逆、 実 開催年月日 |
| 19 | 2 | 開催場コード | ✅ course_code |
| 21 | 2 | 開催回 | ✅ kai |
| 23 | 2 | 開催日 | ✅ nichi |
| 25 | 2 | レース番号 | ✅ race_num |
| 27 | 1 | 曜日コード | ✅ youbi_code |
| **28** | **4** | **特別競走番号 (4 bytes ASCII、 平場=0000)** | ❌ **抜け** |
| 32 | 60 | レース名本題 (60 bytes Shift-JIS) | ❌ **race_name が offset 28 を読んでいる** |
| 92 | 60 | レース名副題 | ❌ 未読 |
| 152 | 60 | レース名カッコ内 | ❌ 未読 |
| 212+ | ... | コース距離 / 馬場 / 賞金 etc. | ❌ 未読 (これが 17 features 一部) |

### bug 確定 (real data 検証)

`data/tfjv/RA_2026.csv` 全 1194 行 確認:
- race_name 値の**先頭 4 char が 全 "0000"** (重賞/特別 含む)
  - 重賞 race の真の race_name (例: "皐月賞") は "0000皐月賞" として記録されている
  - 平場 race は "0000" のみ
- これは **offset 28-32 = 特別競走番号 (=0000 for 平場 / =0001+ for 重賞)** + offset 32-88 = race_name 56 bytes (= 60 - 4) という 連結 string

### B5. 1 R で 真の bug 再現 + 修正案

**再現**: `data/tfjv/RA_2026.csv` row 0:
- race_name = "0000" (= 特別競走番号 raw、 平場 race)

**修正案 1: 最小修正 (4 byte shift)**:
```python
RA_FIELD_OFFSETS = {
    ...
    "youbi_code":     (27, 1),
    "tokubetsu_num":  (28, 4),    # ★ 追加: 特別競走番号 ★
    "race_name":      (32, 60),   # ★ 修正: 28 → 32 ★
}
```

**修正案 2: 副題 + 完全 spec**:
```python
RA_FIELD_OFFSETS = {
    ...
    "youbi_code":      (27, 1),
    "tokubetsu_num":   (28, 4),
    "race_name":       (32, 60),
    "race_name_sub":   (92, 60),
    "race_name_paren": (152, 60),
    # ※ 完全 spec は 1242 bytes、 ここでは 6 field 追加程度に留める
}
```

### 実装工数

- 修正案 1 のみ: **30 分** (tools/tfjv_parser.py L62 編集 + RA_2026.csv 再 extract + JSON 再生成)
- 修正案 2 (本題 + 副題): **1h** (副題は 平場 race の正式名取得に必要)

### V15 投資保護

- tfjv_parser.py 修正は read-only 出力 (data/tfjv/ 別 dir)、 V15 .pkl.gz / predict_core / app.py 完全不変
- 修正は 5/24+ Phase 13 で実装、 本 audit は **読取のみ** で 提案

## 結論

**真の bug**: tools/tfjv_parser.py L62 `RA_FIELD_OFFSETS["race_name"] = (28, 60)` が 4 byte 不足。 **特別競走番号 field (offset 28-32) の skip 抜け**。 修正は offset 28 → 32、 + tokubetsu_num field 4 byte 追加で 30 分-1h で 完了。

5/24+ JV-Link COM 環境への移行時にも 同 spec 流用可能 (binary layout は JV-Link RT も同一)。

詳細 17 features feasibility は phase12_audit_features_feasibility.md。
