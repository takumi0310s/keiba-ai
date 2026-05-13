# Phase 12 audit A: TFJV JSON 構造 audit (read-only)

実行: 2026-05-13 PM、 Opus 4.7、 read-only

## A1-A2. JSON 構造

### dir 内容
- `data/jvlink/2026/04/` : 216 件 (.json)
- `data/jvlink/2026/05/` : 72 件 (.json)
- **合計 288 R per-race JSON** (commit c7d668c1、 5/10 Phase 12 PoC で 生成)
- 期間: 2026-04-13 〜 2026-05-10

### JSON 構造 (3 件 sample 確認)

```json
{
  "race_id": "202603010101",      // 12 digit netkeiba format
  "date": "2026-04-13",
  "ra": {                          // RA record
    "race_id": "202603010101",
    "date": "2026-04-13",
    "course_code": "03",
    "kai": "01",
    "nichi": "01",
    "race_num": "01",
    "race_name": "0000",           // ★ bug: 特別競走番号 raw 値 ★
    "youbi_code": "1"
  },
  "se": [                          // SE records (馬 list)
    {
      "umaban": "01",
      "wakuban": "1",
      "horse_id": "2023100992",    // 10 digit blood_num
      "horse_name": "セルシャーム"
    },
    ...
  ],
  "hr": {                          // HR record (払戻、 1 R 1 件)
    "data_kbn": "...",
    "raw_payouts": "..."
  },
  "source": "TFJV_BINARY_2026",
  "phase": "phase12_poc"
}
```

### 全 field list

| section | fields | source |
|--------|--------|--------|
| top | race_id, date, source, phase | aggregated |
| ra | race_id, date, course_code, kai, nichi, race_num, race_name, youbi_code | TFJV RA_DATA |
| se[] | umaban, wakuban, horse_id, horse_name | TFJV SE_DATA |
| hr | data_kbn, raw_payouts (truncated 200 bytes) | TFJV HR_DATA |

### race_name field 位置 + offset

- `tools/tfjv_parser.py` L62: `"race_name": (28, 60)` (offset 28, length 60 bytes)
- **真の値**: 全 RA 2026 (1194 races) で:
  - 880 races (74%) → "0000" のみ (平場 race)
  - 314 races (26%) → "0000XXXXX" prefix (重賞/特別 race、 真の race name は "0000" の後)

### 結論: 真の bug

- offset 28 は **特別競走番号 (4 bytes ASCII)** の位置
- offset 32 から **race name 本題 (60 bytes Shift-JIS)** が始まる
- current parser は 28-88 (60 bytes) を 1 field で 読んでいる
- 結果: 特別競走番号 4 bytes ("0000") + race name 本題 56 bytes が連結された string が race_name に入る
- 平場 races (重賞番号=0000) では 本題が null-padded のため、 `.rstrip("\x00 ")` 後 "0000" のみ残る

## A3. JV-Link 加入状態

### 確認結果

| 項目 | 状態 |
|------|------|
| JRA-VAN DataLab DLL | ✅ インストール済 (`C:/Windows/SysWow64/JVDTLAB/JVDTLab.dll`) |
| 32-bit Python venv | ❌ 未作成 (`C:/Users/takum/jvlink-venv/` 存在せず) |
| .env JV-Link credentials | ❌ 未設定 (.env に jvlink/jvdtlab 系 entry なし) |
| TFJV binary data | ✅ 既存 (`C:/TFJV` ※ access 検証は別 phase) |
| 直近 parsed CSV | ✅ data/tfjv/RA/SE/HR_2026.csv |

### 判定: **部分加入** (DataLab DLL ✅ + venv ❌ + credentials ❌)

→ JV-Link COM 経由 RT 取得は 5/24+ で 設定必要
→ 但し **TFJV binary 直 parse 経路は今 動作中** (288 R JSON 既存生成)

## まとめ

- JSON 構造 シンプル、 aggregated RA/SE/HR per race
- 288 R 既存 (1 ヶ月 backfill)
- race_name field の **真の bug は parser offset 28 → 32 (4 byte shift)**
- JV-Link 部分加入、 RT 取得は 5/24+
- 既存 TFJV binary 経路 (= 今のbinaryから 再 parse) で **5/13-5/16 内に race_name 修正可能**

詳細 bug confirmation は phase12_audit_parser_bug.md。
