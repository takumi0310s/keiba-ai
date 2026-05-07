# JV-Link fetcher 本実装 (Session #41 B)

**作成**: 2026-05-08 深夜 (Session #41 B、 ユーザー就寝中)
**前提**: Session #39 B 試作 + Session #41 A 32-bit 環境 plan
**ファイル**: `tools/jvlink_fetcher_v2.py` (新規、 Session #39 試作の拡張版)

---

## 1. Session #39 試作版 → 本実装版 (V2) の差分

| 項目 | Session #39 (試作) | Session #41 V2 (本実装) |
|------|----------------|------------------|
| 行数 | 170 | 280 |
| record parser | なし (raw CSV のみ) | RA / SE / HR / O1 placeholder + auto_parse |
| 多 datatype 一括 | 単一のみ | `--datatypes RACE,SE,HR,O1` 対応 |
| 出力 | raw CSV 1 ファイル | raw CSV + parsed CSV + meta JSON |
| 出力 path | `data/jvlink/<dtype>/<date>.csv` | 同左 + `_raw.csv` / `_parsed.csv` / `_meta.json` |
| arch 警告 | なし | 64-bit 検出時 警告 |
| sleep 制御 | なし | datatype 間 1s sleep |

---

## 2. 主要 record parser

### 2.1 RA (race info)

```
pos 0-1:  record_type "RA"
pos 2:    data_kbn ('1'新規, '2'訂正, etc.)
pos 3-7:  year (4 digit)
pos 7-11: month_day (MMDD)
pos 11-13: course_code
pos 13-15: kai
pos 15-17: nichi
pos 17-19: race_num
pos 19+:   race_name, distance, surface, class, etc. (固定長 layout、 仕様書要確認)
```

### 2.2 SE (馬毎レース情報)

```
pos 0-1:  "SE"
pos 2:    data_kbn
pos 3-19: race_id 関連
pos 21-23: umaban
pos 23-33: blood_num
```

### 2.3 HR (払戻)

```
pos 0-1:  "HR"
pos 2:    data_kbn
pos 3-19: race_id
pos 19+:   単/複/枠連/馬連/ワイド/馬単/3連複/3連単 各払戻金 (複雑な layout)
```

### 2.4 O1 (単複オッズ)

placeholder のみ、 実 record 確認後 layout 確定。

### 2.5 公式仕様書

https://jra-van.jp/dlb/manual/recordlayout/

→ Phase 3 後半 (6/9-13 V20 構築期間) で正確な parser 完成予定。

---

## 3. 利用例 (32-bit Python venv で実行)

### 3.1 単一 datatype fetch

```powershell
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_fetcher_v2.py `
    --date 20260503 --datatype RACE
# → data/jvlink/RACE/20260503_raw.csv (records)
```

### 3.2 多 datatype 一括 + parser

```powershell
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_fetcher_v2.py `
    --date 20260503 --datatypes RACE,SE,HR,O1 --parse
# → data/jvlink/RACE/20260503_raw.csv + _parsed.csv + _meta.json
# → data/jvlink/SE/20260503_*  (同様)
# → data/jvlink/HR/20260503_*
# → data/jvlink/O1/20260503_*
```

### 3.3 datatype 一覧

```powershell
python tools\jvlink_fetcher_v2.py --list-datatypes
```

---

## 4. 出力 schema

### 4.1 raw CSV

```
source_file, raw_record
"RA20260503001a.RC", "RA1 ..."
"RA20260503001a.RC", "RA1 ..."
...
```

### 4.2 parsed CSV (--parse 時)

各 record_type の主要 field を column 化:

```
_record_type, _source_file, course_code, data_kbn, kai, month_day, nichi, race_num, year, ...
"RA", "RA20260503001a.RC", "06", "1", "01", "0503", "01", "01", "2026", ...
"RA", "RA20260503001a.RC", ...
"SE", ...
```

### 4.3 meta JSON

```json
{
  "datatype": "RACE",
  "fromtime": "20260503000000",
  "option": 4,
  "n_data": 29,
  "n_files": 29,
  "last_filetime": "...",
  "n_records": 174,
  "fetched_at": "2026-05-08T01:30:00"
}
```

---

## 5. 既存 keiba-ai data source との merge plan

| データ | 旧 source | 新 source (JV-Link) | 切替 phase |
|-------|---------|----------------|------------|
| jra_payouts.csv (4/6 停止) | scrape_jra_payouts.py | HR record | 5/24+ |
| 過去 race 結果 | jra_races_full.csv | RA + SE record | 6/9-13 (V20 構築) |
| 当日 オッズ | netkeiba | O1-O6 record | 5/24+ (paci 自前算出) |
| 馬体重 (当日) | netkeiba | WF record | 5/24+ |
| 調教 | netkeiba | TCOV/WOOD record | 6/9-13 (補完) |

→ 段階的切替、 既存 source は補助として残す。

---

## 6. resume / retry 対応

JV-Link は internal で 進捗 state を持つため、 同じ option=4 で再 JVOpen すると差分のみ取得。
本 script では:
- meta JSON に `last_filetime` を記録
- 再実行時 `--option 1` で full / `--option 4` (default) で差分

---

## 7. error handling

### 7.1 既知 error

| rc | 意味 | 対処 |
|----|------|------|
| 0 | success / EOF | 正常 |
| -1 | ファイル切替 message | 継続 |
| -201 | option 不正 | option 値確認 |
| -301 | サーバー混雑 | sleep + retry |
| -503 | data 期間外 | date 確認 (5/9 はまだ未配信、 5/8 配信予定) |

### 7.2 試作 V2 の対応

- arch check (64-bit warning)
- JVOpen failure → exception raise
- JVRead loop で rc<0 → break + warning
- datatype 間 1s sleep (rate limit)

---

## 8. 5/9 V15 投資保護 (B 領域)

✅ 既存 keiba-ai (predict_core / daily_predict / V15 model) 完全不変
✅ 新規 tool は 32-bit Python venv (別環境) で動作
✅ 出力 path `data/jvlink/` は新規 dir、 既存 data に影響なし

→ **5/9 朝 V15 完全保証**

---

## 9. 結論

✅ B1: 試作 → 本実装 (Session #39 170 行 → V2 280 行)
✅ B2: 主要 record parser (RA/SE/HR/O1) placeholder
✅ B3: raw CSV / parsed CSV / meta JSON 出力
✅ B4: 既存 source merge plan
✅ B5: 統合 doc (本ファイル)

→ **5/24+ Phase 3 で本格活用可能、 6/9-13 で正確な parser 完成予定**

---

**Session #41 B 完了**
