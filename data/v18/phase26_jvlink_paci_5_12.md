# Phase 26: JV-Link parser 拡張 + PACI 修復 (5/12 実装)

caveman mode summary.

## Task #6: JV-Link parser 拡張

### file
- `tools/jvlink_parser.py` (拡張)

### 追加 dataspec (24 件、 5/12 時点 計 32)

#### Phase 22 base (既存 8 件、 無改変)
- RACE (RA), SE, HR, UM, BLOD (HN/SK/BT), WOOD (WC), TCOV (TK), O1

#### Phase 26 新規 蓄積系 (10 件)
| dataspec | record | 用途 |
|---|---|---|
| **DIFF** | DIFF | ★最重要、 馬/騎手/調教師/血統/累計 master 一括 |
| MING | DM | DataMining 公式予想 (展開/タイム) |
| SLOP | HC | 坂路調教 record |
| HOSE | HS | 市場取引価格 |
| HOYU | HY | 馬名由来 |
| COMM | CC | コース距離別 record |
| YSCH | YS | 年間 開催スケジュール |
| TOKU | TK2 | 特別登録 |
| RACE2 | RA | 当日 蓄積 (RACE と同様) |
| MV | MV | 動画系 metadata |

#### Phase 26 新規 速報系 (14 件)
| dataspec | record | 用途 |
|---|---|---|
| 0B11 | WH | 馬体重 LIVE |
| 0B14 | WE | 馬場状態 LIVE |
| 0B15 | HR | 確定 結果 速報 |
| **0B20** | TC/TO | ★ 朝 critical、 出走取消速報 |
| 0B30 | O1-O6 | 速報オッズ 全式 一括 |
| 0B31 | O1 | 速報単勝 |
| 0B32 | O2 | 速報馬連 |
| 0B33 | O3 | 速報ワイド |
| 0B34 | O4 | 速報馬単 |
| 0B35 | O5 | 速報三連複 |
| 0B36 | O6 | 速報三連単 |
| **0B41** | O1H | ★ V20 重要、 オッズ時系列 単複 (1年保持) |
| **0B42** | O2H | ★ V20 重要、 オッズ時系列 馬連 (1年保持) |
| 0B51 | WF | WIN5 速報 |

### parse method 追加 (合計 36)
ra, se, hr, um, blod, wc, tk, o1 (8 既存)
+ diff, dm, hc, hs, hy, cc, ys, tk2, mv (9 蓄積系)
+ wh, we, tc, to, o2, o3, o4, o5, o6, o1h, o2h, wf (12 速報系)
+ hn, sk, bt, ks, ch, bn, ck (7 DIFF/BLOD 内 record 用)

### parse dispatch 拡張
- `JVLinkParser.parse()` を改修
- 速報系 (mode=rt) は raw 先頭 2 bytes の record_type で再 dispatch
- BLOD 等 複合 record の primary は最初の type を採用

### 動作確認 (64-bit でできる範囲)
- [x] `py_compile` PASS
- [x] `python tools/jvlink_parser.py --list` 32 dataspecs 表示 (17 open / 15 rt)
- [x] `--dry-run` で `DIFF` の schema 確認 OK
- [x] fake bytes で `parse('0B11', ...)` → record_type='WH', umaban='11', horse_weight='1' (skeleton)
- [x] fake bytes で `parse('0B20', ...)` → record_type='TC', cancel_kbn='TC'
- [x] 全 36 parse method 存在確認 OK

### 32-bit 実行 doc
README 既存 docstring 内 に追加:
```
python tools/jvlink_parser.py --datatype DIFF --from 20260101 --max 5000
python tools/jvlink_parser.py --datatype 0B30 --realtime --raceid 202605070611
python tools/jvlink_parser.py --datatype 0B41 --realtime --raceid 202605070611
```

### 注意
- Phase 22 既存 parse_*  spec は 不変 (生 byte offset)
- Phase 26 新規 parse_* は skeleton (offset placeholder)、 6/9-6/13 で本実装予定
- V15 production (predict_core.py / V15 model) 完全不変

---

## Task #13: PACI 修復

### file
- `tools/scrape_jrdb_paci.py` (新規)

### 原因 (調査結果)
- `data/jrdb_paci.csv` mtime = 2026-05-03 09:45
- 5/3 以降 PACI ZIP 自体は `data/jrdb/raw/Paci/` に取得済 (PACI260509.zip まで)
- `data/jrdb/extracted/Paci/KYI*.txt` も 5/9 まで存在
- **`tools/parse_jrdb.py` が 5/3 以降 再実行されていない** → CSV 古いまま
- daily_jrdb_kyi.bat は scrape_jrdb.py (LZH 個別) を呼ぶが PACI ZIP 取得 + parse_jrdb 再生成 step 無い

→ 結論: ZIP は届いている。 ZIP → KYI 抽出 → jrdb_paci.csv 再生成 の 連結 step が欠落。

### 修復 アプローチ
2-step pipeline:

1. **download step**: JRDB index ページ から PACI*.zip 一覧取得
   - `tools/download_jrdb.py` の `get_paci_dates()` ロジック 流用
   - HTTPBasicAuth + Shift-JIS index page parse
   - 既存 ZIP は skip、 新規のみ DL
   - DL 直後 ZIP 展開 → `data/jrdb/extracted/Paci/`

2. **parse step**: `tools/parse_jrdb.py` subprocess で 起動
   - 既存 main() を流用 (改変なし)
   - 全 KYI*.txt を再 parse して `data/jrdb_paci.csv` 全件 再生成
   - 前/後 の file size diff を log

### CLI
```
python tools/scrape_jrdb_paci.py                        # 最新 PACI 取得 + parse
python tools/scrape_jrdb_paci.py --dry-run              # 取得対象 list のみ
python tools/scrape_jrdb_paci.py --years 2026           # 2026 のみ
python tools/scrape_jrdb_paci.py --since 20260403       # 4/3 以降のみ
python tools/scrape_jrdb_paci.py --skip-download        # parse のみ
python tools/scrape_jrdb_paci.py --skip-parse           # DL のみ
```

### 動作確認
- [x] `py_compile` PASS
- [x] `--help` 表示 OK (6 options)
- [x] `--dry-run --years 2026 --since 20260403` 実行 OK
  - 当 worktree には .env 無し → "no-credentials" で graceful exit
  - parse step も dry-run で skip 表示 (current jrdb_paci.csv: missing は worktree 故)

### 運用案 (実機運用 時)
- daily_jrdb_kyi.bat 末尾 に 1 行追加:
  ```
  python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1
  ```
- 朝 6:00 JRDB KYI fetch 直後に PACI も同期 → 朝 8:00 daily_predict 時 fresh CSV
- もしくは個別タスク `DailyJrdbPaci` 登録 (5:30 推奨、 KYI 取得後)

### V15 投資保護
- 既存 file 改変なし
- `data/jrdb_paci.csv` は 上書 (parse_jrdb の動作仕様、 5/3 形式と同 schema、 全件 再生成)
- predict_core.py / V15 model / app.py 不変
- 既存 ZIP 群 は `download_zip()` 内 `skip` 判定 で 損なわれない

---

## deliverable check
- [x] tools/jvlink_parser.py 拡張 (24 dataspec 追加、 計 32)
- [x] tools/scrape_jrdb_paci.py (新規、 dry-run + skip option 完備)
- [x] data/v18/phase26_jvlink_paci_5_12.md (本 doc)
- [x] py_compile 両 file PASS
- [x] V15 production 不変
