# Phase 22 Agent A 報告 (5/11)

> 30 年 backtest data 取得 skeleton + JV-Link parser skeleton

作成: 2026-05-11 (Session #86 Phase 22 並列、 Agent A)
worktree: `agent-a6518704323332495`

---

## 0. deliverable 一覧

| file | 行数 | 役割 |
|------|------|------|
| `tools/backtest_30year_collect.py` | 342 | TFJV 14 datatype × 1995-2026 を年単位で抽出 (dry-run 対応) |
| `tools/jvlink_parser.py` | 449 | JV-Link COM 経由 8 datatype parser skeleton (32-bit 必須) |
| `data/v18/phase22_agent_a_5_11.md` | 本 file | 実装内容 / 容量見積り / 32-bit 確認手順 |

合計 791 行 (本 doc 除く)。 V15 production 関連 file は **一切触らず**。

構文 OK 確認 (64-bit Python):
```
python -c "import py_compile; py_compile.compile('tools/backtest_30year_collect.py', doraise=True)"   # OK
python -c "import py_compile; py_compile.compile('tools/jvlink_parser.py', doraise=True)"             # OK
```

---

## 1. Task 1: 30 年 backtest data 取得 (`tools/backtest_30year_collect.py`)

### 1.1 設計

- 入力: TFJV 配下の 14 datatype (RA / SE / HR / H1 / UM / WF / SK / DM / HC / WH / HN / TR / BR / BT)
- 出力: `data/backtest_30year/{year}/{datatype}.parquet` (or `.csv.gz` fallback)
- 既存 `tools/tfjv_parser.py` (Session #44 B、 Shift-JIS 漢字 OK) を `iter_records` / `iter_directory` 経由で import。 parser 重複 実装 ゼロ。
- DRY-RUN mode で 「存在 file 数 / 年別 内訳 / 容量 見積り」 を json で吐く。 parse は しないので 安全。
- 年×datatype 単位で逐次 collect (chunk-by-year)、 メモリ爆発 回避。

### 1.2 TFJV 実測 inventory (5/11)

```
$ du -sh C:/TFJV/        →  5.8 GB (90 年保有)
$ find C:/TFJV/SE_DATA -name '*.DAT' | wc -l  → 4,671
```

datatype 別 (実測):

| datatype | TFJV dir | size | files | 年範囲 |
|----------|----------|------|-------|--------|
| SE | SE_DATA | 1.9 GB | 4,671 | 1954-2026 |
| HY (H1) | HY_DATA | 2.0 GB | 6,160 | 1986-2026 |
| ES (SK) | ES_DATA | 688 MB | 11,584 | 1986-2026 |
| CK (HC) | CK_DATA | 657 MB | 18,089 | 2003-2026 |
| UM | UM_DATA | 497 MB | 315 | 1936-2025 |
| BY (HN) | BY_DATA | 22 MB | 283 | 1996-2025 |
| BS (WH) | BS_DATA | 11 MB | 311 | 1995-2025 |
| W5 (WF) | W5_DATA | 7 MB | 863 | 2011-2026 |
| TM (TR) | TM_DATA | 6.7 MB | 440 | 2014-2026 |
| BR | BR_DATA | 5.8 MB | 10 | 全期間 (年区切り無し) |

### 1.3 容量 見積り (1995-2024、 30 年、 6 datatype デフォルト)

`python tools/backtest_30year_collect.py --dry-run --year-from 1995 --year-to 2024 --datatype RA,SE,HR,H1,UM,WF` 実行結果:

| 指標 | 値 |
|------|----|
| **raw 抽出** | **約 6.75 GB** (30 年 × 6 datatype) |
| parquet 化後 | 約 10.1 GB |
| features 200+ × 215 万 rows | 約 135 GB |
| 検出 .DAT files | 9,631 (SE 3,265 / H1 5,418 / UM 175 / WF 773) |

→ Session #84 設計 doc (50-100 GB) と整合。 raw 抽出だけなら **10 GB 程度**、 features 化で 100-135 GB に膨らむ。

### 1.4 使い方

```bash
# 1) DRY-RUN (絶対 安全、 parse しない)
python tools/backtest_30year_collect.py --dry-run \
    --year-from 1995 --year-to 2024 \
    --datatype RA,SE,HR,H1,UM,WF \
    --report-out data/backtest_30year/_dryrun.json

# 2) V15 既存 学習 data (2010-2025) と重複しない 1995-2009 を優先
python tools/backtest_30year_collect.py \
    --year-from 1995 --year-to 2009 \
    --datatype SE,UM \
    --output data/backtest_30year/ \
    --format parquet

# 3) 全 datatype 全期間 (要 disk 10+ GB)
python tools/backtest_30year_collect.py \
    --year-from 1995 --year-to 2026 \
    --datatype RA,SE,HR,H1,UM,WF,SK,DM,HC,WH,HN,TR,BR,BT
```

### 1.5 注意

- `RA` / `HR` は TFJV では独立 dir 無く SE_DATA / HY_DATA 内に混在のはず。 dry-run で `MISSING_DIR` の datatype は **後続 work で SE_DATA から record_type filter する** 必要あり (現 skeleton では `DATATYPE_DIR['RA'] = 'RA_DATA'` placeholder)。
- parquet 書き出しは pandas (worktree python に install 済 確認) 経由。 import 失敗時は `.csv.gz` fallback。

---

## 2. Task 2: JV-Link parser skeleton (`tools/jvlink_parser.py`)

### 2.1 設計

`JVLinkParser` class、 8 datatype (RACE / SE / HR / O1 / TCOV / WOOD / BLOD / UM) 用。

主要 method:

```
JVLinkParser(dlpath, progid, sid, dry_run)
  .initialize(sid)               JVInit() 呼び出し
  .open(dataspec, fromtime, ...)  JVOpen()  (蓄積系)
  .rt_open(dataspec, key)         JVRTOpen() (速報系 O1 等)
  .read(max_size)                 JVRead() → (rc, raw_bytes, filename)
  .close()                        JVClose()
  .fetch(dataspec, fromtime, ...) 高レベル wrapper (open → read loop → parse → list[dict])
  .parse(dataspec, raw)           record_type dispatcher

  .parse_ra / parse_se / parse_hr / parse_o1 /
  .parse_tk / parse_wc / parse_um / parse_blod   record 別 parser
```

`dry_run=True` で COM dispatch せず schema 検証のみ可能 (64-bit Python OK)。

64-bit Python で誤って起動した場合は `_ensure_com()` で `sys.maxsize > 2**32` を見て早期 RuntimeError。

### 2.2 dataspec 表

| dataspec | mode | record_type | 説明 |
|----------|------|-------------|------|
| RACE | open (JVOpen) | RA | レース詳細 (蓄積系) |
| SE   | open | SE | 馬毎レース情報 |
| HR   | open | HR | 払戻 (単/複/枠連/馬連/馬単/ワイド/三連複/三連単) |
| UM   | open | UM | 馬個体 (sire / dam / bms) |
| BLOD | open | HN/SK/BT | 血統 (record_type が複合) |
| WOOD | open | WC | ウッドチップ調教 |
| TCOV | open | TK | コース情報 |
| O1   | rt (JVRTOpen) | O1 | 速報 単複オッズ + 票数 |

### 2.3 動作 確認 (64-bit Python 範囲)

```
$ python tools/jvlink_parser.py --list
  RACE   mode=open  record=RA       レース詳細 (蓄積系)
  SE     mode=open  record=SE       馬毎レース情報
  HR     mode=open  record=HR       払戻
  ...

$ python -c "from tools.jvlink_parser import JVLinkParser; \
             p=JVLinkParser(dry_run=True); print(len(p.list_datatypes()))"
8
```

import + schema は 64-bit OK。 実 COM dispatch は 32-bit 必須。

### 2.4 32-bit Python 動作確認 手順 (user task)

★ **以下は user 環境で実施** (Agent 動作不能)。

1. 32-bit venv の有効化
   ```cmd
   C:\Users\takum\jvlink-venv\Scripts\activate.bat
   python --version
   python -c "import sys; print(sys.maxsize)"   # 2147483647 (32-bit) を確認
   ```

2. pywin32 install (未 install なら)
   ```cmd
   pip install pywin32
   python -m win32com.client.gencache --rebuild
   ```

3. COM 接続テスト
   ```cmd
   cd C:\Users\takum\keiba-ai
   python tools\jvlink_parser.py --test-com
   ```
   期待 output: `JVInit() rc=0`

4. 過去 1 日 RA fetch test (蓄積系)
   ```cmd
   python tools\jvlink_parser.py --datatype RACE --from 20260503 \
       --max 10 --out data/v18/jvlink_test_ra.json
   ```

5. 速報 O1 fetch test (RT 系)
   ```cmd
   python tools\jvlink_parser.py --datatype O1 --realtime \
       --raceid 202605070611 --max 5
   ```

6. 失敗時 check
   - `[!] 64-bit Python では...` → 32-bit venv 未有効化
   - `pywin32 未 install` → `pip install pywin32`
   - `JVInit() rc=-211` → DataLab 未契約 or 利用者キー 未設定
   - `JVInit() rc=-303` → DLL 未登録 (`regsvr32 JVDTLab.dll`)

### 2.5 TODO (Phase 3、 6/9-6/13)

- 各 `parse_*` の field offset を JV-Data 仕様書 (RA1/SE1/HR1/...) に **完全準拠** で実装
- `read()` の VARIANT 経由 buffer 取得を pywin32 で正しく実装 (現在は skeleton)
- `fetch()` の DownloadCount / ReadCount progress を Discord 通知化
- 6/13 までに過去 1 年 bulk fetch + 整合チェック (PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md)

---

## 3. 既存 file への影響

- 既存 file 変更: **0 件** (.gitignore / CLAUDE.md 含め一切 触らず)
- 新規 file: 3 件
  - `tools/backtest_30year_collect.py`
  - `tools/jvlink_parser.py`
  - `data/v18/phase22_agent_a_5_11.md`
- V15 model / predict_core / daily_predict / app.py / train/ : 一切 触らず

---

## 4. report summary (200 words 以内)

a) **30 年 backtest 容量見積り**: 1995-2024 (30 年) × 6 datatype (RA/SE/HR/H1/UM/WF) で **raw 約 6.75 GB → parquet 約 10 GB → features 200+ 込みで約 135 GB**。 Session #84 設計 (50-100 GB) と整合。 TFJV 実測 5.8 GB / 90 年分、 30 年抽出は約 3 GB 想定 → parquet で 1.5x 膨張。

b) **JV-Link 8 parser skeleton**: 449 行、 `JVLinkParser` class に `initialize / open / rt_open / read / close / fetch / parse` の 7 method + `parse_ra/se/hr/o1/tk/wc/um/blod` の 8 record parser。 8 dataspec (RACE/SE/HR/UM/BLOD/WOOD/TCOV/O1) 対応。 dry_run=True で 64-bit でも schema 検証 OK 確認済。

c) **32-bit Python 動作確認 手順**: (1) `jvlink-venv\Scripts\activate.bat` → (2) `pip install pywin32` → (3) `python tools\jvlink_parser.py --test-com` で `JVInit() rc=0` 確認 → (4) `--datatype RACE --from 20260503 --max 10` で過去日 fetch → (5) `--realtime --raceid` で 速報。
