# Phase 22 Agent C 実調査結果 (2026-05-11)

## 目的
1. jra_payouts.csv 4/6 停止 の 根本原因 究明 + 復活 path
2. アメダス 1 分粒度 取得 の 公的 source 調査
3. JRA 公式 入線写真 の URL pattern 確定 + DL skeleton

★ V15 model 関連 file は 一切 触らず、 新規 file のみ 追加 ★

---

## Task 1: jra_payouts.csv 復活 (4/6 停止 → 原因確定)

### 1-a) 4/6 停止 の 根本原因
JRA は **JRADB CGI 系 (post-back navigation flow)** を 完全 廃止 した。

| 旧 URL | 現状 (2026-05-11 確認) |
|--------|--------|
| `https://www.jra.go.jp/JRADB/accessS.html` | 301 → `/error/error013.html` |
| `https://www.jra.go.jp/JRADB/accessO.html` | 301 → `/error/error013.html` |
| `https://www.jra.go.jp/JRADB/accessD.html` | 301 → `/error/error013.html` |
| `https://www.jra.go.jp/JRADB/accessJ.html` | 301 → `/error/error013.html` ★ |

★ `scrape_jra_track.py` も `accessJ.html` を 使用しているため 同時 影響。
   ただし 馬場情報 は `_data_cushion.html` / `_data_moist.html` から
   別 source 経由 で 取れている。

旧 scraper (`scrape_jra_payouts.py`) は `pw01skl10` (calendar) → `pw01srl10`
(開催日) → `pw01ses10` (全レース一覧) という **POST + CNAME chain** で
払戻 を 取っていた。 全 entry point が 死んだ 結果、 4/6 以降 全 失敗。

### 1-b) 新 path (DRY-RUN 確認済)
JRA は 静的 HTML 配信 に 移行:

```
/datafile/seiseki/replay/{YEAR}/g1.html   ← 年別 G1 index
/datafile/seiseki/g1/{race}/result/{race}{YEAR}.html  ← 個別 G1 結果
```

`<dl>` 構造 で `単勝 / 複勝 / 枠連 / 馬連 / 馬単 / ワイド / 3連複 / 3連単`
を 取得可能。 ★ **encoding は Shift_JIS** ★ (utf-8 だと 文字化け)。

### 1-c) DRY-RUN 結果 (2025 G1 feb2025)
```
旧 URL 3 件 全て 301 → error013 (停止確定)
新 G1 index: 26 race 発見
sample (feb2025) 払戻 parse OK:
  tansho: [430]
  fukusho: [170, 210, 150]
  wakuren: [820]
  umaren: [1830]
  wide: [720, 420, 440]
  trio: [2140]
  tierce: [13510]
```

### 1-d) 復旧 path (今後)
- **G1 (26 race/year)**: 新 path で 直 取得可、 即 復旧可能
- **平場 (~3,400 race/year)**: JRA 公式 では **公開されて いない 模様**
  → 推奨 source:
    - **JV-Link HR (払戻 DB)** ★ 最優先 (5/24+ DataLab 加入後)
    - netkeiba result page (既存 cookie path)
    - TFJV HR_DATA (43,000 files 内に payout 含)

### 1-e) 新規 file
- `tools/scrape_jra_payouts_v2.py` (DRY-RUN 専用、 既存 file 改変 なし)

---

## Task 2: アメダス 1 分粒度

### 2-a) 結論: **JMA は 公的 1 分 endpoint を 公開していない**
気象庁 公式 (`/bosai/amedas/`) は **10 分 が 最細**。 系統 probe で 確認:

| 候補 endpoint | 結果 |
|---------------|------|
| `/bosai/amedas/data/point/{stno}/YYYYMMDDHHMM.json` | 404 |
| `/bosai/amedas_h1m/data/point/{stno}/YYYYMMDD_HH.json` | 404 |
| `/bosai/amedas_1min/data/point/{stno}/YYYYMMDD_HH.json` | 404 |
| `/bosai/amedas/data/point_1min/{stno}/YYYYMMDD_HH.json` | 404 |
| `/bosai/amedas/data/map_1min/YYYYMMDDHHMMSS.json` | 404 |

裏付け:
- `data.jma.go.jp/obd/stats/etrn/` で 明示: `１０分ごとの値を表示`
- map JSON も `sun10m / precipitation10m` フィールド名 = 10 分集計
- 1 分 配信 は 気象庁 有償サービス (AMeDAS リアルタイム配信) のみ。
  個人 公開 API では 提供 なし。

### 2-b) DRY-RUN 結果 (東京、 府中 44132、 2026-05-11 09:00)
```
公式 10 分 endpoint: 8 keys 取得 OK
1 分 endpoint 5 候補: 全 404
→ 10 分 → 1 分 補間 (source="jma_bosai_10min",
                     granularity_real="10min_interpolated_to_1min")
80 rows expanded from 8 10-min records
sample temp=20.6℃ humidity=64% wind=2.8m/s
```

### 2-c) 採用 path
- 公的 source で 最善: **10 分 粒度 を fetch + 1 分 へ 等間隔複製**
- 降水量 は `precipitation10m / 10` で 1 分換算 (粗近似、 真の 1 分 ではない)
- source 列 明示 で 透明性 確保 (将来 1 分 source 発見時 差し替え可)

### 2-d) 新規 file
- `tools/scrape_amedas_1min.py` (DRY-RUN + 1 日 fetch & CSV 保存)

---

## Task 3: JRA 公式 入線写真

### 3-a) URL pattern (確定)
G1 result page に **`/result/photo/` 配下** で 入線写真 (jpg) を 公開。

```
page: /datafile/seiseki/g1/{race}/result/{race}{YEAR}.html
photo: /datafile/seiseki/g1/{race}/result/photo/{YEAR}-{N}.jpg
        N = 1, 2, 3, 4 (通常 4 枚 = ゴール直前 / 入線 / 1着 / シーン)
```

### 3-b) DRY-RUN 結果 (2025 G1 feb2025)
```
G1 index: 26 race 発見
sample (feb2025) photo URL 抽出: 4 件
  https://www.jra.go.jp/datafile/seiseki/g1/feb/result/photo/2025-1.jpg
  https://www.jra.go.jp/datafile/seiseki/g1/feb/result/photo/2025-2.jpg
  https://www.jra.go.jp/datafile/seiseki/g1/feb/result/photo/2025-3.jpg
  https://www.jra.go.jp/datafile/seiseki/g1/feb/result/photo/2025-4.jpg
DL skeleton: data/jra_finish_photos/feb2025/2025-1.jpg (未 DL)
```

### 3-c) 制限事項
- **G1 のみ** (年 ~26 race)、 平場 は 公式 公開 なし
- 平場 入線写真 候補: netkeiba result page (要 確認)、 JRA-VAN ネクスト
- 個人 利用 範囲 OK、 再 distribute は 規約 確認 必要

### 3-d) 新規 file
- `tools/scrape_jra_finish_photos.py` (DRY-RUN + 実 DL 切替)
- `.gitignore` 追加: `data/jra_finish_photos/`

---

## まとめ

| Task | 結論 | 即 実装可? |
|------|------|----------|
| 1. jra_payouts | 旧 JRADB CGI 完全 廃止、 新 path G1 のみ 取得可 | G1: ◎ / 平場: × → JV-Link 待ち |
| 2. amedas 1分 | 公的 1 分 endpoint 不存在、 10分→1分 補間 が最善 | △ (補間で代用) |
| 3. finish photo | URL pattern 確定、 G1 26 race/year | G1 ◎ / 平場 × |

### 新規 file (本 commit)
```
tools/scrape_jra_payouts_v2.py
tools/scrape_amedas_1min.py
tools/scrape_jra_finish_photos.py
data/v18/phase22_agent_c_5_11.md  (本 file)
.gitignore (data/jra_finish_photos/ + data/amedas_1min_*.csv 追加)
```

### 動作 check
- `python -c "import py_compile; ..."` 全 SYNTAX OK
- 3 script 共 dry-run 成功 (本 doc 内 結果)

### V15 投資保護
- predict_core.py / daily_predict.py / app.py / train/ 一切 触らず
- 既存 scrape_jra_payouts.py 改変 なし (並行 v2 ファイル)
- 既存 scrape_weather.py 改変 なし (並行 amedas_1min ファイル)
