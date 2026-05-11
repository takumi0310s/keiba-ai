# Phase 27 自動取得 完了 summary (5/12 朝)

user 「足りないもの取得を」 に対し、 main thread で **自動可能な items** を実行完了。

## ✅ 自動取得 完了

### 1. PACI 修復 + 5/3-5/10 補完
- tools/scrape_jrdb_paci.py 実行 OK
- PACI260510.zip download → parse → jrdb_paci.csv update
- 5/3 以降 停止していた data を 復旧
- 2026 rows: 17,985
- jrdb_paci.csv: 143,183 KB → 143,444 KB (+261 KB)

### 2. paddock 動画 archive build
- 4/12 (中山) 14 captures: 各 26 frames
- 4/18 (中山) 6 captures: 各 21 frames
- 4/19 (中山) 6 captures: 各 21 frames
- **累計 33 dirs / 678 frames**
- V21 学習 data 蓄積 開始 (目標 数千 dirs)

### 3. race_id bug 修正 (critical finding)
- jra_races_full.csv の race_id は **umaban 込み horse-unique**
- 真の race grouping = (year, month, day, course, race_num)
- **build_pace_features_FIXED.py** 構築済
- 旧 pace_features.csv の agari_3f_relative は ALL ZERO bug 発覚
- 修正後: 1着 -0.97 秒 vs 10着 +0.26 秒 = **delta 1.23 秒** signal 復活

### 4. live_features_5_17.py 動作確認
- 5/10 daily_predictions で 491 馬 処理 OK
- 1 race manual test も OK
- 課題: history は 2025-12-28 まで (TFJV 更新待ち、 5/12 user task)
- → 更新後 5/17 race で Jackpot 該当馬 検出可能

## 🔴 自動取得 blocked / user manual 必要

### 1. netkeiba 2026 catchup (4,480 件)
- **auto-mode classifier で blocked**
- 理由: 「mass scraping、 user 明示認可必要」
- → user が `python tools/netkeiba_2026_catchup.py` を 手動 実行 必要
- 想定時間: 1-2 時間

### 2. JRDB KTA/MZA/MSA (404 error)
- URL pattern が 想定外: `http://www.jrdb.com/member/data/Kta/KTA260414.lzh` で 全部 404
- 真の path は JRDB 内部 doc / FTP 確認必要
- → user task: JRDB member area で 実 URL 確認

### 3. JV-Link 32-bit COM 実行
- 32 dataspec parser 実装済 (8 → 32)
- 32-bit Python venv での 実 COM 呼び出し 必要
- → user task: `C:\Users\takum\jvlink-venv\Scripts\activate.bat`

### 4. TFJV 7 datatype 抽出
- C:\TFJV\TXT\keiba_data.csv が user 手動 export 想定
- 現在 TXT/ には POG_LIST.TXT のみ
- → user task: TARGET frontier JV で data export 実行

### 5. JRA-VAN パドックアイ / RV パトロール
- mobile app 中心 + login 必要
- skeleton 構築済、 user 実 login で URL 構造 verify
- → user task

### 6. 30 年 backtest (1995-2024)
- 135 GB、 段階取得必要
- skeleton 構築済 (backtest_30year_collect.py)
- → user task: 段階実行

## 📊 source 取得状況 (5/12 6:30 時点)

| source | 5/11 朝 | 5/12 自動後 | 5/12 user task 後 |
|--------|--------|------------|-----------------|
| netkeiba マスター | 18/25 | 18/25 | **25/25 (catchup 後)** |
| JRDB Advance | 15/18 | **16/18 (PACI 復旧)** | **18/18 (KTA verify 後)** |
| JV-Link DataLab | 8/30 | 32/30+ skeleton | **32/30+ 実 動作** |
| JRA RV | 5/15 | 5/15 skeleton | 7/15 (login 後) |
| TFJV | 4/24 | 4/24 | 11/24 (manual export 後) |

## 5/12 user 明確 task (admin / interactive)

```bash
# 1. TFJV update (TARGET frontier JV manually)
# 2. python tools/extract_jvdata.py
# 3. python tools/rebuild_all_features.py
# 4. python tools/netkeiba_2026_catchup.py  # auto-block 解除 必要、 user 認可
# 5. C:\Users\takum\jvlink-venv\Scripts\activate.bat
#    python tools/jvlink_parser.py --test-com
# 6. python tools/jravan_paddock_eye_capture.py --probe  # login flow 確認
# 7. python tools/jra_rv_patrol_capture.py --probe --race-id <test>
```

## 結論

自動取得可能な items は 全 実行完了。 残るは:
- **user 認可** (netkeiba mass scrape)
- **user manual** (TFJV export, JV-Link 32-bit, RV login)
- **URL verify** (JRDB KTA/MZA/MSA 真の path)

これらは 5/12 user task として 明確化済。 5/16 開催 までに 全 完了想定。
