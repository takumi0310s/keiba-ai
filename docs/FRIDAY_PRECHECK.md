# 金曜 前日チェックリスト (FRIDAY_PRECHECK)

> **使い方**: 毎週金曜「前日チェック」時にこの1枚で漏れなく点検する。
> 各項目に【確認コマンド】【正常の基準】【異常時の対処】を明記。
> ★ 自動Discord通知は増やさない。結果は Claude Code 出力で判断する ★
> 最終更新: 2026-05-30 (paci/ZE "静かな停止" 事件の教訓を反映)

PYEXE = `C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe`
作業は **読み取り中心**。修復は **安全手順(bare parse直叩き禁止)** のみ。本番プロセス無干渉。

---

## ⓪ 最優先サマリ (まずこれ)
1コマンドでデータ鮮度を一括確認（明日の開催日を引数に）:
```
python tools/data_freshness_monitor.py --date <土曜YYYYMMDD> --no-notify
```
→ PACI 100% / mtime 全OK / 「全ソース閾値クリア✅」 なら **データは健全**。
  WARN が出たら下記 ■1 の対処へ。`--no-notify` で Discord に出さず手元で確認。

---

## ■1. データ鮮度 ★最重要 (paci/ZE事件の教訓)★

### 1-1. 高gainソースの mtime (静かな停止検知)
V15予測力の**約65%**を占める2大ソースが「ファイルが進まず静かに停止」していないか。
- **確認**: `python tools/data_freshness_monitor.py --no-notify` の「mtime鮮度」節
- **正常**: paci(gain52.6%)/ze(gain12.7%)/kyi/oz/kta が **9日以内に更新**（直近開催で更新）
- **異常時**:
  - paci/ze が古い → `python tools/daily_paci_refresh.py` を実行（PACIパックDL→jrdb_paci.csv+jrdb_ze.csv 安全再生成）
  - ★ **絶対に bare `parse_jrdb.py` を直叩きしない** ★ — kab/cyb を破壊する（cyb 395→551,542行に暴走した実績）。必ず `daily_paci_refresh.py`（backup→parse→restore内蔵）経由

### 1-2. 当日カバレッジ (paci が当日race_idを持つか)
- **確認**: 同上モニタの「当日race_id」「jrdb_paci 当日 XX%」
- **正常**: **PACI 当日 ≥80%**（理想100%）。kyi も100%付近
- **異常時**: PACI<80% → 当日paciが入っていない → `daily_paci_refresh.py` 実行 → 再度モニタで100%確認
- 補足: sed/tyb が当日0%は**正常**（前走/結果・直前データで当日race_idを持たない設計）

### 1-3. 全データソースの最終更新日 (4月で止まっていないか)
- **確認**:
```
python -c "import os,glob,time; [print(f'{os.path.basename(f):40s} {(time.time()-os.path.getmtime(f))/86400:5.1f}d') for f in sorted(glob.glob('data/jrdb_*.csv'))+sorted(glob.glob('data/netkeiba_*.csv'))]"
```
- **正常**: V15が使う**当日系**(kyi/oz/jo/cha/joa/sed/tyb/cyb/kab/kta/sr/kka/paci/ze + netkeiba thisweek/speed/stable/training)が直近開催で更新
- **異常時**: 当日系が40日以上古い → そのソースの取得経路を確認（paciと同型の「日次パイプライン漏れ」の疑い）。daily_jrdb_kyi.bat / daily_premium_scrape.py の取得対象に含まれるか確認
- 補足: 静的参照(blood_full / training_times累積 / feature_lookups / *_v2 / skb=leak未使用 / zk)は古くても**正常**(gain<0.6%か未使用)

---

## ■2. スケジューラ

### 2-1. 全本番タスクが翌土日に発火設定か (Next Run Time)
- **確認**:
```
powershell -Command "Get-ScheduledTask | ?{$_.TaskName -match 'DailyPredict|RaceAutoNotify|DailyResults|SaveAllHorse|PaciDailyRefresh|DataFreshness|DailyJrdbKyi|DailyPremium|MorningWeight'} | %{$i=$_|Get-ScheduledTaskInfo; '{0,-30} Next={1} LastRes={2}' -f $_.TaskName,$i.NextRunTime,$i.LastTaskResult}"
```
- **正常**: 各タスクの Next が翌土曜/日曜。朝順序:
  `DailyPremiumScrape 3:00 → DailyJrdbKyi 6:00 → PaciDailyRefresh 6:50 → DataFreshnessMonitor 7:30 → DailyPredict 8:00 → RaceAutoNotify 8:45 → SaveAllHorseScores 9:00 → MorningWeightCheck 9:30 → DailyResults 18:00`
- **異常時**: Next が空欄 = 発火しない（`*_5_9` 等の一回限りタスクは無視可）。本番タスクが空欄なら schtasks 再登録

### 2-2. 先週 異常終了したタスクの良性判定
- **確認**: 上記の LastRes（前週の結果コード）
- **良性と判定してよいもの**（再調査不要）:
  - `RaceAutoNotify Res=3221225786 (0xC000013A)` = 全R正常完了後の終了。`logs/race_auto_notify_YYYYMMDD.log` 末尾に `All races processed. Exiting.` があれば良性
  - `DailyPredict Res=1` = 非開催日「レース見つからず」の正常exit（`logs/daily_predict_watchdog_*_subproc.log` で確認）
- **真の異常**: `logs/discord_failures.log` の更新（送信失敗）/ ログに traceback / 全R処理されず途中終了
- **対処**: ログ精読 → 良性なら GO、真異常なら原因修正

### 2-3. per-race通知プロセスの起動方式 (今朝の死亡対策)
- **背景**: `RaceAutoNotify` をタスク経由(可視コンソール)で起動すると **Ctrl+C/Closeイベントで途中死亡** → per-raceタイマー全消滅（2026-05-30 に9:00死亡を経験）
- **確認**: 土曜稼働中に `powershell -Command "(Get-ScheduledTask -TaskName RaceAutoNotify_Sat -TaskPath '\keiba-ai\').State"` が `Running` か / ログ末尾に `^C` がないか
- **異常時(死亡)**: タスク再起動でなく **デタッチ起動**（コンソール免疫）で復活:
```
python -c "import subprocess,os; env=os.environ.copy(); env.update({'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8','SCRAPER_GUARD_DISABLE':'1','KEIBA_OPERATIONAL_MODE':'1','FOR_DISABLE_CONSOLE_CTRL_HANDLER':'1','KMP_DUPLICATE_LIB_OK':'TRUE'}); log=open(r'logs/race_auto_notify_<YYYYMMDD>.log','a',encoding='utf-8',errors='replace'); subprocess.Popen([r'<PYEXE>','-u','tools/race_auto_notify.py'],cwd=r'C:\Users\takum\keiba-ai',stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)"
```
（起動前に多重起動防止で既存プロセス確認。10:30以降の起動は朝バッチ再送スキップ=重複なし）
- 補足: process_watchdog_v2 は kill-switch(`data/v18/process_watchdog_v2.kill`)で無効中＝自動復旧しない。恒久対策は要承認

---

## ■3. モデル・環境

### 3-1. V15/V16 ロード可能か
- **確認**:
```
python -c "import gzip,pickle; [print(f, 'OK' if pickle.load(gzip.open(f,'rb')) else 'NG') for f in ['keiba_model_v15_central_live.pkl.gz','keiba_model_v15_central.pkl.gz','models/v16_ability_candidate.pkl.gz']]"
```
- **正常**: 3ファイルとも例外なくロード。V15 booster=145 features
- **異常時**: ロード失敗 → ファイル破損/欠落。git/バックアップから復元（★上書きはしない★）

### 3-2. python実体・ライブラリ
- **確認**: `python -c "import sys;print(sys.executable)"` / `python -c "import lightgbm,xgboost,pandas,numpy,sklearn;print('libs OK')"`
- **正常**: `sys.executable` が `...pythoncore-3.14-64\python.exe`（WindowsApps stub でない）。全lib import可
- **異常時**: stub を指す → PATH/エイリアス確認。lib import失敗 → 該当pip再確認

### 3-3. ディスク容量
- **確認**: `powershell -Command "Get-PSDrive C | Select-Object @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}"`
- **正常**: 空き ≥ 10GB（JRDB/予測ログ/HTML生成の余裕）
- **異常時**: 逼迫 → 古いログ/一時ファイル整理（data/_tmp* 等、本番データは消さない）

---

## ■4. 通知・出力

### 4-1. Discord webhook 生存
- **確認**: `.env` に `DISCORD_WEBHOOK_BETS / _UPDATES / _URL` が SET（https://で始まる）
- **正常**: 3つ設定済。必要なら #アップデートにテスト1通（`send_discord(..., channel='updates')`）
- **異常時**: 未設定/失効 → webhook再発行して .env 更新

### 4-2. 全馬HTML連携
- **確認**: `data/allscores/` ディレクトリ存在（中身は土曜朝に生成=金曜は空が正常）。`tools/build_allscores_html.py` `build_allscores_txt.py` compile可
- **正常**: 朝(daily_predict 8:00)=予測版HTML→#買い目 / 夕(daily_results 18:00)=結果版HTML→#買い目 の2回送信が配線済
- **異常時**: 送信失敗ログ → notify.send_discord_file / webhook確認

### 4-3. Discord送信失敗ログ
- **確認**: `logs/discord_failures.log` の最終更新日
- **正常**: 直近の race weekend に新規失敗エントリなし
- **異常時**: 最近の失敗 → 該当時刻のタスクログ精読

---

## ■5. 戦略・投票設定

### 5-1. 現行戦略フラグ
- **確認**: `tools/race_auto_notify.py` で以下が想定通りか（grep確認）
- **正常**:
  - 京都フィルタ(案C・平場除外) / STRATEGY_C4_ENABLED=True(条件A 1600-1800m) / 条件X除外(案C) / 条件E・B除外
  - STRATEGY_TOKYO_PAPER_ONLY=True、B1/B2/C1/C2/C3 = PAPER_ONLY(paper蓄積のみ・投票に影響せず)
  - 障害(surface=='障')は全段階除外
  - 買い/見送り判定 = `strategy_filters.evaluate_bet_decision`(従来フィルタ＝単一真実源)
- **異常時**: フラグが意図と違う → 変更経緯をgit logで確認（★勝手にロジック変更しない★）

### 5-2. Kelly投資額
- **確認**: `tools/kelly_criterion.py` の min_bet/max_bet
- **正常**: ¥300–¥700（quarter-Kelly、rolling ROI調整、fallback ¥700）
- **異常時**: 範囲逸脱 → 設定確認

### 5-3. 買い目ロジック
- **正常**: 三連複=TOP1軸×[TOP2,3]×[TOP2-6]の**上位6頭7点** / 条件E=馬連 上位3頭2点
- 確認は読み取りのみ（`tools/predict_core.py generate_trio_bets`）

---

## ■6. 先週の振り返り

### 6-1. 先週ログのエラー/異常
- **確認**:
```
powershell -Command "Get-ChildItem logs | ?{$_.LastWriteTime -gt (Get-Date).AddDays(-7)} | Sort LastWriteTime -desc | Select Name,LastWriteTime -First 20"
```
  + 主要ログ(daily_predict subproc / race_auto_notify / daily_results)に `Traceback` / `ERROR` / `失敗` がないか
- **正常**: 致命的traceback なし。データ取得の部分失敗(SED一部欠落等)は許容
- **異常時**: 繰り返すエラー → 原因調査（例: Morning_Sat Res=255 が継続=morning_top_races.sh 要調査だが非コア）

### 6-2. 先週の結果照合・ROI
- **確認**: `data/daily_results/<先週土日>.csv` 生成、3区分(実投票/見送りシャドー/全部買い)サマリーが出ているか
- **正常**: 結果照合完了・cumulative更新済
- **異常時**: 未照合 → `python tools/daily_results.py --date <先週日付>`（再照合は安全）

---

## 📊 チェック完了の総合判定
- 全項目「正常」 → **GO（明日の自動運用OK）**
- ■1(データ鮮度)で WARN → **最優先で修復**（daily_paci_refresh.py）してから GO
- ■2-3(per-race起動)は土曜当日に再確認（金曜は設定確認のみ）
- 真の異常が1つでも未解決 → **修正するまで保留**

---

## 🔧 (任意) 一括点検スクリプトの提案 (実装は承認後)

**提案**: `tools/friday_precheck.py` — このリストの機械化可能な項目を1コマンドで点検し、結果を **Claude Code出力(標準出力)** にまとめて表示（★Discord通知は出さない★）。

点検内容(案):
1. データ鮮度: `data_freshness_monitor.py --no-notify` を内部呼び出し（PACI/ZE/mtime）
2. スケジューラ: 本番タスクの Next Run + 先週 LastResult（良性コード 0xC000013A/1 は自動で「良性」表示）
3. モデル: V15/V16 ロード可否 + lib import + python実体パス
4. webhook: .env の3 webhook SET確認（送信はしない）
5. ディスク: C: 空き容量
6. 先週ログ: 直近7日のログに Traceback/ERROR 件数

出力形式: 各項目を `✅OK / ⚠️WARN(対処) / 🔴NG` で色分け表示し、末尾に「総合: GO / 要対応N件」。
- 引数: `--date <土曜YYYYMMDD>`（鮮度チェックの対象日）
- **通知なし**: 結果は標準出力のみ。れんはすが出力を見て判断
- 安全: 全て読み取り（修復は行わず「実行すべきコマンド」を提示するだけ）

→ 作成可否はご判断ください（実装は別途・承認後）。本リスト(手動)だけでも完全に点検可能です。

---

## 付録: 今シーズンの教訓 (なぜこの項目があるか)
| 事件 | 教訓 → チェック項目 |
|------|------|
| paci 4/4 静かに停止 (gain52.6%、1ヶ月気づかず) | ■1-1 mtime監視 / ■1-2 当日カバレッジ |
| ZE 5/1 静かに停止 (gain12.7%、日次取得対象外) | ■1-1 ze も監視対象 |
| bare parse_jrdb.py が kab/cyb 破壊 | ■1-1 daily_paci_refresh.py 経由必須 |
| race_auto_notify 9:00 コンソールkill | ■2-3 デタッチ起動 |
| 終了コード 0xC000013A/Res=1 の誤検知 | ■2-2 良性判定基準 |
