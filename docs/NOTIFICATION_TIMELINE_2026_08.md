# 通知タイムライン全量調査 (2026-08-15)

読み取りのみ・変更なし。8/15(土) 実ログ + タスクスケジューラ実設定 + スクリプト実コードから作成。
docs/NOTIFICATION_INVENTORY_2026.md (8/11) の後継。**インベントリとの食い違い2点を B-差分に明記**。

- ch 凡例: **bets**=#買い目 / **updates**=#アップデート / **TEST1**=検証専用ch (BETS/UPDATES非混入)
- Phase 列: **P1**=8/17 phase1 で停止 / **P2**=8/21 phase2 で停止 or 置換 / **残存**=現計画では触らない / **停止中**=既に無音
- 「計画なし★」= phase1/2 のどちらにも入っておらず、今回の取捨判断の対象

---

## ★ 8/15 確定版の取捨 (ユーザー承認・同日実装済み) ★

通知層のみの変更。予測ロジック・データ・モデルは不変。

| 対象 | 取捨 | 実装 (8/15) |
|------|------|------------|
| Stage2 通知 33通/日 (PreRacePredict_Watchdog_5_9) | **即時停止** | タスク disable 済み (SUCCESS) |
| Morning_Sat/Sun 06:30 先行買い目 8通+HTML | **即時停止** | タスク disable 済み (SUCCESS)。paper 記録は daily_predict 側で継続 |
| 恒常CRITICAL誤報 3通 (PreFire/AM3/AM6) | **即時停止** (Phase1 前倒し) | タスク disable 済み (SUCCESS) |
| DailyPredict 08:00 の整形買い目 8通 (bets) | **停止** (#買い目 は 09:30 朝一括に一本化 — 8/11 設計の取り残し) | `daily_predict.py` の notify_formatted 呼び出しを `if False:` ゲート (コメント付き・1行で復活可) |
| ROI Monitor の DANGER/WARNING 通知 (旧142.6%基準) | **9月ゲートまで通知オフ** (基準改定でなく停止を選択: 途中で非登録基準を発明すると事前登録ゲートの純度を汚す。資金保護は kill ライン+T1v2 が担当。console/roi_alerts.json 記録は継続) | `roi_monitor.py` に `DISCORD_NOTIFY_ENABLED = False` |
| RaceDayReport_Sat/Sun 18:00 (恒常不発・daily_results と重複) | **停止** | タスク disable は Access denied → `race_day_report.bat` スタブ化 (バックアップ `race_day_report.bat.bak_20260815`)。正式 disable は `tools/disable_legacy_audit_admin.bat` に追記済み (要管理者実行) |
| 09:30 朝一括 / 18:00 daily_results / TYB公開検出 / T1v2 の真の CRITICAL/WARN / 月曜 WeeklyReport / s2b TEST1 | **残す** | 変更なし |
| DailyPredict 08:00 の全馬スコアHTML添付 1通 (bets) | **残置** (朝一括に無い独自コンテンツ・1通のみ。不要なら次回1行で停止) | 変更なし |
| Phase1/2 との整合 | AM3/AM6 を `apply_phase1_20260817.bat` と `disable_legacy_audit_admin.bat` から消し込み (二重適用防止)。PreFire/Watchdog/Morning/RaceDayReport は元々 phase リスト外 | 済み |

**research→本番 経路の横断チェック結果 (停止前確認・タスク全数)**: `if exist` 条件実行パターンは `pre_race_predict_runner.bat` の1件のみ。branch 専用 (main 不在) スクリプトがタスクから実行可能なのは ①`tools/stage2_predict.py` (今回停止) ②`tools/realtime_5_9.py` (`realtime_5_9_runner.bat` 経由で 5/9 系7タスクに配線されているが、全タスク Next Run N/A=期限切れで**休眠**・発火経路なし) の2件。8/11-12 再建の新設 bat/py 群 (supply/t1v2/batch_notify/probe/3パス) も main 不在だが意図的な新インフラ。research/ ディレクトリへの実行参照はゼロ (コメント2件のみ)。

---

## A. 開催日 (土日) 設計タイムライン

| 時刻 | 送信元 (タスク → スクリプト) | ch | 内容 | 頻度 | 分類 | Phase |
|------|------------------------------|----|------|------|------|-------|
| 02:55 | Keiba-PreFireCheck → pre_fire_check | updates | 環境チェック結果 (現在は premium タスク Disabled 検知で**恒常CRITICAL誤報**) | 1回 | 成功報告系(誤報化) | **計画なし★** |
| 03:15 | Keiba-AM3FireCheck | updates | premium scrape 発火確認 (スタブ化済みで**恒常CRITICAL誤報**) | 1回 | レガシー誤報 | P1停止 |
| 06:15 | Keiba-AM6FireCheck | updates | DailyJrdbKyi 発火確認 (disable済みで**恒常CRITICAL誤報**) | 1回 | レガシー誤報 | P1停止 |
| 06:30 | Keiba-Morning_Sat/Sun → **morning_top_races.bat (v17パイプライン)** | **bets**+updates | JRDB取得後に daily_predict を内部実行 → **買い目8通 + paper shadow stats 1通** (08:00と同内容の先行送信=重複) | 8+1通 | **レガシー重複★** | **計画なし★** |
| 07:00 | Keiba-MorningDigest → morning_dashboard | updates | 朝ダッシュボード | 1回 | 成功報告系 | P1停止 |
| 07:30 | DataFreshnessMonitor | updates | 鮮度警告 (OK時は無送信) | 異常時のみ | 異常時 | P1停止 |
| 07:30 | JrdbHealthCheck_Sat/Sun | updates | JRDB 6ファイル健全性 (正常時は無送信=ログのみ確認) | 異常時のみ(推定) | 異常時 | 残存 |
| 08:00 | DailyPredict → daily_predict.py | **bets**+updates | **整形済み買い目8通(bets)** + paper shadow stats(updates推定) + 取り漏れ警告(updates)。T1v2 BLOCKフラグで停止 | 8+1通 | 本命(朝予測) | 残存 |
| 08:45 | RaceAutoNotify_Sat/Sun | bets | 5分前レース毎通知 — **bat 不在で完全無音** (8/11停止措置。「スタブ化」でなく実体は bat 消失) | 停止中 | レガシー | 停止中 (9月GO時に復活判断) |
| 08:50 | Keiba-AM8FireCheck | updates | DailyPredict 発火確認 | 1回 | 成功報告系 | P1停止 |
| 08:50 | T1v2Audit (土日=dump監査) | updates | CRITICAL時のみ通知 + BLOCK_NEXT.flag 生成 | 異常時のみ | **本命 (fail-closed)** | 残存 |
| 09:00 | PaperS2BPredict → paper_trade_s2b | TEST1 | s2b 全R embed (検証用) | 1バッチ(35R) | 検証用 | 残存 |
| 09:00-17:00 30分毎 | **Keiba-PreRacePredict_Watchdog_5_9 → pre_race_predict_runner.bat → stage2_predict.py** | **bets** | Stage2(発走1h前) 個別R予測+朝予測との差分。「学習用・投票非推奨」明記だが **#買い目にレース毎送信** | **レース毎 (8/15実績33通)** | **レガシー重複★** (名前はWatchdogだが実体は通知経路) | **計画なし★** |
| 09:30 | MorningBatchNotify → morning_batch_notify + feature_health_report | **bets**+updates | 朝一括 (サマリ1+買いembed) + 特徴量健全性1通 — **8/11再整備の中核** | 1バッチ | **本命 (中核)** | P2で v2 bat に置換(継続) |
| 09:30 | Keiba-MorningWeightCheck_Sat/Sun | updates | 馬体重再チェック結果 | 1回 | 成功報告系 | P2停止 |
| 10:00/14:50/15:45 | Keiba-MultiStagePredict×3 | updates | テスト予測 (paper eval) | 各1通 | 検証用 | 計画なし★ |
| 毎時:30 | Keiba-TybPublishMonitor | updates | TYB 公開の初検出時に1通 (それ以外は無送信) | 条件付き | 成功報告系 | P1停止 |
| 18:00 | DailyResults_Sat/Sun → daily_results.py | updates | 結果照合embed + 全馬照合HTML添付 + ROI Monitorアラート (**旧142.6%基準の誤DANGER既知**) | 2-3通 | 本命 | 残存 |
| 18:00 | Keiba-RaceDayReport_Sat/Sun | updates | 当日レポート — daily_results と**同刻起動で毎回「該当なし」=実質不発** | 条件付き(恒常不発) | レガシー(バグ気味) | 計画なし★ |
| 20:00 | DailyResultsEvening | updates | 残レース照合 + ROIアラート再送 | 1-2通 | 本命 | 残存 |
| 20:30 | PaperS2BResults | TEST1 | s2b 結果照合 | 1回 | 検証用 | 残存 |
| 20:30 | Keiba-RaceNotifyLogV2-Aggregator | updates | 8戦略集計 — 5分前ループ停止中は空データ | 条件付き(現状空) | レガシー | 計画なし★ |
| 20:45 | PerRaceCoverageCheck | updates | 供給カバレッジ監査 (8/12新設) | 異常時のみ(推定) | 本命監査 | 残存 |
| 21:00 | Keiba-DailyCumulativeAudit | updates | 台帳監査 (8/15 rc=1 = 何か警告中) | 異常時のみ | 監査 | 計画なし★ |
| 22:00 | Keiba-FeaturesIntegrityCheck | updates | 特徴量整合 (8/15 rc=1) | 異常時のみ | 監査 | 計画なし★ |
| 23:00 | Keiba-NightlySanity | updates | 夜間サニティ | 1回/異常時 | 監査 | 計画なし★ |
| 17:00 | anomaly_auto_detector (NarDailyPredict と同刻連鎖・実行元要確認) | updates | 異常検知 (critical時のみ送信・今日は warning=3 で無送信) | 異常時のみ | 異常時 | 計画なし★ |

**通知を送らない (ログのみ) タスク**: JrdbSupplyDaily 03:20 / JvlinkSupplyDaily 03:40 / ProbeSaturdayPublish 05:00 / JrdbSupplyWeekendAM 06:35 / PaciDailyRefresh 06:50 / Keiba-ScrapeProgress 07:00 / Keiba-JrdbRetryAm9 09:00 / SaveAllHorseScores 09:00 / watchdog_v2 (5分毎・kill-switch で no-op) / NAR 系 5 本 (13:00-21:30、Discord送信コードなし)

---

## B. 8/15(土) 実績 (実ログ突合・20:20 時点集計)

| 時刻 | 通知 | ch | 通数 | 備考 |
|------|------|----|------|------|
| 02:55 | PreFireCheck「CRITICAL: 失敗」 | updates | 1 | 誤報 (premium タスク Disabled 検知) |
| 03:15 | AM3「CRITICAL: AM3:00 失敗」 | updates | 1 | 恒常誤報 |
| 06:15 | AM6「CRITICAL: DailyJrdbKyi 失敗」 | updates | 1 | 恒常誤報 |
| 06:30-07:05 | Morning_Sat: 整形済み買い目 **8通** + paper shadow stats 1通 | **bets**+updates | 9 | ★重複1回目 (v17パイプラインが daily_predict を内部実行) |
| 07:00 | MorningDigest 朝ダッシュボード | updates | 1 | CRITICAL 1 表示 (=PreFireの誤報を転記) |
| 08:00 | DailyPredict: 整形済み買い目 **8通** + paper shadow stats 1通 | **bets**+updates | 9 | ★重複2回目 (06:30 と同系内容) |
| 08:50 | AM8「DailyPredict 正常発火」 | updates | 1 | |
| 08:50 | T1v2 dump監査 PASS (races=35) | — | 0 | 設計どおり無送信・BLOCKなし |
| 09:00 | PaperS2B 35R embed | TEST1 | 1バッチ | BETS非混入を確認 |
| 09:00-16:32 | **Stage2 個別R通知 33通** | **bets** | **33** | ★設計外 (下記差分1) |
| 09:30 | MorningBatchNotify: サマリ1+買い**24embed** (見送り11はサマリ内) | **bets** | 25 | 本命。8/11再整備どおり |
| 09:30 | 特徴量健全性レポート | updates | 1 | |
| 09:30 | MorningWeightCheck「対象なし」 | updates | 1 | 案B改候補 0 件 |
| 10:02/14:52/15:47 | MultiStagePredict テスト予測 | updates | 3 | 各1通 |
| 18:00 | daily_results: 結果照合(33R) embed + 全馬照合HTML添付 + ROIアラート10件 | updates | ~3 | ROI DANGER = 旧基準誤報継続 |
| 18:00 | RaceDayReport_Sat「該当なし」 | — | 0 | 同刻競合で恒常不発 |
| 19:30 | TybPublishMonitor「TYB FIRST PUBLISH」 | updates | 1 | TYB は 19:30 に公開 (probe: KAB は 05:00 公開) |
| 20:00 | daily_results: 残3R照合 + ROIアラート | updates | ~2 | |
| 20:20時点 未発火 | 20:30 PaperS2BResults / Aggregator、20:45 Coverage、21:00/22:00/23:00 監査、21:30 NAR | — | — | |

**チャンネル別合計 (概算)**: **#買い目 = 76通** (Morning_Sat 8+HTML1 + DailyPredict 8+HTML1 + 朝一括 25 + Stage2 33。後検証で 06:30/08:00 とも全馬スコアHTML添付も送信していたことを確認) — うち投票に使うのは 09:30 の 25 通のみ / #アップデート ≈ 15通 (うち恒常誤報3) / TEST1 = 35R embed

### 設計 (インベントリ 8/11) との差分

1. **★Stage2 33通/日が #買い目 に出ている (インベントリ未記載)**。`Keiba-PreRacePredict_Watchdog_5_9` (土日 09:00 起動・30分毎リピート) が `pre_race_predict_runner.bat` を実行。この bat は「main には stage2_predict.py が無いので no-op スタブ」という Session #77 設計だが、**research/ruiji ブランチには tools/stage2_predict.py が存在するため毎30分フル動作**し、`channel="bets"` で送信している。5分前通知 (RaceAutoNotify) を止めても「レース毎通知」が別経路で生きていた。
2. **★Morning_Sat の実体はインベントリと違う**。記載は「morning_go_check」だが実際は `morning_top_races.bat` (v17 11R/12R 自動パイプライン) で、JRDB 取得後に **daily_predict を内部実行して買い目8通を 08:00 より前に送信**。08:00 DailyPredict と二重。
3. RaceAutoNotify_Sat/Sun は「bat スタブ化」ではなく **bat ファイル自体が不在** (発火→即失敗で無音。結果は同じだが状態表記の訂正)。
4. RaceDayReport_Sat 18:00 は daily_results と同時刻起動のため**毎回「該当なし」で不発** (4月から全日不発をログで確認)。
5. ROI Monitor の DANGER (累積88.9%<100% 等10件) は既知の旧 142.6% 基準誤報のまま送信継続。

---

## C. 平日タイムライン

| 時刻 | 送信元 | ch | 内容 | 頻度 | 分類 | Phase |
|------|--------|----|------|------|------|-------|
| 02:55 | Keiba-PreFireCheck | updates | **恒常CRITICAL誤報** (8/12実証) | 1回/日 | 誤報 | **計画なし★** |
| 03:15 | Keiba-AM3FireCheck | updates | **恒常CRITICAL誤報** (8/12実証) | 1回/日 | 誤報 | P1停止 |
| 03:20/03:40 | Jrdb/Jvlink SupplyDaily | — | ログのみ (P1 で NightlyDiffPass 03:25 に統合) | — | — | P1置換 |
| 06:15 | Keiba-AM6FireCheck | updates | **恒常CRITICAL誤報** (8/12実証) | 1回/日 | 誤報 | P1停止 |
| 07:00 | Keiba-MorningDigest | updates | 朝ダッシュボード (誤報CRITICALを含む) | 1回/日 | 成功報告系 | P1停止 |
| 07:00 | Keiba-ScrapeProgress | — | ログのみ | — | — | 計画なし |
| 07:30 | DataFreshnessMonitor | updates | 鮮度警告 | 異常時のみ | 異常時 | P1停止 |
| 08:00 | DailyPredict | — | 非開催日は「JRDB出走表なし=非開催日(正常)」で即終了・**無送信** (8/12実証) | 0 | — | 残存 |
| 08:50 | Keiba-AM8FireCheck | updates | 「DailyPredict 正常発火」(非開催日も送信、8/12実証) | 1回/日 | 成功報告系 | P1停止 |
| 08:50 | T1v2Audit (平日=source-check) | updates | 異常時のみ通知 + BLOCK フラグ | 異常時のみ | 本命 (fail-closed) | 残存 |
| 毎時:30 | Keiba-TybPublishMonitor | updates | 平日は TYB 404 のまま=無送信 | 0 | — | P1停止 |
| 17:00 | anomaly_auto_detector | updates | critical 時のみ | 異常時のみ | 異常時 | 計画なし★ |
| 20:00 | DailyResultsEvening | updates | 非開催日は照合対象なし=実質無送信 | 条件付き | 本命 | 残存 |
| 20:30 | Keiba-RaceNotifyLogV2-Aggregator | updates | 空データ (5分前停止中) | 条件付き | レガシー | 計画なし★ |
| 21:00 | Keiba-DailyCumulativeAudit | updates | 台帳監査 | 異常時のみ | 監査 | 計画なし★ |
| 22:00 | Keiba-FeaturesIntegrityCheck | updates | 特徴量整合 | 異常時のみ | 監査 | 計画なし★ |
| 23:00 | Keiba-NightlySanity | updates | 夜間サニティ | 1回/異常時 | 監査 | 計画なし★ |
| 月 08:00 | WeeklyReport | updates | 週次レポート (**旧142.6%基準の誤DANGER既知**・最終実行 8/10) | 週1 | 本命(要基準修正) | 残存 |
| 月 08:30 | KeibaAI_DriftDetector | updates | drift 検知 | 異常時のみ(推定) | 異常時 | 計画なし★ |
| 毎日 13:00-21:30 | NAR 系 5タスク | — | Discord送信コードなし (データのみ) | — | — | 計画なし |

**平日の実送信は通常 5通** (02:55誤報 / 03:15誤報 / 06:15誤報 / 07:00ダッシュボード / 08:50 AM8) — うち **3通が恒常誤報**。phase1 適用 (8/17) 後は 02:55 誤報1通のみ残る。

---

## 取捨判断の論点 (Phase1/2 でカバーされない「計画なし★」)

| # | 対象 | 現状 | 判断が必要な点 |
|---|------|------|----------------|
| 1 | **Stage2 通知 (PreRacePredict_Watchdog_5_9)** | 土日 33通/日を #買い目 に送信 | paper期間に必要か。止めるなら task disable 1つ (bat は他ブランチ共用なので task 側で) |
| 2 | **Morning_Sat/Sun (morning_top_races)** | 06:30 に買い目8通を先行送信 (08:00 と重複)。副作用として JRDB 06:30 先行取得もこの経路 | 通知だけ止めるか丸ごと止めるか。丸ごと止める場合 06:35 WeekendAM が取得を代替済みなことは確認済 |
| 3 | Keiba-PreFireCheck 02:55 | 恒常CRITICAL誤報 (phase1 リスト漏れ) | P1 と同時に停止するのが自然 |
| 4 | RaceDayReport_Sat/Sun | 恒常不発 (同刻競合) | 削除 or DailyResults 後に移動 |
| 5 | RaceNotifyLogV2-Aggregator | 空回り | 5分前復活まで disable 可 |
| 6 | 夜間監査4本 (Coverage/Cumulative/Integrity/Sanity) + anomaly | 異常時のみで無害だが、Cumulative/Integrity が rc=1 継続中 | 残すなら rc=1 の原因確認が先 |
| 7 | ROI Monitor / WeeklyReport の旧142.6%基準 | 誤DANGER を毎開催日送信 | 既知の要承認残件 (roi_monitor/weekly_report/roi_analysis 3ファイル) |
| 8 | MultiStagePredict ×3 | updates に各1通 (paper eval) | Stage2 (#1) と役割重複気味・統合検討 |
