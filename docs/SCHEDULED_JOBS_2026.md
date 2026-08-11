# 恒常スケジュールジョブ一覧 (2026-08-11/12 再建フェーズで新設分)

Claude セッション側の定期実行は**なし** (検証等は one-shot CronCreate で都度仕込み・発火後自動削除)。
恒常ジョブは全て Windows タスクスケジューラ。既存タスク (DailyPredict 等) は CLAUDE.md 「定期タスク」参照。

## 新設タスク (恒常・survives 再起動)

| タスク名 | 時刻 | 目的 | 実体 |
|---------|------|------|------|
| `keiba-ai\JrdbSupplyDaily` | 毎日 03:20 | JRDB 供給 (datazip日次zip 6種 + KTA/KKA日次lzh + JOA + CSV種別単独フル再構築)。旧 DailyJrdbKyi (0件バグ)+金曜Paci週次の代替 | tools/daily_jrdb_supply.bat → daily_jrdb_supply.py |
| `keiba-ai\JrdbSupplyWeekendAM` | 土日 06:35 | 同上の週末朝2パス目 (TYB 等の朝配信を予測前に取り込む) | 同上 |
| `keiba-ai\JvlinkSupplyDaily` | 毎日 03:40 | JV-Link 供給 (RACE=SE/HR/RA/O1-O6 + SLOP + WOOD + DIFF を checkpoint 差分で raw保存→CSV化) | tools/daily_jvlink_supply.bat → jv_daily_fetch.ps1(32bit) + jv_daily_parse.py |
| `keiba-ai\T1v2Audit` | 毎日 08:50 | 特徴量監視。土日=ライブ feat_dump 監査 (ゾンビ/JRDB死/スコア圧縮→CRITICAL で予測ブロック) / 平日=供給 source-check (JRDB KYI/SED + JV SE/HR 鮮度、PASS で BLOCK 自動解除) | tools/t1v2_audit.bat → t1v2_feature_audit.py |
| `keiba-ai\MorningBatchNotify` | 土日 09:30 | 朝一括通知 (T1v2 fail-closed・買い目 embed+特徴量詳細) + 特徴量健全性レポート | tools/morning_batch_notify.bat → morning_batch_notify.py + feature_health_report.py |

## 停止・削除方法

```bat
:: 一時停止 (推奨: 状態を残す)
schtasks /change /tn "keiba-ai\JrdbSupplyDaily" /disable
:: 再開
schtasks /change /tn "keiba-ai\JrdbSupplyDaily" /enable
:: 完全削除
schtasks /delete /tn "keiba-ai\JrdbSupplyDaily" /f
```
(他タスクも同名パターン。管理者不要 = 全て当ユーザー権限で登録済)

★注意: JrdbSupplyDaily/JvlinkSupplyDaily を止めると T1v2 source-check が供給鮮度 NG →
次の開催日に BLOCK フラグが立ち予測が自動停止する (fail-closed 設計・意図どおりの連鎖)。★

## 8/11 に無効化した旧タスク (再有効化しないこと)
| タスク | 状態 | 理由 |
|--------|------|------|
| keiba-ai\DailyJrdbKyi | disabled | scrape_jrdb --date TODAY の恒常0件バグ (死因#2) |
| keiba-ai\Keiba-WeeklyScrapeResume | kill+disabled | netkeiba premium 一括 (解約) |
| keiba-ai\DailyPremiumScrape / Keiba-FridayWeekendScrape | bat スタブ化 (タスク disable は要管理者 = tools/disable_netkeiba_tasks_admin.bat) | premium 死 / Paci は JrdbSupplyDaily が代替 |
| keiba-ai\RaceAutoNotify_Sat/Sun | bat スタブ化 (同上) | 5分前通知は paper 期間停止 (復元 = race_auto_notify.bat.bak_20260811) |

---

# 統合再設計 (2026-08-12 承認・段階移行)

## 移行スケジュール (各段階の完了条件 = T1v2 PASS 継続)
| 日付 | 適用 | 手段 |
|------|------|------|
| 8/15-16 (今週末) | **現行構成のまま** (変数を足さない)。追加は ①KAB取得の現行朝パス同乗 (実装済・jrdb_kab.csv 8/9まで復活) ②土曜05:00-08:00 当日TYB/KAB配信時刻probe (`ProbeSaturdayPublish` 8/15 05:00 one-time) | 実装済 |
| 8/17(月)〜 | 系統C: `NightlyDiffPass` 03:25 (毎日・JRDB+JV統合1本) 開始 / 旧 JrdbSupplyDaily+JvlinkSupplyDaily disable / 監査系8本 disable (am3/am6/am8=ゾンビ監視・Digest・Freshness・TybPublish・JrdbRetry×2=死因#2経路) | apply_phase1_20260817.bat (cron f42eeaac 自動 or 手動実行) |
| 8/21(金)〜 | 系統A: `WeeklyMainPass` 金20:10 初回 (週末バンドル+JV今週系TCOV/RCOV option=2) | タスク登録済 (/sd 08/21) |
| 8/22-23 | 系統B: `RaceDayMorningPass` 土日06:50 (暫定。probe結果+15分で確定) / 系統E: 通知v2 (健全性1メッセージ同梱・別送廃止・MorningWeightCheck停止) / JrdbSupplyWeekendAM disable | apply_phase2_20260821.bat (cron 2572ec4b 自動 or 手動実行) |

## 新構成 (2週目〜) の5系統
A 金曜夜 20:10 週次メイン / B 開催日朝 06:50 当日差分(TYB/KAB) / C 毎日 03:25 差分追随 /
D 監査 08:50 T1v2 (発火確認を吸収済=供給ログALL OK必須化) / E 通知 = 開催日 朝1バースト(09:30 サマリー+健全性同梱+買いembed) + 夜(18:00 結果系)・平日ゼロ(異常時のみ)

## fail-closed 連鎖 (維持・強化)
供給NG or ★ジョブ未発火★ → T1v2 source-check FAIL → BLOCK フラグ → daily_predict 停止 (発火確認吸収により従来より強い)

## セッション断時の手動フォールバック
- 8/17朝: `tools\apply_phase1_20260817.bat` を実行 → T1v2 PASS 確認
- 8/21夜: probe ログ確認 → `tools\apply_phase2_20260821.bat` を実行
- denied 分の正式 disable: `tools\disable_legacy_audit_admin.bat` (管理者)
