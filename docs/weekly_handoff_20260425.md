# 来週末 (2026-04-25 Sat / 04-26 Sun) 申し送り

## 起床時 (土曜 AM7:00) 一発確認

```bash
cd C:\Users\takum\keiba-ai
python tools/check_scheduler_integrity.py
tail -30 logs/nightly_sanity_20260425.log
tail -30 logs/premium_scrape_20260425.log
tail -30 logs/jrdb_health_check_20260425.log
tail -50 logs/daily_predict.log
```

## 本日までに保証済みの項目

### ✅ SCRAPER-GUARD 修正 (commit `4f613a03`)
- Sat/Sun/Mon の 03:00-05:59 に daily_premium_scrape 特例許可
- OPERATIONAL_CALLERS に daily_results を追加 (defensive)
- test_scraper_guard.py 57/57 PASS
- tools/verify_scraper_guard_sunday.py で 16/16 PASS

### ✅ E2E 検証 (commit `04d972ca`)
- tools/dryrun_weekend_full.py で 4/25-27 の 17 タスク全 PASS
- SCRAPER-GUARD / import / file / dir の 4 軸を全時刻で検証

### ✅ 事前予防 (commit `04d972ca`)
- Keiba-NightlySanity: 毎日23:00 自動実行
- 翌日のタスク発火予定 + 必要ファイル + Guard挙動 を事前確認
- 異常は Discord #アップデート に red / yellow / green 色分け通知

### ✅ daily_predict 強制終了対策 (commit `1e208b97`)
- Windows コンソール Ctrl+C 防止
- 逐次 CSV 書き込み + resume 対応

### ✅ process_watchdog v2 (commit `17e3f044`)
- ログ鮮度ベース死活監視
- プロセス停止を検知したら Discord Critical 通知

## 想定される自動発火タスク

### Saturday 2026-04-25
| 時刻 | タスク | 期待動作 |
|------|--------|----------|
| 03:00 | DailyPremiumScrape | 土曜早朝特例で premium データ事前取得 |
| 06:00 | DailyJrdbKyi | KYI/SED/TYB/CYB/JOA/KAB + batch2/extra DL |
| 07:30 | JrdbHealthCheck_Sat | JRDB鮮度チェック (必要なら再取得) |
| 08:00 | DailyPredict | 当日全レース予測生成、CSV保存 |
| 08:45 | RaceAutoNotify_Sat | レース5分前Discord通知スケジュール起動 |
| 各レース5分前 | (RaceAutoNotify内から) | 買い目 Discord #買い目 投稿 |
| 18:00 | DailyResults_Sat | 結果照合、ROI計算、累積更新 |
| 20:00 | DailyResultsEvening | 再照合 (pending解消用) |

### Sunday 2026-04-26
同様フロー (RaceAutoNotify_Sun / DailyResults_Sun / 20:00 DailyResultsEvening)

## 緊急手動リカバリ

### ケース1: 朝起きたら AM3:00 の DailyPremiumScrape が失敗
```bash
set PYTHONIOENCODING=utf-8
set KEIBA_OPERATIONAL_MODE=1
python tools/daily_premium_scrape.py
```

### ケース2: AM8:00 DailyPredict がない・古い
```bash
python tools/daily_predict.py
# or with resume
python tools/daily_predict.py --resume
```

### ケース3: RaceAutoNotify が動いていない
```bash
# Task Scheduler が止まっている可能性
cmd /c "schtasks /run /tn RaceAutoNotify_Sat"
# or
python tools/race_auto_notify.py
```

### ケース4: Discord 通知が来ない
```bash
# Cookie切れ疑い
python tools/refresh_cookie.py --check
python tools/refresh_cookie.py --auto
# Webhook再設定
python tools/setup_discord.py
```

### ケース5: モデルロード失敗
```bash
ls -la keiba_model_v15_central*.pkl.gz
# 不在なら git checkout 可能な最新から復元
git log --oneline -- keiba_model_v15_central_live.pkl.gz
```

### ケース6: scraper_guard 誤停止再発
```bash
python tools/verify_scraper_guard_sunday.py
# NG があれば
cat report/task_scheduler_audit_20260419.md
```

## Discord チャネル

- #買い目 — レース予測、買い目フォーメーション、軸馬スコア、配当レンジ
- #アップデート — スクレイピング完了、結果照合、週次レポート、nightly_sanity

## 参考ドキュメント

- `CLAUDE.md` — プロジェクト全体
- `docs/incident_report_20260419.md` — 今回の事故経緯
- `report/task_scheduler_audit_20260419.md` — タスク整合性詳細
- `report/weekend_e2e_verification_20260419.md` — 本検証レポート
- `report/incident_impact_analysis_20260419.md` — 機会損失分析
- `report/v16_status_20260419.md` — v16再学習状況
- `docs/daily_handoff_20260420.md` — 月曜朝申し送り

## 今週末終了後 (4/26 夜〜4/27 AM)

- `python tools/daily_results.py --date 20260426` で当日結果確認
- CLAUDE.md の実戦成績セクション更新
- v16 データ補充の進捗確認 (`python tools/coverage_report.py`)
