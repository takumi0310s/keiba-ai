# 明朝 (2026-04-20 Mon) 申し送り

## 起床直後 (AM7:00〜9:00) に一発確認

```bash
cd C:\Users\takum\keiba-ai
python tools/check_scheduler_integrity.py
tail -20 logs/nightly_sanity_*.log
tail -30 logs/premium_scrape_20260420.log
tail -30 logs/daily_predict.log
tail -30 logs/jrdb_kyi_auto_20260420.log
tail -30 logs/weekly_report.log
```

## 確認項目

### AM03:00 DailyPremiumScrape
```bash
tail -30 logs/premium_scrape_20260420.log
```
期待:
- `[SCRAPER-GUARD] BYPASS: caller=daily_premium_scrape reason=...` OR 普通に実行
- 途中で `[SCRAPER-GUARD] ... 週末レース時間帯...` が **出ないこと** (出たら事故再発)

### AM06:00 DailyJrdbKyi
```bash
tail -30 logs/jrdb_kyi_auto_20260420.log
```
期待: KYI/SED/TYB/CYB/JOA/KAB/kta/cha/kka/jo の DL 完了

### AM08:00 DailyPredict (非開催日なので軽量実行)
```bash
tail -30 logs/daily_predict.log
```
期待: `[INFO] 20260420 のレースが見つかりません（非開催日の可能性）` で正常終了

### AM08:00 WeeklyReport
```bash
tail -30 logs/weekly_report.log
```
期待: 週次ROIレポートがDiscord #アップデート に投稿されている

### AM08:30 KeibaAI_DriftDetector
```bash
tail -30 logs/drift_detector.log
```
期待: モデルドリフト検知の結果が出力

### 万一 DailyPremiumScrape が blocked されていたら
手動リカバリ:
```bash
# 03:00の自動実行が失敗していた場合、手動実行
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set KEIBA_OPERATIONAL_MODE=1
python tools/daily_premium_scrape.py
```

### 万一 DailyPredict が crash していたら
```bash
# resume対応しているので単純に再実行
python tools/daily_predict.py --resume
```

## 先週の WeeklyReport 解説 (参考)

CLAUDE.md §実戦成績 (2026-03-14〜04-18, 324R):
- 全体 ROI 120.2%, +45,920円
- 条件A: 122.9% (目標143.7%) — やや劣勢だが上向き
- 条件D: 144.3% (目標95.2%) — 好調継続
- 条件B/E/X: 低ROI (N小サンプル)

## 今夜 Committed Changes

```
4/19 23:00までのpush履歴:
- 04d972ca feat: 来週末E2E検証 (フェーズD)
- d354a7a7 feat: カバレッジレポート + v16判定 (フェーズC)
- af193394 docs: 事故インパクト分析 (フェーズB)
- 705879f3 Merge fix/sunday-am3-regression-check
- 4f613a03 fix: 明日AM3:00再発防止 (フェーズA)
```

## 新規スケジューラ登録

- Keiba-NightlySanity (PM23:00 毎日) — 翌日発火予定タスクを事前チェック

## Discord通知チャネル

- #買い目 — レース予測、買い目フォーメーション
- #アップデート — スクレイピング完了、結果照合、nightly_sanity, 週次レポート

## 次ステップ (来週)

1. 月曜平日: scrape_missing_all を再開して v16 データ補充
2. 木曜: カバ率確認 (`python tools/coverage_report.py`)
3. 金曜23:00: Keiba-NightlySanity が翌Sat予告チェック
4. 土日: v15 継続運用、race_auto_notify で買い目発信

## 緊急連絡

- 何かおかしい → `python tools/project_status.py` で全体確認
- ログ取得に失敗 → `tools/refresh_cookie.py --auto` でCookie更新
- SCRAPER-GUARD 誤停止再発 → `report/task_scheduler_audit_20260419.md` の経緯確認
