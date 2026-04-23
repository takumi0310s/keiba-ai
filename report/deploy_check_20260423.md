# Deploy Check 20260423

実行日時: 2026-04-23 14:23
本番: 2026-04-25 (土)

## 判定

**🟡 警告 (2) — 確認推奨**

- pytest FAIL (環境互換問題の可能性):   rootdir: C:\Users\takum\keiba-ai
- cookies.pkl 不在 (env のCOOKIEのみで運用)


## Cookie 状態
- .env NETKEIBA_COOKIE 設定: True
- cookies.pkl: False
- 期限切れの場合: [CRITICAL] cookie期限切れ、金曜昼に python tools/refresh_cookie.py 必須

## pytest
- ok: False
- summary: `  rootdir: C:\Users\takum\keiba-ai`

## タスクスケジューラ (Keiba)
(State enum: 1=Disabled, 3=Ready, 4=Running)

- Keiba-AM3FireCheck: Ready
- Keiba-AM6FireCheck: Ready
- Keiba-AM8FireCheck: Ready
- Keiba-FridayWeekendScrape: Ready
- Keiba-MorningDigest: Ready
- Keiba-NightlySanity: Ready
- Keiba-PreFireCheck: Ready
- KeibaAI_DriftDetector: Ready
- Keiba-ScrapeProgress: Ready
- Keiba-WeeklyScrapeResume: Ready

## モデル
- keiba_model_v15_central_live.pkl.gz: exists=True size=2.0MB age=14日
- keiba_model_v15_central.pkl.gz: exists=True size=2.0MB age=14日

## JRDB データ
- jrdb_kyi.csv: exists=True age=4日 size=88.9MB
- jrdb_sed.csv: exists=True age=4日 size=25.6MB
- jrdb_tyb.csv: exists=True age=4日 size=210.3KB
- jrdb_cyb.csv: exists=True age=4日 size=151.3KB

## ディスク
- 空き: **745.7GB** (使用率 21.7%)

## 構文チェック
- app.py: ok=True
- tools/predict_core.py: ok=True

## 直近予測
- {'exists': True, 'count': 8, 'latest': ['20260418.csv', '20260418_prerace.csv', '20260419.csv']}

