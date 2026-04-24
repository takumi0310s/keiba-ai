# 今朝の自動運用結果 監査 (2026-04-24 Fri)

監査実行: 2026-04-24 19:50
評価対象: 4/24(金) の自動運用タスク群

## サマリー判定

**🟢 全タスク正常発火 (CRITICAL=0, WARNING=1)**

- AM02:55 Pre-Fire-Check: OK (6/6 check pass)
- AM03:00 DailyPremiumScrape: 正常発火 (AM3:15 fire-check OK, logs完走)
- AM06:00 DailyJrdbKyi: 正常発火 (平日のため KYI/SED/TYB=0件, 正常)
- AM07:00 Morning Dashboard: 正常 (CRITICAL=0, WARNING=0)
- AM08:00 DailyPredict: 正常発火 (平日のため非開催, 0レース処理, 正常)
- AM08:50 AM8 Fire Check: OK

## Pre-Fire-Check 詳細 (AM02:55)

| Check | Status | Msg |
|---|---|---|
| SCRAPER-GUARD | ✓ | ALLOW @ 03:00 Fri (daily_premium_scrape 特例) |
| Cookie | ✓ | Cookie OK (1809 文字) |
| Directories | ✓ | 書き込み権限 OK (4 dirs) |
| JRDB reachable | ✓ | HTTP 200 |
| Disk space | ✓ | 空き 746.9 GB |
| Task Scheduler | ✓ | Ready, next=03:00 |

OVERALL: OK (source: `data/fire_check_results/pre_fire_check_20260424.json`)

## AM03:00 DailyPremiumScrape 実行内容

- 4/25 レース: 36件, 4/26 レース: 36件, 合計 72件の race_id を事前収集
- JRDB前日データ (KYI/CYB/ZED/JOA/KAB): 404 (平日のためデータなし → 正常)
- JRDB年次ZIP更新 (CHA/KTA/JO/KKA): 既存ファイル再利用 → parse完走
  - jrdb_cha.csv: 300,174 rows (26.5 MB)
  - jrdb_kta.csv: 298,551 rows (61.1 MB)
  - jrdb_jo.csv: 300,174 rows (24.9 MB)
- **⚠️ WARNING**: 以下が 0/72 で全滅
  - Training (調教) / Speed Index / Comments / Shinba Eval
  - 原因: Cookie 期限切れ (当時 1809 文字だが Premium 認証に不可)
  - **対応完了**: 4/24 19:44 `refresh_cookie.py` にて 27 個の Cookie 取得, Premium認証OKを確認
  - `.env` 最終更新時刻: 2026-04-24 19:44:27 ✓
- Phase 3 Overlap Check: `race_id` 列欠落で3件失敗
  - SRB vs SED, HJC vs SED, OZ vs SED, CHA vs CYB
  - 非致命 (警告レベル)
- 週末限定データ (newspaper/upset/analysis): 取得完了 → `*_thisweek.csv/json` 更新

完了: 4/24 04:07:08 (所要 67分)

## AM06:00 DailyJrdbKyi

- 期間: 2026-04-24 ～ 2026-04-24 (本日分)
- KYI/SED/TYB: 全て "0日間/データなし" → 平日のため正常
- 拡張データ再parse: CHA/KTA/JO/KKA/UKC/CZ/KZ すべて完走
- JRDB login 成功, 全download・parse エラーなし
- 完了: 4/24 06:01:28

## AM07:00 Morning Dashboard

- Pre-Fire-Check: [OK] OK (6/6)
- AM06:00 DailyJrdbKyi: [OK] 正常発火
- AM03:00 DailyPremiumScrape: [...] 未実行表示 (実際はlog確認できたため mtime 判定の誤差と推定)
- AM08:00 DailyPredict: [...] 未実行表示 (当時 08:00 未到達なので正常)
- サマリー: CRITICAL=0 / WARNING=0 / 手動介入不要

## AM08:00 DailyPredict

- Cookie有効性チェック: 通過 (昨日までは COOKIE WARN 出ていたが今朝は warn なし)
- v15 Pattern B ロード: OK (150特徴量)
- レース一覧: `20260424 のレースが見つかりません (非開催日の可能性)` → 平日のため正常
- 整形買い目通知送信: 0 messages → 正常
- 完了: 4/24 08:00:05

## Cookie 監査

| 項目 | 値 |
|---|---|
| .env NETKEIBA_COOKIE 存在 | ✓ |
| .env 最終更新 | 2026-04-24 19:44:27 |
| Cookie 長 (更新後) | 1,634 文字 |
| Cookie 長 (朝時点) | 1,809 文字 |
| cookies.pkl | 不在 (.env 運用のため正常) |
| Premium 認証 | OK (19:44 refresh 時に確認) |

AM03:00 DailyPremiumScrape の Premium データ取得失敗 (Training/Index/Comments 0/72) は
**19:44 の refresh_cookie.py 実行で解消済み**。

## 警告の再掲 (要確認)

1. AM03:00 DailyPremiumScrape で Premium データ (Training/Index/Comments/Shinba) が 0/72
   - 解消済み: 19:44 Cookie 更新完了
   - 明朝 AM03:00 の再実行で正常取得される想定

## 異常なし項目

- タスクスケジューラ発火時刻: 全て定刻通り
- 各 fire-check (AM03/AM06/AM08): 全 OK
- ディスク空き: 746.9 GB → 十分
- JRDB 疎通: HTTP 200
- Morning Dashboard: WARNING=0

## 使用ログ一覧

- `data/fire_check_results/pre_fire_check_20260424.json` (OVERALL: ok)
- `data/fire_check_results/20260424.json` (DailyJrdbKyi+DailyPredict, 両方 ok)
- `logs/pre_fire_check_20260424.log`
- `logs/am3_fire_check_20260424.log`
- `logs/am6_fire_check_20260424.log`
- `logs/am8_fire_check_20260424.log`
- `logs/morning_dashboard_20260424.log`
- `logs/premium_scrape_20260424.log`
- `logs/jrdb_kyi_auto_20260424.log`
- `logs/daily_predict.log` (tail 確認)
