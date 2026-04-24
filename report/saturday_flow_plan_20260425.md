# 土曜本番フロー計画 2026-04-25 (土)

作成: 2026-04-24 20:00
適用: 4/25(土) 02:55 〜 20:00

## タイムテーブル (タスクスケジューラ自動発火)

| 時刻 | タスク | 想定Discord通知 | 人間アクション |
|---|---|---|---|
| 02:55 | Keiba-PreFireCheck | (失敗時のみ) `#updates` にCRITICAL通知 | — (就寝中) |
| 03:00 | DailyPremiumScrape | 完了時 `#updates` Premium取得ログ | — |
| 03:15 | Keiba-AM3FireCheck | 異常時 `#updates` | — |
| 06:00 | DailyJrdbKyi | JRDB KYI/SED/TYB 取得 → CSV更新 | — |
| 06:15 | Keiba-AM6FireCheck | 異常時 `#updates` | — |
| 07:00 | Keiba-MorningDigest | `#updates` 朝サマリー | 朝起きたらスマホで確認 |
| 07:00 | Keiba-ScrapeProgress | スクレイプ進捗レポート | — |
| 07:30 | JrdbHealthCheck_Sat | JRDB 取得健全性 → `#updates` | 失敗ならスマホ通知受領 |
| 08:00 | DailyPredict | v15 Pattern B で全レース予測 → `data/daily_predictions/20260425.csv` 出力 | — |
| 08:45 | RaceAutoNotify_Sat | 買い目 整形Discord `#bets` | **スマホで買い目確認** |
| 08:50 | Keiba-AM8FireCheck | 異常時 `#updates` | — |
| 09:00〜 | (発走30分前自動通知ループ) | `#bets` 各レース個別通知 | JRA即PATで馬券購入 |
| 18:00 | DailyResults_Sat | 結果照合 → 的中/ROI集計 → `#updates` | 夕方成績サマリー確認 |
| 20:00 | DailyResultsEvening | 平日含む結果照合フォールバック | — |
| 23:00 | Keiba-NightlySanity | 翌日(日)予定タスク事前チェック | 就寝前確認 |

## チェックリスト (朝起きて確認する順)

### 07:00〜08:00 (朝食時)
1. `#updates` に `MorningDashboard` が来ているか
2. 「CRITICAL: 0 / WARNING: 0 / 手動介入不要」なら放置
3. JrdbHealthCheck_Sat 通知を確認
4. CRITICAL あれば即 Claude Code 起動して対処:
   ```bash
   cd keiba-ai && claude --dangerously-skip-permissions
   ```

### 08:45〜09:00 (発走前)
1. `#bets` に「全Nレース 買い目一覧」通知が来る
2. スマホで投資額合計 (通常 25,000〜28,000円) を確認
3. 条件X/C/B の本命レースを軸にチェック

### 各レース発走前 (ループ)
1. `#bets` に個別レース通知 (`RaceAutoNotify`)
2. 買い目を JRA 即PAT で購入
3. スマホPUSH → 5分以内に決裁

### 18:00 (結果照合)
1. `#updates` に「結果照合」通知
2. 的中率 / ROI / 累積利益を確認
3. CRITICAL (連敗閾値超え) あれば後で対処

## 想定 Discord 通知タイムライン (土曜分)

```
02:55 [PreFireCheck] 状態: OK (2秒)
03:00 [PremiumScrape] 開始
04:07 [PremiumScrape] 完了 (72レース取得)
06:00 [DailyJrdbKyi] 開始
06:01 [DailyJrdbKyi] 完了 (KYI/SED/TYB 更新)
06:15 [AM6FireCheck] OK
07:00 [MorningDigest] 朝サマリー (📊 ダッシュボード)
07:30 [JrdbHealthCheck] 健全性レポート
08:00 [DailyPredict] 開始 → 36レース予測 → CSV保存
08:20 [DailyPredict] 完了
08:45 [RaceAutoNotify] 買い目一覧 (前半レース)
09:00〜 個別レース通知 (発走30分前)
15:40 頃 メインレース通知
18:00 [DailyResults] 結果照合 → ROI 通知
23:00 [NightlySanity] 翌日(日)タスク事前チェック
```

## 緊急時アクション

| 症状 | 対処 |
|---|---|
| `#updates` に 🔴 CRITICAL | `claude --dangerously-skip-permissions` 起動, 内容確認 |
| DailyPredict失敗 (8:20までに完了通知なし) | `python tools/daily_predict.py --date 20260425` 手動実行 |
| Cookie WARN | `python tools/refresh_cookie.py --auto` |
| RaceAutoNotify 無反応 | `python tools/notify_bets_all_in_one.py --date 20260425` 手動実行 |
| JRDB age=N/A | `python scripts/daily_jrdb_kyi.py` 手動実行 |

## 現在の準備状態 (2026-04-24 20:00)

- ✅ Cookie 更新済 (19:44)
- ✅ pytest 全PASS
- ✅ タスクスケジューラ 21/22 Ready (ProcessWatchdog Disabled は仕様)
- ✅ v15 model 存在
- ✅ ディスク空き 743.7 GB
- ⚠️ JRDB csv age=5日 → 明朝 06:00 自動解消
- ✅ 土日本番タスク NextRun 正しい

→ **手動介入不要見込み**。朝起きたら Discord 確認のみで OK。
