# 4/27 (月曜) v16 準備朝活レポート

## 📅 作業時刻
- 開始: 2026-04-26 21:00 (日曜深夜)
- 一旦休憩: 2026-04-27 00:43 (月曜)
- 再開: 2026-04-27 09:00 (月曜)
- 終了: 2026-04-27 16:00 頃 (予定)

## 🎯 達成した成果

### コード修正
1. **戦略⑦自動化** (`tools/race_auto_notify.py`)
   - 06_特別 / 京都 / 条件E / 条件B を除外
   - 期待効果: ROI 119% → 140% (+21pt)
   - 実証効果: 直近4週で +24pt 改善

2. **course_renovated 永久化** (`tools/predict_core.py`)
   - 12ヶ月制限 → 永久フラグ
   - 京都2万件のデータ活性化

3. **predict_core.py FutureWarning 修正**
   - 14箇所の dtype を `int 0` → `float 0.0` に変更

4. **1レース再予測ツール** (`tools/predict_one_race.py`)
   - 取消発生時の緊急予測対応
   - 動作確認: 東京11Rフローラ S 成功

5. **CSV重複削除** (`data/cumulative_results.csv`)
   - 4/05 + 4/11 山藤賞 重複1件削除

6. **nightly_sanity_check 修復** (`tools/nightly_sanity_check.py`)
   - 絵文字 ✅/❌/⚠ → ASCII [OK]/[NG]/[WARN]
   - cp932 codec エラー解消

7. **race_auto_notify.bat 修復**
   - powercfg restore による誤エラーコード解消
   - exit /B %EXITCODE% 追加

### GitHub コミット
- `364a9260`: v16 prep (戦略⑦/course_renovated/FutureWarning/1race-predict)
- `9a52c443`: nightly_sanity 修正
- `fc87f838`: race_auto_notify.bat 修復

### 重要発見
1. **gaisha_rank 100%カバー判明** (元データ jrdb_kyi.csv)
   - v16 計画変更: 削除 → 残す
   - 訓練データキャッシュ再構築で復活

2. **戦略⑦ 実証効果**
   - 4/12: +30.7pt
   - 4/19: +7.8pt
   - 4/25: +14.7pt
   - 4/26: +34.5pt

3. **RaceAutoNotify 健全** (誤検知だった)
   - エラーコード -1073741510 は powercfg 失敗が原因
   - Python 本体は正常動作

4. **JRDB 全タイプ最新化**
   - jrdb_paci.csv: 4/4 → 4/26
   - 547,602 行に拡張

## 🎯 5/2 (土) GW初日の予測

### 想定動作
03:00 DailyPremiumScrape
03:15 Keiba-AM3FireCheck
06:00 DailyJrdbKyi (KYI取得)
06:15 Keiba-AM6FireCheck
07:00 Keiba-MorningDigest + Keiba-ScrapeProgress
07:30 JrdbHealthCheck_Sat
08:00 DailyPredict (全レース予測)
08:45 RaceAutoNotify_Sat (戦略⑦適用 + Discord通知)
08:50 Keiba-AM8FireCheck
18:00 DailyResults_Sat
20:00 DailyResultsEvening
23:00 Keiba-NightlySanity

### 期待値
- 予想レース数: 32R (戦略⑦適用後)
- 投資額: 22,400円
- 期待ROI: 109.6%
- 期待損益: +2,158円

## 🚀 v16 開発計画 (修正版)

### 削除する特徴量
- `prev_race_pace_diff` (0%充填、修復困難) ← 削除
- ~~`gaisha_rank`~~ ← 残す! (元データ100%カバー判明)

### 修正する特徴量
- `course_renovated`: 永久化済み

### 期待値
- 特徴量: 150 → 149
- 訓練データ: 527,280 → 547,000+
- 期待 WF AUC: 0.895-0.905
- 期待 ROI: 140%+

## 📋 残タスク (中期)

### 5/2 後 (5/4-5/6)
- [ ] jra_payouts.csv 4/12以降の修復
- [ ] daily_results.py の top1_num/score 95%欠損修復
- [ ] predict_and_log.py の v15 対応化

### 5/11 以降 (GW後)
- [ ] v16 学習実行 (train/run_v16_and_am8_wf.py)
- [ ] 京都データ蓄積後の戦略⑦再評価

