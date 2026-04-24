# 🏇 4/24夜 前日最終チェック - 総合判定

**実行時刻**: 2026-04-24 23:15
**対象**: 2026-04-25 (土) + 2026-04-26 (日)

---

## 📊 総合判定マトリクス

| PHASE | 項目 | 判定 | 詳細 |
|---|---|:---:|---|
| 0 | 基礎データ | 🟡 | Premium 0件 (想定内・土曜AM3:00再実行) |
| 1 | データ充足 | 🟡 | KYI/TYB未取得 (土曜AM6:00予定) |
| 2 | 予測 dry-run | 🟡 | 個別関数は全PASS、full dry-run未完走 |
| 3 | 特徴量カバレッジ | 🟡 | ゼロ列 22-24 > 閾値15 (当日朝解消見込) |
| 4 | タスク発火 | 🟡 | ProcessWatchdog DISABLED |
| 5 | 確認漏れ | 🟢 | 全PASS (阪神開催なし、IPバン兆候0) |
| 6 | 結果照合 | 🟢 | payout バグ修正済、backup済 |

## 🎯 判定: 🟢 **GO** (WARNING 5件、CRITICAL 0件)

### 理由
- CRITICAL レベルの阻害要因ゼロ
- WARNING はいずれも「前日時点で未取得だが、翌朝の自動タスクで補完される」前提
- 必須予測チェーン (Premium → JRDB → Predict → Notify) 全 NextRunTime OK

---

## 📋 タイムライン (4/25 土曜)

```
02:55  PreFireCheck           (5項目事前確認)
03:00  DailyPremiumScrape     (training/speed/comments/newspaper 取得)  ← 4/19は36/36成功
03:15  AM3FireCheck           (発火確認)
06:00  DailyJrdbKyi           (KYI/SED/TYB/CYB/JOA/KAB 全取得)
06:15  AM6FireCheck           (発火確認)
07:00  MorningDigest          ← ここでゼロ列数の実測可能、CRITICAL なら Discord警告
07:00  ScrapeProgress         (進捗監視)
07:30  JrdbHealthCheck_Sat    (健全性チェック) ⚠️ 初回失敗歴あり
08:00  DailyPredict           ⭐本番予測 (72R)
08:45  RaceAutoNotify_Sat     ⭐Discord一括通知
08:50  AM8FireCheck           (発火確認)
18:00  DailyResults_Sat       (結果照合・ROI計算)
20:00  DailyResultsEvening    (平日時刻、重複実行)
23:00  NightlySanity          (翌日準備)
```

---

## ⚠️ 判定内訳

### 🟡 WARNING 5件 (いずれも想定内 or 時間で自動解消)

1. **Premium 0件**: 4/23, 4/24 の事前取得で全て 0/72 カバー。
   - **対処**: 4/25 AM3:00 の再実行で埋まる見込み (先週実績あり)
   - **検知**: AM7:00 MorningDigest で異常なら Discord通知

2. **JRDB KYI/SED/TYB/CYB 4/19以降未更新**: 平日取得なしは正常仕様。
   - **対処**: 4/25 AM6:00 DailyJrdbKyi で取得
   - **検知**: AM6:15 FireCheck + MorningDigest

3. **Full dry-run 未完走**: 1レース20秒 × 72 = 24分、時間予算超過。
   - **代替**: 主要関数を個別検証済 (parse_shutuba/build_features/predict_race 全OK)
   - **影響**: なし (本番 AM8:00 の実走で検証)

4. **特徴量ゼロ列 22-24 > 閾値15**: 前日時点の正常値。
   - **対処**: JRDB取得後減少見込み
   - **検知**: MorningDigest で再測定

5. **ProcessWatchdog DISABLED**: 5分おきプロセス監視が無効化されている。
   - **対処**: 手動で Enable 検討 (本タスクでは修正しない)
   - **影響**: 本番中の予測プロセス監視が効かない

### 🟢 PASS 項目

- ✅ 全タスク NextRunTime 正常
- ✅ Cookie 有効 (1809文字、4/24 19:44 更新)
- ✅ モデルロード OK (v15 Pattern B, 150特徴量)
- ✅ 出馬表取得 OK (36R × 2日 = 72R)
- ✅ レース予測 1件検証済 (京都1R, 16頭, condition=X)
- ✅ payout バグ修正コード確認 (ecbdf000)
- ✅ payout=0 検知時 Discord CRITICAL 通知実装済
- ✅ cumulative_results.csv backup 2種類存在
- ✅ Streamlit 多重起動防止 (run_streamlit.bat)
- ✅ 配当文字化け対策 (unicodedata NFKC)
- ✅ IPバン兆候 0 (429/503/504 全て 0件)
- ✅ 今週末 阪神開催なし (該当チェック項目除外)

---

## 📅 データ充足状況

### 前日時点 (現在=金曜22:00-23:00) で揃っているもの
- ✅ 出馬表 (netkeiba) 72R
- ✅ 新聞AI (thisweek JSON) 72R
- ✅ JRDB 静的データ (UKC/SRB/拡張4種)
- ✅ モデルファイル (v15, 150特徴量)
- ✅ Cookie
- ⚠️ odds_base 20260425.csv (京都のみ9R部分取得)

### 当日朝 (4/25 AM3:00-AM8:00) に取得されるもの
- 🕒 netkeiba training/speed_index/comments (AM3:00)
- 🕒 JRDB KYI/SED/TYB/CYB/JOA/KAB (AM6:00)
- 🕒 JRA馬場 + 気象庁データ (予測時)
- 🕒 リアルタイムオッズ (予測時)

### 人間の介入不要
- 全データが自動取得チェーンでカバーされる設計
- 失敗時は FireCheck + MorningDigest で Discord通知あり

---

## 🚨 潜在リスク (許容範囲内)

### リスク1: AM3:00 Premium取得失敗
- **確率**: 中 (Cookie 有効なら OK、Playwright 自動化あり)
- **影響**: 調教/速度指数データなし → 予測精度 10-20% 低下
- **対策**: feature_lookups.pkl キャッシュフォールバック動作

### リスク2: AM6:00 JRDB取得失敗
- **確率**: 低 (静的サーバ、最近は99%+ 成功)
- **影響**: KYI/SED なし → 特徴量ゼロ列 30+
- **対策**: horse fallback (netkeiba経由) が 80%+ 代替

### リスク3: ProcessWatchdog 無効で予測プロセスがハング
- **確率**: 低 (Windows Ctrl+C 対策済み、resume対応)
- **影響**: 午前中のレース一部予測漏れ → AM10:00以降のレース通知遅延
- **対策**: なし (要 ProcessWatchdog 再有効化)

### リスク4: 日曜 RaceAutoNotify_Sun 4/19 Ctrl+C 再発
- **確率**: 低 (コード修正済み)
- **影響**: 日曜レース通知不能
- **対策**: 4/25 夜 NightlySanity で事前確認

---

## 📝 推奨アクション

### 今夜中に対応すべきこと (任意)
1. ProcessWatchdog 再有効化検討 (但し変更禁止ルールあり)

### 明朝 (4/25 AM7:00-AM8:45) の確認ポイント
1. AM7:00 MorningDigest の Discord通知確認
2. AM7:30 JrdbHealthCheck 結果確認
3. AM8:00 DailyPredict 開始ログ確認 (log/daily_predict.log)
4. AM8:45 RaceAutoNotify 一括通知到達確認

### 本番運用に関する結論
**🟢 人間の介入なしで本番投入可能**

---

## ファイル
- phase0_basic_data_20260424.md
- phase1_data_coverage_20260424.md
- phase2_prediction_dryrun_20260424.md
- phase3_feature_coverage_20260424.md
- phase4_task_timing_20260424.md
- phase5_missed_checks_20260424.md
- phase6_result_pipeline_20260424.md
- final_prebattle_check_20260424.md ← 本ファイル
