# Phase 20 C: 5/17 (土) 当日 schedule

**作成**: 2026-05-10 (Session #92 Phase 20 C、 ★ Opus 4.7 ★)
**前提**: V15 production 単独運用、 paper_trade_v22_runner.py 並行
**目的**: 5/17 (土) 朝-夜の全 タスク + paper trade 並行 を時系列に整理

---

## 1. 5/17 当日 タスク 時系列

| 時刻 | タスク | 内容 | schtask 名 (推定) |
|------|-------|------|------------------|
| 03:00 | DailyPremiumScrape | netkeiba premium 事前取得 (Cookie 有効化前提) | DailyPremiumScrape |
| 06:00 | daily_predict | 当日全 R 予測 (V15) → daily_predictions/20260517.csv | DailyPredict |
| 06:30 | GO 判定 worksheet | 10 項目 confirm (Phase 20 B) | (手動) |
| 07:00 | daily_predict_full | 朝 V15 全頭 score 出力 → daily_predictions_full/ | DailyPredictFull |
| 08:00 | 朝候補通知 | Discord #買い目 (V15 strategy ⑦ 候補のみ) | RaceAutoNotify (08:45-) |
| 08:45+ | RaceAutoNotify | 各 R 5 分前 通知 + 投票候補 | (土日 schtask) |
| 09:30 | SaveAllHorseScores | Stage 2 朝 全頭 score retrain (馬体重未入手) | (新規 schtask) |
| 10:00 | SaveAllHorseScores #2 | Stage 2 (馬体重 確定後) | (新規 schtask) |
| 11:00-16:00 | PreRacePredict_Watchdog | 各 R 30 min 前 Stage 2 → daily_predictions_full | (5 分 polling) |
| 各 R 30 min 前 | Stage 2 verdict | 朝予測 vs Stage 2 diff Discord | (auto) |
| 17:00 | Cumulative | 累計収支 update + Discord | (新規 schtask) |
| 18:00 | daily_results | 結果照合 + ROI 計算 | DailyResults |
| 20:30 | Summary | 当日 ROI summary + Discord | (既存 schtask) |
| 23:00 | nightly_sanity | 翌日 schtask 事前確認 + Discord | Keiba-NightlySanity |

---

## 2. paper trade 並行 schedule (5-model)

| 時刻 | タスク | コマンド | 出力 |
|------|-------|---------|------|
| 06:30 | 朝 paper baseline (前日 累計) | `python tools/paper_trade_v22_runner.py --rolling` | console |
| 19:00 | 夜 paper trade summary (5/17) | `python tools/paper_trade_v22_runner.py --date 20260517 --notify` | data/v22/paper_trade_*.csv + Discord |
| 21:00 | 累計 rolling update | `python tools/paper_trade_v22_runner.py --rolling > logs/paper_rolling_5_17.log` | logs/ |

### 2.1 schtask 候補登録 (5/17 朝までに ユーザーが setup_all_tasks.bat で登録)

```bat
:: tools/run_paper_trade_v22.bat (新規候補)
@echo off
cd /d %~dp0..
python tools/paper_trade_v22_runner.py --date %1 --notify
```

```powershell
# schtasks 登録 (毎晩 19:00、 daily_results 18:00 後)
schtasks /create /tn "Keiba-PaperTradeV22" `
  /tr "C:\Users\takum\keiba-ai\tools\run_paper_trade_v22.bat" `
  /sc daily /st 19:00 /ru takum
```

→ Phase 20 では schtask 登録は実施しない (50 件 既存 schtask 触らない方針)。
   user 手動 setup を doc 化。

---

## 3. 5/17 想定流れ (時系列)

```
03:00 DailyPremiumScrape (約 30 min、 SCRAPER-GUARD 早朝特例)
03:30  → premium data 取得完了 (調教 + comments + speed_index)
06:00 DailyPredict (約 30 min、 35 R 前後)
06:30  → V15 朝予測完了、 GO 判定 worksheet 確認
07:00 DailyPredictFull (約 20 min、 全頭 score 出力)
08:00 朝候補通知 (戦略 ⑦ 適用後 候補 Discord)
08:45 RaceAutoNotify 起動 (土日 schtask)
09:30 SaveAllHorseScores #1 (馬体重 未入手前)
10:00 SaveAllHorseScores #2 (馬体重 確定後)
10:00- 各 R 発走 (関東 = 関西 = 福島 = ローテ)
   各 R 30 min 前: Stage 2 → diff verdict Discord
   各 R 5 min 前: 投票候補 Discord
   発走後: なにもしない (実弾投入は ユーザー判断)
17:00 Cumulative update (累計収支 + Discord)
18:00 DailyResults (約 1 h、 全 R 結果取得 + ROI 計算)
19:00 PaperTradeV22 (5-model paper、 約 5 min、 Discord)
20:30 Summary (当日 V15 ROI 公式 + Discord)
21:00 paper rolling update + log
23:00 NightlySanity (翌日 5/18 schtask 事前確認)
```

---

## 4. 障害 R skip 設計

| 障害種別 | 検出 | 対応 |
|---------|------|------|
| 出走取消 (出馬表変更) | netkeiba shutuba 再取得時 馬数 mismatch | predict 再実行 (predict_one_race.py) |
| 中止 (天候) | live_odds 取得不能 | 当該 R は paper trade からも除外 (load_history で skipped) |
| 払戻なし (該当なし) | trio_payout = 0 + status=settled | hit=False で計上 (V15 production と同) |
| 結果未確定 (data 遅延) | daily_results に 該当 R なし | 翌日 paper trade で再評価 (追加処理不要) |

---

## 5. 5/17 確認 checklist (各時刻)

```
06:30 [ ] V15 朝予測 完走 (35 R)        [ ] paper baseline rolling 取得
07:00 [ ] daily_predict_full 完走         [ ] daily_predictions_full/*.json 全 R 揃う
08:00 [ ] 朝候補通知 Discord 受信         [ ] 戦略⑦ 除外件数 妥当
09:30 [ ] SaveAllHorseScores #1 完走     [ ] Stage 2 進捗 OK
10:00 [ ] SaveAllHorseScores #2 完走     [ ] 馬体重 反映確認
各 R 30min前 [ ] Stage 2 verdict Discord 受信
各 R 発走  [ ] V15 production 投票実施 (ユーザー判断)
17:00 [ ] Cumulative update Discord     [ ] 累計収支 更新確認
18:00 [ ] DailyResults 完走            [ ] 35/35 R hit/miss 確定
19:00 [ ] PaperTradeV22 Discord 受信   [ ] V15 vs V18-V22 比較
20:30 [ ] 当日 summary Discord          [ ] V15 ROI 公式値
23:00 [ ] NightlySanity 5/18 PASS     [ ] 翌日 schtask 事前確認 OK
```

---

## 6. 5/17 想定 5-model paper 結果

過去 1 ヶ月 (~ 350 R 蓄積) の延長で 5/17 朝 V15 朝予測 + 戦略 ⑦ 適用後:

| model | 想定 bet 数 (35 R 中) | 想定 ROI |
|-------|---------------------|---------|
| V15 (案 B 改 strict、 threshold 0.70) | 3-7 | 100-130% |
| V18 cand (threshold 0.75) | 1-3 | 95-110% |
| V20 cand (V18 + 16 頭+ 重み) | 1-3 | 95-115% |
| V21 cand (V20 + 動画 placeholder) | 1-3 | 95-115% (V21 は実装未) |
| V22 RL (PPO 学習済、 5000 steps) | 0-2 | 0-100% (学習量不足) |

→ 5/17 単日では 統計信頼性 低、 5/24+ で 1 週間 paper 蓄積後に評価

---

## 7. V15 投資保護 (絶対遵守)

✅ V15 model file 不変 (md5 監視)
✅ predict_core / daily_predict / app.py 不変
✅ paper trade engine は read-only
✅ 50 件既存 schtask は変更しない (新規 schtask は 5/17 朝 user 手動 setup)
✅ 累計収支 +¥14,140 維持

---

## 8. 結論

✅ 5/17 当日 schedule 整理完了
✅ paper trade 並行 schedule (06:30 / 19:00 / 21:00) 設計
✅ 障害 R skip 設計 (4 種別)
✅ 各時刻 checklist 18 項目 整備
✅ V15 投資保護完全

---

**Phase 20 C 完了** (Opus 4.7)
