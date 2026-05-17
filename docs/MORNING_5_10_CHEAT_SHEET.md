# 5/10 朝確認用 Cheat Sheet

> Session #86 (5/9 21:00) 作成
> GW 146h+ マラソン 後の 翌朝 (5/10 土) スムーズ起動用
> 起床後 5-10 分で 全体把握 + 投資判断

## 起床直後 (08:00-08:30) Discord 確認 list

優先順 (上から下):

1. **DailyPredict_0800 完了通知** (Discord #アップデート)
   - 全 R 予測完了 line
   - エラーあれば 「ERR」 / 「FAIL」 line で 即覚知

2. **MorningWeightCheck_0930 通知** (Discord #アップデート)
   - 馬体 ±10kg alert
   - 朝予測との diff alert

3. **5/9 1 day summary** (Discord、 前日 20:00 送信済)
   - 5/9 戦績 ROI
   - 累計 (前日 +13,530 円 → 5/9 反映後) ※ 旧 +13,530 円 は当時 record、 5/16 P0-1 真値 +¥5,240 / n=563 (docs/ROI_DISCREPANCY_2026_05_16.md)

4. **SaveAllHorseScores_0930 完了通知** (★ 5/10 初回 ★)
   - 全馬 score 保存 (Session #71 新機能)
   - これで 翌週以降 V18/V19 retro が完全 data で実行可能

5. **PreRacePredict 関連** (5/9 まで disable、 5/10 朝 確認)
   - Session #78 で disable 済 → 通知なし が正常

6. **NightlySanity (前夜 23:00)**
   - 翌日 schtask 健全性 check 結果
   - 全 schtask Ready なら OK

## schtasks 動作確認 (CMD or PowerShell)

```powershell
# 朝 (08:30 までに) 動いた task list
schtasks /Query /FO LIST | Select-String "DailyPredict|MorningWeightCheck|SaveAllHorseScores" -Context 0,2
```

期待結果:
- DailyPredict_0800: Status=Ready or Running
- MorningWeightCheck_0930: 09:30 まで Ready (08:30 時点)
- SaveAllHorseScores_0930: 同上 (★ 5/10 初回 ★)

## V15 投資 候補確認 path

```
1. data/daily_predictions/20260510.csv
   ↓
2. 戦略⑦ + 案B改 適用
   - 06_特別 除外
   - 京都 除外
   - 条件 E / B 除外
   ↓
3. 案 A 想定 (700円×3R) or 案 B 改 (12R 1勝クラスのみ 上限 2,100円)
   ↓
4. 12:00 までに 投票 PAT 入力
```

## 5/10 (土) のレース schedule 想定

| 場 | 重賞 | 一般 R |
|----|------|-------|
| 東京 | NHKマイルC (G1) 5/10 R11 | 5/10 R1-12 |
| 京都 | 京王杯SC (G2) 5/10 R11 (or 18) | 京都 5/10 R1-12 |
| 新潟 | — | 新潟 5/10 R1-12 |

★ 京都 R は 戦略⑦ で 投票候補から除外 (data 蓄積待ち、 5/11 以降に再評価) ★
→ 投票候補は **東京 + 新潟 のみ**

## 5/10 ★ 重要 ★ 新機能 確認 list

| 機能 | 期待動作 | 確認 method |
|------|---------|------------|
| SaveAllHorseScores_0930 | 全馬 score を CSV 保存 | data/v18/horse_scores_20260510.csv 生成確認 |
| PreRacePredict (disable 中) | 通知 ゼロ | Discord 送信なし が正常 |
| morning_weight_check | 09:30 完了 | data/morning_weight_check/20260510.csv 生成 |
| NightlySanity | 前夜 23:00 完了 | logs/nightly_sanity_20260509.log 確認 |

## 5/16 V18 trial までの 1 週 schedule

| 日 | task |
|----|------|
| 5/10 (土) | V15 案 A 維持 投票、 全馬 score 初収集 |
| 5/11 (日) | V15 案 A 維持 投票、 京都 data 蓄積開始 |
| 5/12 (月) | NightlySanity + WeeklyReport (08:00) 確認 |
| 5/13 (火) | sib_*_exp v2 LIVE retro (Session #38 課題) |
| 5/14 (水) | V18 trial pre-flight check (V18_TRIAL_5_16_CHECKLIST.md) |
| 5/15 (木) | ★ JRA-VAN NEXT trial + RV trial 開始 ★ |
| 5/16 (金) | V18 trial 投入 (FRI 重賞 から、 上限 2,000円/日) |
| 5/17 (土) | V18 trial 本格運用 (ヴィクトリアM 含む) |

## 撤退ライン (絶対)

- 単日 ROI < 50% → 翌日 投票 中止
- 累計 -10,000 円 → yellow alert (Discord 通知)
- 累計 -50,000 円 → ★ halt ★ (即時 全停止)

現状 (5/9 朝): **+13,530 円** / 撤退余裕 **+63,530 円** ※ 当時 record、 5/16 P0-1 → 5/17 V15-audit-4 真値: **¥-6,920** / 撤退余裕 **¥43,080** (docs/V15_AUDIT_4_CUMULATIVE_ROI_5_17_2026.md)

## 困った時の参照

| 症状 | doc |
|------|-----|
| 朝予測 失敗 | [EMERGENCY_RUNBOOK_5_9_QUICK.md](EMERGENCY_RUNBOOK_5_9_QUICK.md) |
| schtask 動かない | [SCHEDULE_PREDICTION_PIPELINE.md](SCHEDULE_PREDICTION_PIPELINE.md) |
| Cookie 期限切れ | `python tools/refresh_cookie.py --auto` |
| Discord 通知来ない | `python tools/setup_discord.py` |
| 投票漏れ | 5/15 以降 JRA-VAN NEXT 自動分配 で解消 |

## 全体 status (5/9 21:00 時点)

- ✅ V15 production: 健全 (AUC 0.8939) ※ ★ 5/17 V15-audit-2 で 真値訂正: stored .pkl.auc 0.8939 は LGB train-set self-eval (in-sample LEAKY)、 genuine WF LGB+XGB 0.8678 / Grid 4-model 5-fold 0.8858 (docs/V15_AUDIT_2_WF_AUC_2026_05_17.md) ★
- ✅ schtasks 50 件: 健全 (Session #77 で silent_runner.vbs 修正済)
- ✅ Stage 2 通知: 5/16 復活予定 (dev/two-stage merge plan あり)
- ✅ 全馬 score: 5/10 から 自動収集
- ⏸ V18/V19: NO-GO 維持 (5/16 trial で 再判定)
- ⏸ V20: 学習 data 6 年分 ready (PoC AUC 0.8752)、 7/1 投入候補
- ⏸ V21 (動画): 5/15 RV trial 開始、 9/2 投入候補
- ⏸ V22 (RL): 10-12 月 設計、 12/1 100% 自動化候補

## 関連 doc

- [FULL_AUTOMATION_ROADMAP.md](FULL_AUTOMATION_ROADMAP.md) — 完全自動化 plan
- [MEMORY_INDEX.md](MEMORY_INDEX.md) — docs 全索引
- [PLAN_5_16_V18_TRIAL_FINAL_v5.md](PLAN_5_16_V18_TRIAL_FINAL_v5.md) — V18 trial plan
- [JRA_VAN_NEXT_TRIAL_5_15.md](JRA_VAN_NEXT_TRIAL_5_15.md) — 5/15 NEXT trial
- [RV_TRIAL_5_15_CHECKLIST.md](RV_TRIAL_5_15_CHECKLIST.md) — 5/15 RV trial
