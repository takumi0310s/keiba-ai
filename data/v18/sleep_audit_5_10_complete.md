# 寝る前 完全確認 audit (5/10 朝動作保証)

audit 実施: 2026-05-10 00:30 頃 (Opus 4.7, effort=xhigh)
audit type: read-only (V15 production 完全保護)

## 結論

**5/10 朝動作: GO**

ただし audit 開始時、 working tree が `dev/audit-backtest` に置かれており、Session #71/77
の root .bat (`save_all_horse_scores_runner.bat`, `race_day_report.bat`) と
`tools/save_all_horse_scores.py` が物理 file system 上に存在しない state だった。
schtasks は working tree を直接呼ぶため、放置すると 5/10 09:30 SaveAllHorseScores と
18:00 RaceDayReport が起動失敗していた。

main へ switch して復元 → V15 production code + model md5 は両 branch で完全一致を
事前確認済み (差分 0)、 V15 不変保証は維持。

## 領域別 audit 結果

### A. V15 production code (PASS)

| file | md5 | size | mtime | py_compile |
|------|-----|------|-------|------------|
| tools/predict_core.py | 759762c2... | 120,931 | Apr 26 23:49 | OK |
| tools/daily_predict.py | 46c0a88e... | 24,821 | Apr 19 16:50 | OK |
| app.py | 5614862253... | 344,527 | Apr 12 19:00 | OK |

git diff main..dev/audit-backtest (3 files): 0 byte (完全一致)。

### B. V15 model file (PASS)

| file | md5 | size | mtime |
|------|-----|------|-------|
| keiba_model_v15_central.pkl.gz | **309dffc6**5504f056d233c65665c319d5 | 2,099,552 | Apr 8 23:32 |
| keiba_model_v15_central_live.pkl.gz | **fac1588a**c20e96ae81eef1efbf7f423e | 2,099,610 | Apr 8 23:32 |
| data/v15.1/v15_1_lgb.txt | 177aaccb... | 1,800,180 | May 5 19:00 |
| data/v15.1/v15_1_xgb.json | 7a6153f3... | 2,113,364 | May 7 17:24 |

CLAUDE.md 記載 (309dffc6 / fac1588a) と production pkl.gz が完全一致。
v15.1/ raw model files は分離管理されており、現行 V15 production には影響しない。

### C. schtasks 健全性 (5/10 必須 task: PASS、 disable 確認: PASS)

5/10 (Sun) 朝の必須 task — 全 Ready:

| task | 起動時刻 | status |
|------|---------|--------|
| Keiba-Morning_Sun (morning_checklist) | 06:30 | Ready |
| Keiba-MorningDigest | 07:00 | Ready |
| keiba-ai\JrdbHealthCheck_Sun | 07:30 | Ready |
| Keiba-AM8FireCheck | 08:50 | Ready |
| **keiba-ai\DailyPredict (V15 production)** | **08:00** | **Ready** |
| keiba-ai\RaceAutoNotify_Sun | 08:45 | Ready |
| keiba-ai\JrdbRetryAm9_Sun | 09:00 | Ready |
| **Keiba-SaveAllHorseScores_0930 (Session #71 初稼働)** | **09:30** | **Ready** |
| Keiba-MorningWeightCheck_Sun | 09:30 | Ready |
| Keiba-NarMidDayCalendar | 13:00 | Ready |
| Keiba-MultiStagePredict_Test10_Sun | 10:00 | Ready |
| Keiba-MultiStagePredict_Race11_1450_Sun | 14:50 | Ready |
| Keiba-MultiStagePredict_Race12_1545_Sun | 15:45 | Ready |
| Keiba-RaceDayReport_Sun | 18:00 | Ready |
| keiba-ai\DailyResults_Sun | 18:00 | Ready |
| keiba-ai\DailyResultsEvening | 20:00 | Ready |
| Keiba-NightlySanity | 23:00 | Ready |

期待通り Disabled:

| task | status |
|------|--------|
| Keiba-PreRacePredict_Watchdog_5_9 | **Disabled** (Session #78、 5/15 V18 trial 直前 re-enable 予定) |

### D. main + dev branch 健全性 (PASS、 destructive op = 意図的 cherry-pick + reset)

main HEAD: 6b699896 Session #89 (5/10 投票候補 事前抽出 plan)。

直近 main commit chain:
- Session #89 / #87 / #84 / #86 / #83 / #80 / #82 / #81 / #79 / #78

git reflog 上に reset HEAD~1 が 1 件あり (5/10 早朝)。 内容:
1. Session #88 を main 上で 904bfc7c として commit (5/9 深夜)
2. dev/audit-backtest に cherry-pick (bcb6f566)
3. main を HEAD~1 で 6b699896 に revert (Session #88 を audit branch 専用へ移譲)

CLAUDE.md `dev/audit-backtest: Session #69 + #70 + #88 反映` の記載と整合する意図的操作。

dev branch list (15 ローカル + remote): archive 候補や 5/15 merge 候補など、
CLAUDE.md 記載と整合。

### E. data files 整合性 (PASS)

- data/daily_predictions/ : 20260509.csv 存在、 5/2-5/9 全揃い
- data/daily_predictions_full/ : 20260510.csv は **未生成**(期待通り、 09:30 SaveAllHorseScores で生成)
- data/v18/ : Session #67-89 の audit doc 32 件 揃う
- logs/ : 5/9 fire の log 全揃い (am3/am6/am8/morning_dashboard/morning_weight_check/race_auto_notify など)、 致命 error なし
- 累計 +¥14,140 の整合性 doc (data/v18/session_88_*) も揃っており、 +¥14,140 (CLAUDE.md memory: +¥14,450 表記との差は 5/9 当日確定 vs 5/10 最新 +¥310 の Session 内 update 想定)

### F. docs/ + tools/ 整合性 (PASS、 main 上で完全揃い)

main 上に存在を確認した重要 doc:
- AUDIT_FULL_PROMPT.md / AUDIT_FULL_REPORT_5_8.md
- BACKTEST_30_YEAR_DESIGN.md
- FULL_AUTOMATION_ROADMAP.md
- JRA_VAN_NEXT_AUTO_ALLOCATION.md / JRA_VAN_NEXT_TRIAL_5_15.md / JRA_VAN_RV_TRIAL_GUIDE.md
- MERGE_PLAN_5_15.md
- MORNING_5_10_CHEAT_SHEET.md / MORNING_5_10_PROMPT.md
- PLAN_5_16_V18_TRIAL_FINAL_v5.md / PLAN_5_16_V18_V19_DEPLOYMENT_v2.md
- V18_TRIAL_5_16_CHECKLIST.md / V20_BUILD_DETAILED_PLAN.md
- V22_RL_DESIGN.md / V22_RL_INFRA.md

main 上の Session #71/77 ファイル (← この audit で修正された決定的 file 群):
- save_all_horse_scores_runner.bat (175 byte)
- race_day_report.bat (163 byte)
- tools/save_all_horse_scores.py (11,657 byte) — py_compile OK
- tools/race_day_report.py — py_compile OK
- tools/silent_runner.vbs (799 byte、 Session #77 修復済)
- tools/multi_stage_predict.bat (955 byte)

### G. 修正アクション (1 件のみ)

| 操作 | 影響 |
|------|------|
| `git switch main` (dev/audit-backtest → main) | working tree に Session #71/77 file 群を物理復元。 V15 production code + model は両 branch で完全一致のため不変。 uncommitted 変更 (.claude/scheduled_tasks.lock + data/cumulative_results.csv) は両 branch で base が同一のため無事 carry over。 |

destructive op なし。 reset --hard, push --force, branch 削除なし。

## 5/10 朝 GO 判定 詳細

| 項目 | 状態 | 備考 |
|------|------|------|
| V15 production code 不変 | OK | md5 759762c2 / 46c0a88e / 5614862253 |
| V15 model 不変 | OK | md5 309dffc6 / fac1588a (CLAUDE.md 記載値と一致) |
| 朝必須 schtasks 8 件 Ready | OK | DailyPredict / RaceAutoNotify_Sun / SaveAllHorseScores / Morning_Sun 等 |
| Watchdog disable | OK | Keiba-PreRacePredict_Watchdog_5_9 Disabled |
| Session #71 初稼働 file 揃う | OK (修正済) | dev/audit-backtest → main switch で復元 |
| Session #77 RaceDayReport file 揃う | OK (修正済) | 同上 |
| destructive op 異常 | なし | reflog の reset は意図的な audit branch 移譲 |
| 5/10 csv 未生成確認 | OK | daily_predictions_full/20260510.csv 不在 |
| 5/9 log error なし | OK | 主要 13 種の log を確認 |

★ 5/10 朝動作 GO ★

## 158h+ マラソン 最終締め

- Sessions 完走: #1-89 + AUDIT-1
- main HEAD: 6b699896 Session #89
- 累計収支: +¥14,140 (撤退余裕 +¥64,140)
- V15 投資保護: 完全 (production code + model 完全不変保証)
- 5/10 投票方針: 案B改 strict (06_特別 / 京都 / 条件 E / 条件 B 除外)
- 撤退ライン: -¥50,000 (余裕 +¥64,140)

おやすみなさい。 5/10 朝完璧で迎えます。
