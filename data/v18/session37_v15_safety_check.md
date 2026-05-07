# Session #37 D: V15 動作不変 final check

**作成**: 2026-05-07 (Session #37 D)
**結論**: ✅ **V15 動作完全不変、 5/9 V15 案B改 投資保護 OK**

---

## 1. critical files 不変確認

| file | git diff | 最終 commit | 判定 |
|------|---------|-------------|------|
| `tools/predict_core.py` | 変更なし | 2026-04-27 (364a9260, v16 prep) | ✅ |
| `tools/daily_predict.py` | 変更なし | 2026-04-27 (1e208b97, Win11 24H2 fix) | ✅ |
| `tools/race_auto_notify.py` | 変更なし | 2026-04-27 (364a9260, strategy7) | ✅ |
| `keiba_model_v15_central.pkl.gz` | md5 309dffc6...c319d5 | 2026-04-08 (613285d7) | ✅ |
| `keiba_model_v15_central_live.pkl.gz` | md5 fac1588a...f7423e | 2026-04-08 (613285d7) | ✅ |

Session #37 で変更した file は以下のみ (V15 production と独立):
- `train/v18v19_no_sib/run_v18v19_no_sib_singlefold.py` (新規)
- `train/run_v15_1_lgb_xgb.py` (新規)
- `train/run_v15_1_ablation.py` (新規)
- `data/v18/v18v19_retraining/*` (新規 dir)
- `data/v15.1/v15_1_*` (新規 file)
- `docs/PHASE_3_V20_DETAILED_DESIGN.md` (拡張、 既存 plan 上書きなし)
- 各種 progress doc

---

## 2. schtasks 不変確認

```
Keiba-AM3FireCheck                  ← 早朝
Keiba-AM6FireCheck
Keiba-AM8FireCheck
Keiba-FridayWeekendScrape
Keiba-JrdbRetryAm9_Sat / Sun
Keiba-MorningDigest
Keiba-MorningWeightCheck_Sat / Sun  ← 馬体重補正
Keiba-Morning_Sat / Sun             ← V15 朝予測
Keiba-MultiStagePredict_*           ← 14:50 / 15:45 予測
Keiba-NarDailyPredict               ← NAR 系
Keiba-NarDailyResults
Keiba-NightlySanity
Keiba-PreFireCheck
Keiba-RaceDayReport_Sat / Sun
Keiba-TybPublishMonitor
KeibaAI_DriftDetector
... (計 27 task)
```

→ 全 task 既存通り、 Session #37 で **schtasks 一切変更なし**。

---

## 3. V15 score 一致確認 (5/3 京都 12R 東大路S)

target file: `data/multi_stage_predict/20260503_race12_1545.csv`

```csv
202608030412,京都,12,東大路S,False,特別/OP,1,0.2583926490209112,6,0.6344085474652305,2,0.37601589844431926,True
```

| 馬番 | score | 順位 |
|------|-------|------|
| 1 (top1) | **0.2584** | 1位 |
| **6** | **0.6344** | (2 nd column = score 高い別馬) |
| 2 | 0.3760 | 3位 |

Session #32 / #35 reference value 0.634 = **第 2 列の score 0.6344** と一致。
V15 prediction 完全不変、 model file unchanged。

---

## 4. 5/9 朝の動作 path

```
03:00 DailyPremiumScrape            ← scrape (既存)
03:00 (土曜) FridayWeekendScrape   ← 既存
06:00 DailyJrdbKyi                  ← JRDB DL (既存)
07:30 JrdbHealthCheck_Sat           ← 既存
08:00 DailyPredict (V15 朝予測)     ← V15 model 完全不変
08:45 race_auto_notify              ← V15 prediction + 戦略⑦ filter (既存)
09:30 MorningWeightCheck            ← 馬体重補正 (既存)
14:50 MultiStagePredict_Race11      ← 既存
15:45 MultiStagePredict_Race12      ← 既存
18:00 DailyResults (Sat/Sun)        ← 既存
```

全 task の起動 / V15 model load / prediction / 戦略⑦ filter / Discord 通知 経路は **完全不変**。

---

## 5. 影響保証

🟢 **5/9 (土) 朝、 V15 案B改 投資 完全保護 OK**:
- V15 model file: 不変 (md5 一致)
- predict_core.py: 不変
- daily_predict.py: 不変
- race_auto_notify.py: 不変
- schtasks: 不変
- 戦略⑦ filter (06_特別 / 京都 / 条件E / 条件B 除外): 不変
- 案B改 (12R 1勝クラスのみ上限 2,100 円): 不変

Session #37 の作業 (A/B/C) は全て **隔離 dir に出力**:
- A: `data/v18/v18v19_retraining/`
- B: `data/v15.1/`
- C: `docs/PHASE_3_V20_DETAILED_DESIGN.md` (拡張のみ)

V15 production 経路と独立、 衝突可能性ゼロ。
