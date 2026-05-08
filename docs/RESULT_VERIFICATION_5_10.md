# 5/10 朝 結果照合 plan (Session #42 D)

**作成**: 2026-05-08 (Session #42 D、 ユーザー仕事中)
**目的**: 5/9 V15 投資結果 → 5/10 朝 自動照合 → 5/16 GO/no-go 判定材料化

---

## 1. tool: tools/result_verification_5_10.py (新規)

### 1.1 機能

5/9 投資結果 を 自動集計し、 5/16 投入判定 (Session #42 H plan v2 §4) の verdict を返す。

### 1.2 入力

| source | 内容 | 取得 timing |
|--------|------|----------|
| `data/daily_predictions/20260509.csv` | V15 朝 予測 (08:00 自動生成) | 5/9 朝 |
| `data/daily_results/20260509.csv` | 5/9 結果 (18:00 自動照合) | 5/9 18:00 |
| `data/cumulative_results.csv` | 累計収支 | (随時) |

### 1.3 出力

```
data/v18/result_verification_5_10.json
{
  "date": "20260509",
  "v15_today": {
    "v15_full": {"n_races":..., "roi_pct":..., "profit":...},
    "v15_case_b": {"n_races":..., "roi_pct":..., "hit_rate_pct":..., "profit":...}
  },
  "cumulative": {"raw_cumulative_jpy":..., "retire_margin_jpy":...},
  "judge_5_16": {
    "verdict": "大成功/期待通り/微益/微損/損失/大損失",
    "go_probability_pct": 15-85,
    "recommendation": "..."
  }
}
```

### 1.4 利用例

```bash
# 5/10 朝 自動実行 (推奨 schtasks 追加)
python tools/result_verification_5_10.py --date 20260509

# テスト実行 (5/3 data で動作確認、 本 Session)
python tools/result_verification_5_10.py --date 20260503
# → V15 case B 6 races: ROI 87.62%, profit -520円, verdict 微損, GO 45%
```

---

## 2. 5/9 朝 + 5/10 朝 timeline

### 2.1 5/9 (土) 朝

```
05:00  PC ON
06:00  Keiba-NightlySanity (5/8 23:00 起動分) → 翌日 task pre-check
06:30  Keiba-FinalHealthCheck (Session #40 A4、 admin 推奨追加)
07:00  Keiba-MorningDigest
08:00  DailyPredict (V15 全レース、 約 10-15 min)
       → data/daily_predictions/20260509.csv 生成
08:45  RaceAutoNotify (戦略⑦ + 案B改 → Discord #bets / #investments)
09:00  予測結果 手動 確認 + 投票候補 list 確定
09:30  PAT login + 入金確認
10:00- レース毎 投票 (1勝 のみ、 700円 × max 3R = 2,100円)
14:50  multi_stage_predict.py race11_1450 stage 自動 trigger
15:45  multi_stage_predict.py race12_1545 stage 自動 trigger
18:00  DailyResults_Sat 自動 結果照合
       → data/daily_results/20260509.csv 生成
20:30  振り返り (data/v18/post_5_9_improvement_template.md)
22:00  ユーザー就寝
```

### 2.2 5/10 (日) 朝 (本 doc メイン)

```
05:00-08:00  PC ON 後 任意の時間
08:00       (任意) 自動起動なら schtasks Keiba-ResultVerification_5_10 で:
            python tools/result_verification_5_10.py --date 20260509
            → Discord #investments に verdict 通知

または ユーザー manual:
            $ cd C:\Users\takum\keiba-ai
            $ python tools\result_verification_5_10.py --date 20260509

→ 出力 確認 (verdict + GO 確率 + recommendation)
```

### 2.3 schtasks 追加候補 (admin、 推奨)

```cmd
schtasks /Create /TN "Keiba-ResultVerification_5_10" ^
    /TR "powershell -ExecutionPolicy Bypass -Command \"cd C:\Users\takum\keiba-ai; python tools\result_verification_5_10.py --date 20260509\"" ^
    /SC ONCE /SD 05/10/2026 /ST 08:00 /F
```

→ 5/10 08:00 自動実行、 Discord 通知。

---

## 3. 5/9 dry-run 推奨 step (5/8 夜)

### 3.1 multi_stage_predict.py の存在確認

```bash
# 既存 production tool
ls tools/multi_stage_predict.py 2>&1
```

### 3.2 dry-run (5/3 data で 動作確認)

```bash
# Session #38 で動作確認済み
python tools/daily_predict.py --date 20260503 --dry-run
# (production daily_predict は --dry-run option 確認後に)
```

### 3.3 5/9 当日 race info 想定

5/9 (土) 主要 race (推定):
- 11R 京王杯スプリングカップ (G2、 1400m、 G2 のため案B改 除外)
- 12R 1勝クラス (推奨 race、 700円 投資)

→ 案B改 上限 3R (推定 12R 各場の 1勝クラス) で 700×3=2,100円

---

## 4. 5/9 結果 → 5/16 判定 verdict 表

| profit | verdict | GO 確率 | recommendation |
|--------|---------|--------|--------------|
| ≥ +1,000 | 大成功 | 85% | V18 sib_exp 単独 trial 推奨 |
| +400~+1,000 | 期待通り | 75% | V18 sib_exp 単独 trial OK |
| 0~+400 | 微益 | 65% | V18 sib_exp 単独 trial 慎重 |
| -700~0 | 微損 | 45% | V15 単独継続 推奨 |
| -1,400~-700 | 損失 | 30% | V15 単独継続、 5/22 再判定 |
| ≤ -1,400 | 大損失 | 15% | V18/V19 NO-GO、 V15 単独継続 |

→ 詳細: [`docs/PLAN_5_16_V18_V19_DEPLOYMENT_v2.md`](PLAN_5_16_V18_V19_DEPLOYMENT_v2.md)

---

## 5. 5/9 V15 投資保護 (D 領域)

✅ result_verification_5_10.py は read-only 集計、 production 経路 影響なし
✅ V15 model file md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ predict_core / daily_predict / app.py / V15 model 不変
✅ schtasks 既存 task 不変 (新規追加は admin 推奨)

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ D1: tools/result_verification_5_10.py (新規、 約 200 行)
✅ D2: 動作確認 (5/3 data で test、 verdict 動作 OK)
✅ D3: 5/10 朝 schtasks 追加 plan (admin 推奨)
✅ D4: 5/9 verdict 6 シナリオ → GO 確率 + recommendation
✅ D5: V15 production 完全不変

→ **5/10 朝 結果照合 自動化 完了、 5/16 GO/no-go 判定 即可能**

---

**Session #42 D 完了**
