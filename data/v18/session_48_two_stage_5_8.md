# Session #48 B: Stage 2 二段階予測 system

**作成**: 2026-05-08 (Session #48 B、 dev/two-stage)
**目的**: V15 朝予測 (Stage 1) + 各 R 70 分前 当日体重反映 (Stage 2)

---

## 1. 構成

```
Stage 1 (朝 08:00):
  V15 daily_predict (production、 不変)
  features: Tier 1 (過去成績 + 調教 + 血統)

Stage 2 (各 R 70 分前):
  ★ 当日体重 反映 ★
  features: Tier 1 + Tier 2 (当日体重 + 変化率)
  trigger: 各 R 65 分前 schtasks (5/16+ admin で追加)
```

---

## 2. 実装 file (3 件、 dev/two-stage)

| file | 内容 |
|------|------|
| `tools/two_stage_predict.py` | Stage 1 vs Stage 2 比較 + Discord 差分通知 |
| `tools/race_day_weight_collector.py` | 当日体重 取得 (JV-Link WF + netkeiba fallback) |
| `tools/race_day_weight_features.py` | 体重 features 6 件 計算 + backtest |

---

## 3. features (6 件)

```python
- current_weight (kg)
- weight_change_kg (前走比)
- weight_change_pct (%)
- weight_vs_3r_avg (kg)
- weight_vs_same_dist_avg (kg)
- weight_extreme_change_flag (±10kg 超)
```

---

## 4. backtest (jra_races_full、 472K races with prev_weight)

```
weight_num corr target:    +0.0155 (体重 重 → top3 微増 weak)
weight_change corr target: -0.0509 (体重 増 → top3 微減 ★ signal あり)
```

→ 体重 **減量** が top3 と弱い positive 相関 (信号 あり)
→ 体重変化 features は V15 学習 で追加候補

---

## 5. production trigger 設計 (5/16+ admin)

```cmd
# 各 R 発走 65 分前 trigger (土日)
schtasks /Create /TN "Keiba-Stage2_Predict" ^
    /TR "powershell -ExecutionPolicy Bypass -Command \"cd C:\Users\takum\keiba-ai; python tools\two_stage_predict.py --date %date:~0,4%%date:~5,2%%date:~8,2%\"" ^
    /SC HOURLY /MO 1 /F  # 簡略、 実際は時刻精細化
```

→ 5/9 では schtasks 追加なし、 5/16+ V18 trial 投入時に検討。

---

## 6. caveat + 制限

### 6.1 当日体重 source の現実性

- **JV-Link WF**: 32-bit Python venv 必要 (Session #41 A)、 5/16+ install 後利用可
- **netkeiba scraping**: Cookie 必要、 polling で BAN リスク

→ 5/9 では当日体重取得 logic は **deferred**、 5/16+ V18 trial 投入と並行検討

### 6.2 backtest の限界

- 体重変化 features は V15 既存に部分的に含まれる (weight_change 等)
- 真の AUC contribution は **production deploy 後** で確認

---

## 7. V15 投資保護

✅ V15 production 完全独立、 main 不変、 dev/two-stage only
✅ V15 model md5: 842b9a5f... 不変
✅ predict_core / daily_predict / app.py 完全不変

→ **5/9 朝 V15 完全保証**

---

## 8. 結論

✅ B1: tools/two_stage_predict.py (130 行) — Stage 1 vs Stage 2 比較
✅ B2: tools/race_day_weight_collector.py (110 行) — 当日体重 取得 design
✅ B3: tools/race_day_weight_features.py (140 行) — 6 features + backtest
✅ B4: backtest weight_change corr -0.0509 (signal あり)
✅ B5: production trigger は 5/16+ admin schtasks

→ **dev/two-stage 完了、 5/15 merge 候補、 5/16+ V18 trial と並行検討**

---

**Session #48 B 完了 (dev/two-stage)**
