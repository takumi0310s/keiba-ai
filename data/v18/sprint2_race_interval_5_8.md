# Sprint 2 B: race_interval_features (Session #47 B)

**作成**: 2026-05-08 (Session #47 B、 dev/sprint2)

---

## 1. ★ Strong signal 発見 ★

backtest 全期間 (jra_races_full、 472K races with prev):

| category | days | n_races | top3 rate |
|---------|------|---------|----------|
| 連闘 | 1-7 | 3,626 | **17.65%** (低) |
| 中1週 | 8-14 | 29,391 | 20.29% |
| **中2-4週 ★** | **15-28** | **155,215** | **24.71%** (最高) |
| 中5-8週 | 29-56 | 123,152 | 23.03% |
| 休み明け | 57+ | 161,135 | 20.48% |

→ **中2-4週 vs 連闘 で +7.06pt 差** = 強い signal

---

## 2. features (4 件)

```python
- days_since_prev_race (int)
- interval_category (0-4)
- interval_distance_interaction (category × distance bucket)
- interval_top3_rate_history (馬の同 category 過去 top3 率)
```

---

## 3. 期待 AUC contribution

中2-4週 vs 連闘 の +7pt 差は強い signal、 既存 V15 でも一部 capture 済 (rest_days)。
本 feature は category 化 + interaction で追加 +0.001-0.003 期待。

---

## 4. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only

→ **5/9 朝 V15 完全保証**

---

**Session #47 B 完了 (dev/sprint2)**
