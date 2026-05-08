# Sprint 2 C: running_style_change (Session #47 C)

**作成**: 2026-05-08 (Session #47 C、 dev/sprint2)

---

## 1. ★★ 超強 signal 発見 ★★

backtest 全期間 (jra_races_full、 528K races):

| 脚色 | code | n | top3 rate |
|------|------|---|----------|
| 逃げ | 0 | 38,127 | **49.42%** ★ 最高 |
| 先行 | 1 | 122,773 | 36.21% |
| 差し | 2 | 218,020 | 19.26% |
| 追込 | 3 | 149,345 | **6.57%** ★ 最低 |

→ **逃げ vs 追込 で +42.85pt 差** = 超強 signal

---

## 2. features (4 件)

```python
- running_style_recent_5r_mode (0-3、 過去 5R 最頻 脚色)
- running_style_recent_3r_mean (float、 過去 3R 平均)
- style_change_count (int、 過去 5R で異なる style 数)
- style_jockey_change_match (1/0、 騎手交代 + 脚色変化)
```

---

## 3. 期待 AUC contribution

V15 既存 features:
- `prev_pass4`: 前走 4 角通過順位 (生値)
- `bracket_pos`: 枠位置 (内/中/外)
→ 一部 capture 済

Sprint 2 C 追加:
- 過去 5R style sequence の category 化
- 変化 pattern (騎手交代相関)
→ AUC contribution +0.001-0.005 期待

---

## 4. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only

→ **5/9 朝 V15 完全保証**

---

**Session #47 C 完了 (dev/sprint2)**
