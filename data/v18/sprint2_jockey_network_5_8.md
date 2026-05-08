# Sprint 2 E: jockey_network (Session #47 E)

**作成**: 2026-05-08 (Session #47 E、 dev/sprint2)

---

## 1. 構成

過去 3 年 (2022-2025、 74K races) の jockey co-occurrence graph:
- node: 騎手
- edge: 同 race 出走 (重み = 共出走回数)

```
graph 構築結果:
- jockeys: 248
- edges: 約 30,000
- races: 74K
```

---

## 2. features (4 件)

```python
- jockey_degree         # 出走回数 (node degree)
- jockey_top_partner_count  # 上位 10 共出走 partner 数
- jockey_top_partner_top3_rate_avg  # 上位 partner の top3 率 平均
- jockey_isolation_score  # 0=hub、 1=isolated
```

---

## 3. 上位 5 騎手 sample features (BT)

| 騎手 (degree 順) | degree | isolation | top3 率 |
|------|--------|-----------|---------|
| (上位騎手 1) | 3,XXX | 0.06 (hub) | 0.36 |
| (上位騎手 2) | 3,XXX | 0.11 | 0.41 |
| (上位騎手 3) | 3,XXX | 0.13 | 0.27 |

→ degree 高い騎手 = hub、 各 race で多様な partner と co-occur

---

## 4. caveat

- top_partner_count 計算に 一部 bug (logic 修正 5/15 merge 後)
- isolation_score は近似値 (degree / max_degree)
- 真の中心性 (betweenness、 pagerank) は networkx package 未 install のため簡易計算

→ Phase 3 後半 (5/16-6/8) で `pip install networkx` + 詳細化

---

## 5. 期待 AUC contribution

V15 既存 features:
- jockey_wr_calc (騎手 expanding 勝率)
- jockey_horse_top3r (騎手×馬)

network features は 新規 dimension で +0.001-0.003 期待 (Phase 3 で詳細実装後)

---

## 6. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only

→ **5/9 朝 V15 完全保証**

---

**Session #47 E 完了 (dev/sprint2)**
