# Phase 4 voting design (Session #40 E4 試案)

**作成**: 2026-05-07 (Session #40 E4)
**対象期間**: 7月 V20 投入後 〜 9月 V21 (V20+動画) 投入候補
**目的**: 複数 model (V15 / V18/V19 / V20) の並列 voting 設計

---

## 1. 動機

V20 (Phase 3) が 7/1 投入 / V21 (Phase 4) が 9/1 投入候補。 7-8 月 voting で V15 + V18/V19 + V20 を並列実行し、 多様性で AUC boost を狙う。

---

## 2. アーキテクチャ

```
race i (出走表)
   │
   ├─ V15 model (4-model ensemble)  → score_v15
   ├─ V18 (単勝) / V19 (複勝) v2     → score_v18, score_v19
   └─ V20 (JRA+NAR 統合 4-model)    → score_v20
   │
   └→ voting layer
      ├─ method: "weighted"  (各 model の AUC 重み)
      ├─ method: "majority"  (top1 馬の合意)
      └─ method: "bayesian"  (posterior、 var-based)
   │
   └→ final_score → bet 生成 (戦略⑦ + 案B改 / V20 推奨)
```

---

## 3. voting 関数 (試作)

```python
def voting_score(
    scores_per_model: dict[str, dict[str, float]],
    aucs: dict[str, float],
    method: str = "weighted",
) -> dict[str, float]:
    """各馬 final_score を返す.

    scores_per_model: {model_name: {horse_id: score}}
    aucs:             {model_name: WF_AUC}
    method:           "weighted" / "majority" / "bayesian"
    """
    horses = set()
    for d in scores_per_model.values():
        horses |= set(d.keys())

    out = {}
    if method == "weighted":
        total_w = sum(aucs.values())
        for h in horses:
            score = sum(
                scores_per_model[m].get(h, 0.5) * aucs[m]
                for m in aucs
            ) / total_w
            out[h] = score
    elif method == "majority":
        top1_per_model = {
            m: max(scores_per_model[m], key=scores_per_model[m].get)
            for m in scores_per_model
        }
        # top1 が 2 以上の model で一致した馬 → boost
        from collections import Counter
        votes = Counter(top1_per_model.values())
        for h in horses:
            base = sum(scores_per_model[m].get(h, 0.5) for m in scores_per_model) / len(scores_per_model)
            boost = 0.05 * (votes.get(h, 0) - 1) if votes.get(h, 0) >= 2 else 0
            out[h] = base + boost
    elif method == "bayesian":
        # 各 model の score 分布の var を 重みに使う (信頼度 高 = var 低)
        import numpy as np
        for h in horses:
            ss = [scores_per_model[m].get(h, 0.5) for m in scores_per_model]
            # 各 model の信頼度 (= AUC^2 を proxy)
            ws = [aucs[m] ** 2 for m in scores_per_model]
            score = np.average(ss, weights=ws)
            out[h] = float(score)
    return out
```

---

## 4. method 比較

| method | 計算 | 強み | 弱み |
|--------|------|------|------|
| weighted | O(N) | 単純、 高 AUC model 重視 | 重み 推定誤差 |
| majority | O(N) | top1 合意で 安定 | 多様性 損失 |
| bayesian | O(N) | 信頼度 統合 | overfit risk |

→ Phase 4 では **weighted を 主軸**、 majority / bayesian は実験的

---

## 5. 期待効果

### 5.1 多様性 boost 試算

3 model の AUC 平均 0.88、 model 間の予測 相関 0.85 想定:
```
ensemble AUC ≈ 0.88 + 0.005 × (1 - 0.85)/0.5 ≈ 0.881-0.882
```

→ 単独 V20 0.880 → ensemble 0.882 = +0.002 改善 期待

### 5.2 ROI 影響

- BT ROI 140% → 142-145% (+2-5pt)
- 月利 +5-10 万円 → +5.5-11 万円

→ 多様性 boost は微増、 V21 (動画) との 比較で 採用判定

---

## 6. 実装 schedule (7-8 月)

| 期間 | 内容 |
|------|------|
| 7/1-7/14 | V20 production deploy + voting 実装 (weighted) |
| 7/15-7/21 | weighted vs single V20 の AUC / ROI 比較 BT |
| 7/22-7/31 | majority + bayesian 試作 |
| 8/1-8/15 | 3 method 比較、 best 選定 |
| 8/16-8/31 | best voting method を V21 (V20 + 動画) に統合 |
| 9/1 | V21 投入判定 + voting 採用判定 |

---

## 7. 採用判定 (9/1)

| # | 条件 | 必要値 |
|---|------|--------|
| 1 | voting AUC | ≥ V20 + 0.001 |
| 2 | voting LIVE retro winner_top1 | ≥ V20 + 0.5pt |
| 3 | 計算 latency | ≤ 30s/race (V20 単独 ~10s × 3 model 並列) |
| 4 | feature LEAK 監査 | PASS |

→ 全 PASS で voting + V21 採用、 1 つでも NG なら V20 単独 + V21 (動画) のみ

---

## 8. risk

| risk | mitigation |
|------|----------|
| model 間 相関 高すぎ | model diversification (V21 動画 が 大幅 model 異質化) |
| latency 増 | 並列 thread 実行、 GPU 共有 |
| voting 計算 自体 のバグ | unit test / WF 検証で早期検出 |
| AUC +0.001 の 微増 が production で消える | 控えめな 採用、 V20 単独 fallback 確保 |

---

## 9. V15 / V20 動作不変

本 voting は V20 投入後の追加 layer:
- V15 → V20 投入 (7/1) は voting 無しで先に実装
- voting は 7/15+ 段階的に追加、 V20 単独 fallback 保持
- V21 (動画) と voting は 9/1 同時 採用 OR 個別 採用

---

## 10. 結論

✅ 3-way voting (V15 + V18/V19 + V20) 設計完了
✅ method: weighted 主軸、 majority + bayesian 実験
✅ 期待 AUC +0.002 / ROI +2-5pt
✅ schedule: 7/1-9/1 段階的検証
✅ V20 / V21 fallback 保持

→ **Phase 4 voting 設計 完備**、 V21 動画 と 並行候補

---

**Session #40 E4 完了**
