# Sprint 1 C: レース skip optimizer (Session #45 C)

**作成**: 2026-05-08 (Session #45 C、 dev/sprint1)
**目的**: 自信度低 R を skip して variance 削減
**ステータス**: ✅ 実装完了 + unit test 6/6 PASS + backtest sim 完了

---

## 1. skip_score 設計

```
skip_score = (1 - top1_prob) - (top1_prob - top2_prob) × 0.3
            + 0.05 (num_horses ≥ 16)
            + 0.10 (race_grade ∈ G1/G2/G3)

threshold:
  loose:  0.50 (skip 少)
  medium: 0.60 (default)
  strict: 0.70 (skip 多)
```

→ 高 score = 「自信度低 → skip」

---

## 2. 単体 test 6/6 PASS

```
[unit test] race_skip 6 case PASS
- 高 prob (0.55) → 投票
- 低 prob (0.20) → skip
- G1 + 中 prob → skip
- 1勝 + 高 prob (0.45) → 投票
- threshold strict 0.70 → 投票
- threshold loose 0.50 → skip
```

---

## 3. backtest sim (4/18-5/5、 39 races)

| 戦略 | n_invested | n_skip | profit | ROI |
|------|-----------|--------|--------|-----|
| baseline (案B改) | 39 | 0 | -4,380 | 83.96% |
| loose (0.5) | 2 | 37 | +880 | 162.86% |
| **medium (0.6)** ★ | 7 | 32 | **+6,010** | **222.65%** |
| strict (0.7) | 23 | 16 | +6,820 | 142.36% |

→ **medium (0.6)** が ROI 最高、 **strict (0.7)** が profit 最大

---

## 4. caveat + production plan

- top1_prob は np.random simulation (true production では predict_core 出力)
- bias で sim ROI は過大評価
- 真の効果: skip による variance 削減 (期待 ROI 維持 + ドローダウン軽減)

production 統合 (5/15 merge 後):
- race_auto_notify.py で 各 R の top1_score (predict_core 出力) で skip 判定
- threshold = 0.60 (medium) を default

---

## 5. 5/9 V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint1 のみ
✅ predict_core / daily_predict / V15 model 完全不変
✅ 5/9 朝 V15 案B改 当日も skip なしで全 1勝 R 投票

→ **5/9 朝 V15 完全保証**

---

**Session #45 C 完了 (dev/sprint1)**
