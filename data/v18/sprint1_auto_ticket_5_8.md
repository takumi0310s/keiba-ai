# Sprint 1 B: 馬券種類 自動選択 (Session #45 B)

**作成**: 2026-05-08 (Session #45 B、 dev/sprint1)
**目的**: 単勝/複勝/馬連/3連複 の EV 計算 + max EV 自動選択
**ステータス**: ✅ 実装完了 + 動作確認

---

## 1. 設計

### 1.1 計算式

```
EV = sum(P(combination) × payout(combination)) / cost
cost = 投票点数 × 100 円
min_ev = 1.0 (期待値 1.0 未満は skip)
```

### 1.2 JRA 控除率 (還元率)

| 券種 | 還元率 |
|------|------|
| 単勝/複勝 | 80.0% |
| 馬連/ワイド/馬単 | 77.5% |
| 3連複/3連単 | 72.5% |

### 1.3 実装 ticket types

```python
- tansho:  1 点 (top1) — 100 円
- fukusho: 1 点 (top1) — 100 円
- umaren:  2 点 (top1+top2, top1+top3) — 200 円
- trio:    7 点 (案B改 baseline、 TOP1 軸 - TOP2,3 - TOP2-6) — 700 円
```

---

## 2. 動作確認 (test sample)

```python
probs=[0.45, 0.25, 0.15, 0.08, 0.04, 0.03]
odds=[3.5, 6.0, 9.0, 18, 30, 50]
odds_fuku=[1.5, 2.0, 2.8, 4.0, 6.0, 8.0]
odds_umaren={(0,1): 8.5, (0,2): 12.0}
odds_trio={(0,1,2): 25, (0,1,3): 60, (0,2,3): 100}

→ select_best_ticket() 結果:
   1. trio:    EV 3.00 ★ best (3 点 / 300 円)
   2. umaren:  EV 1.77 (2 点 / 200 円)
   3. tansho:  EV 1.58 (1 点 / 100 円)
   4. fukusho: EV 1.15 (1 点 / 100 円)
```

→ 高 prob (top1=0.45) + 高 odds (trio 25-100) で **trio が最高 EV**
→ 案B改 三連複 7 点 が 経験的に最良 と一致

---

## 3. backtest 設計 (5/15 merge 後 production 検証)

実 production では:
- predict_core 出力: top1-6 prob
- netkeiba 直前オッズ: 単勝 / 複勝 / 馬連 / 3連複
- レース毎に 4 ticket type の EV 計算
- max EV を選択 (現状 trio 700 円固定)
- min EV 1.0 未満は **skip** (新機能)

---

## 4. caveat

- P(馬連 / 3連複) は 簡易計算 (p_a × p_b × 2 等)
- 実際の prob は race 内 dependence 考慮要 (top1 馬が top2 入る prob は p_a × p_b より高い)
- production では top1_p3 (3 着以内) を 別途 predict (V19) で精緻化推奨

---

## 5. production 統合 plan (5/15 merge 後)

```python
# tools/race_auto_notify.py で 既存 trio 7 点 を overlay
from tools.auto_ticket_selector import select_best_ticket

best = select_best_ticket(probs=top_probs, odds=tansho_odds,
                          odds_umaren=umaren_dict, odds_trio=trio_dict,
                          min_ev=1.0)
if best['ticket'] == 'skip':
    return None
# 案B改 baseline と best を比較、 ROI 改善見込みあれば自動切替
```

---

## 6. 5/9 V15 投資保護

✅ V15 production 完全独立、 main 不変、 dev/sprint1 only
✅ 5/9 朝 案B改 (trio 7 点 700 円) 完全維持

→ **5/9 朝 V15 完全保証**

---

**Session #45 B 完了 (dev/sprint1)**
