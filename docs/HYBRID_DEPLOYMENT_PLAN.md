# hybrid 戦略 投入 plan (Session #82)

> Session #69 + Session #82 を踏まえた hybrid 戦略の段階的投入 schedule。
> 作成: 2026-05-09 (Session #82)

---

## 1. 投入 schedule (確定案)

| 時期 | event | hybrid 状態 | V20 状態 | 投資保護 |
|------|-------|------------|---------|---------|
| 5/9-5/15 | V15 案B改 strict | paper のみ | 未投入 | V15 production |
| 5/16 | V18 trial (1 day) | paper のみ | 未投入 | V18 上限 5,000円 |
| 5/17-6/30 | V15 + data 蓄積 | paper のみ | 未投入 | V15 production |
| 7/1 | ★V20 投入★ | paper のみ | 段階投入 (5,000円/日) | V20 監視 |
| 7/1-7/31 | V20 単独 1 ヶ月 | paper のみ | production | V20 単独評価 |
| 7/15 | hybrid backtest 拡張 (~400 R) | 評価 | production | — |
| 8/1 | hybrid 採用判定 | GO/no-go | — | — |
| 8/1+ (GO 時) | V20 + hybrid | **本投入** | + hybrid | 上限 1万円/日 |

---

## 2. 5/16 V18 trial との関係

### 2.1 hybrid は V18 trial に影響しない
- V18 trial (Session #74 plan v5) = V18 model 単独評価
- hybrid = 投票点数戦略 (model と直交)
- V18 trial 中も V15 case では 7 点 strict 維持

### 2.2 V18 trial データの hybrid backtest への寄与
- 5/16 1 day = ~12 R 追加
- cell 1 (芝 14 頭以下) 該当 ~6 R
- → hybrid backtest には微増 (大勢に影響なし)

---

## 3. 7/1 V20 投入時の hybrid 評価

### 3.1 V20 投入時 hybrid は paper のみ
理由:
- V20 + hybrid 同時で **効果切り分け不能**
- V20 単独 ROI 確認が最優先
- 1 ヶ月単独運用 → V20 baseline 確定

### 3.2 V20 単独評価 KPI (7/1-7/31)
- ROI 125-135% 確認
- WF AUC 維持 (V20 = 0.880+)
- LIVE retro shift <= 12x
- max DD < 30%

→ ALL PASS なら 8/1 hybrid 追加判定。

---

## 4. 8/1 hybrid 採用判定

### 4.1 採用条件 (5/5 件)
- [ ] V20 単独 1 ヶ月 ROI 125%+
- [ ] hybrid backtest cell 1 P(11>7) >= 75%
- [ ] cell 1 サンプル >= 200 R
- [ ] 撤退余裕 +¥40,000 以上
- [ ] Session #82 risk 分析 で red flag なし

### 4.2 採用時の投資額
- baseline V15 案B改 (現状): 7 点 × ~50 R/月 = ¥17,500/月
- V20 投入 (7/1+): 同水準 (上限 5,000円/日 strict)
- V20 + hybrid 8/1+: 芝 14 頭以下 cell で +57% 投資
  - cell 1 推定 25 R/月 × 1.57 = ¥13,750/月 → 全体 ¥21,250/月
  - 上限 1万円/日 strict 維持

---

## 5. 撤退条件 (hybrid 投入後)

### 5.1 即時撤退 (< 2 週間)
- hybrid 単独 ROI < 90%
- cell 1 で 連続 5 戦不的中
- 累計 -¥30,000 接近

→ hybrid 即時取消、 V20 単独に戻す。

### 5.2 段階撤退 (~ 1 ヶ月)
- hybrid 1 ヶ月 ROI < 100%
- cell 1 不調 (P(hybrid > strict) < 50%)

→ 8/末 再判定、 NG なら V20 単独継続。

---

## 6. リスク管理

- 投資金額増加 risk → 上限額 strict 維持 (1 日 1 万円)
- サンプル不足 risk → cell 1 で 200 R 達成まで投入しない
- over-fitting risk → bootstrap CI95 で確認、 P 値 厳守

---

## 7. 関連 doc

- `docs/STRATEGY_HYBRID_DESIGN.md` — 戦略確定案
- `docs/HYBRID_BACKTEST_PLAN.md` — backtest 設計
- `docs/HYBRID_RISK_ANALYSIS.md` — risk 分析

---

**結論**:
- 5/9-6/30: V15 case 単独 + data 蓄積
- 7/1-7/31: V20 単独投入、 hybrid は paper のみ
- 8/1: hybrid 採用判定 (5 件 ALL PASS なら GO)
- 8/1+: V20 + hybrid 本投入 候補
