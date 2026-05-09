# RL vs 既存 strategy 比較 plan

**作成**: Session #83
**比較期間**: 2026-11-01 〜 2026-11-30 (paper trade 30 日)
**用途**: 12/1 V22 投入 GO/NO-GO 判定

---

## 1. 比較対象 3 戦略

| # | 戦略名 | 構成 | 投票決定 |
|---|-------|------|---------|
| **A** | V20 + 案B改 strict | V20 ensemble + 7 点固定 (1-2-5)、 ¥700/R | rule-based 固定 |
| **B** | V20 + hybrid (Session #82) | V20 + 条件別動的 strategy | rule-based 動的 |
| **C** | V22 RL (完全 AI) | V20/V21 score → PPO → action | RL agent |

---

## 2. 比較 metrics

### 2-1. 収益性

| metric | 算出 | 目標 (V22 GO 条件) |
|--------|------|---------------------|
| **ROI** | 払戻 / 投資 - 1 | **≥ A + 5pt** AND ≥ B + 3pt |
| 月次 ROI 安定性 | std(daily_ROI) | A/B 同等以下 |
| 累計利益 (30 日) | sum(profit) | A/B より大 |
| 1 R 当たり 期待値 | mean(profit/R) | A/B より大 |

### 2-2. リスク

| metric | 算出 | 目標 |
|--------|------|------|
| **Sharpe ratio** | mean(daily_return) / std(daily_return) | **≥ 1.5** |
| **max drawdown** | 最大 連続 損失 | **≤ 15%** |
| 連敗 max | 連続外し 最長 | A/B 同等以下 |
| VaR 95% (1 日) | 5% tail 損失 | A/B 同等以下 |

### 2-3. 投票行動

| metric | A 案B改 | B hybrid | C RL |
|--------|--------|----------|------|
| 投票 R 数 / 日 | 3 R 固定 | 1-3 R | **動的 1-10 R** |
| 馬券種 | trio 7 点 | trio + uma | **8 種 全選択肢** |
| 1 R 投資 | ¥700 | ¥700-2,100 | **¥0-10,000 (RL 決定)** |
| skip 率 | 0% | 0-30% | **可変 (期待 30-60%)** |

---

## 3. paper trade 設定 (11 月)

### 3-1. 並行運用 logic

```
06:00 AM: V20/V21 prediction 実行 (production 共通)
06:30 AM: 各戦略の投票候補を計算
  ├─ A 案B改 strict (production = 実投票)
  ├─ B hybrid (paper)
  └─ C RL (paper)
22:00 PM: 結果照合 → 3 戦略 metrics 比較 → daily report
```

### 3-2. データ output

| ファイル | 内容 |
|---------|------|
| `data/v22_paper_compare_YYYYMMDD.md` | daily 3 戦略 metrics |
| `data/v22_paper_summary_2026_11.md` | 月次 summary (30 日合算) |

---

## 4. 統計的有意性 検定

### 4-1. ROI 差の検定

- **bootstrap CI 95%** (1,000 resample)
- 3 戦略 の ROI 95% CI を比較
- C (RL) の CI 下限 > A/B の CI 上限 → **有意に C 優位**

### 4-2. Sharpe ratio 比較

- Jobson-Korkie test (paired Sharpe ratio test)
- p < 0.05 で C > A/B 有意確認

### 4-3. sample size 評価

- 30 日 paper で N ≥ 30-100 R 想定 (RL は skip 多い可能性)
- N < 20 R なら判定 延期 (paper +14 日)

---

## 5. GO 判定 (V22 投入条件、 全 5 項目 PASS)

| # | 条件 | 閾値 |
|---|------|------|
| 1 | walk-forward backtest ROI (test 2024-2025) | ≥ 150% |
| 2 | paper 30 日 ROI | ≥ A + 5pt AND ≥ B + 3pt |
| 3 | Sharpe ratio | ≥ 1.5 |
| 4 | max drawdown | ≤ 15% |
| 5 | risk audit (V22_RISK_ANALYSIS.md) | PASS |

★ 5/5 PASS で 12/1 投入 ★
★ 4/5 → 再評価 (12/15 + paper +14 日) ★
★ 3/5 以下 → NO-GO (V20 + B hybrid または A 案B改 で運用継続) ★

---

## 6. NO-GO 時の対応

| 失敗 # | 対応 |
|--------|------|
| ROI 不足 | reward function 再調整、 12/15 再判定 |
| Sharpe 不足 | drawdown penalty 強化、 risk-aware reward 強化 |
| drawdown 過大 | Eighth Kelly cap 厳格化、 max ¥/R 縮小 |
| paper 不安定 | episode 単位 (R vs day) 再検討 |
| risk audit FAIL | 該当 risk 個別対応、 該当 action 制限 |

★ NO-GO でも V20 + 案B改 / hybrid で運用継続 (損失なし) ★

---

## 7. 比較 dashboard 出力 (毎日 22:00)

```
=== 2026-11-XX V22 paper compare ===

[ROI 30 日累計]
  A 案B改:    ¥XX,XXX (ROI: XX.X%)
  B hybrid:   ¥XX,XXX (ROI: XX.X%)
  C V22 RL:   ¥XX,XXX (ROI: XX.X%) ★

[Sharpe ratio]
  A: 1.2 / B: 1.4 / C: 1.7 ★

[max drawdown]
  A: 8.5% / B: 9.2% / C: 6.3% ★

[投票行動]
  A: 3 R 固定 / B: 2.1 R 平均 / C: 4.3 R 動的
  C skip 率: 35% / 高賭金回数: 12 / 低賭金回数: 23
```

---

## 8. 関連

- [V22_RL_DESIGN.md](V22_RL_DESIGN.md)
- [V22_RL_INFRA.md](V22_RL_INFRA.md)
- [V22_RISK_ANALYSIS.md](V22_RISK_ANALYSIS.md)
- [HYBRID_STRATEGY_DETAILED_PLAN.md] (Session #82 hybrid 詳細、 既存)
- [V20_VS_V15_COMPARISON.md](V20_VS_V15_COMPARISON.md) (V20 paper 比較 framework)
