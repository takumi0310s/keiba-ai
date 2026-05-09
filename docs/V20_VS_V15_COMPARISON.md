# V20 vs V15 比較 framework

**作成**: Session #79
**比較期間**: 2026-06-08 〜 2026-06-30 (paper trade 30 日)
**用途**: 7/1 V20 投入 GO/NO-GO 判定

---

## 1. 比較 metrics

### 1-1. AUC (overall + class 別)

| metric | V15 baseline | V20 GO 条件 |
|--------|-------------|------------|
| overall WF AUC | 0.886 | **≥ 0.895** (+9bp) |
| 2025 単年 AUC | 0.8851 | ≥ 0.895 |
| class A | (測定要) | V15 同等以上 |
| class B | (測定要) | V15 同等以上 |
| class C | (測定要) | V15 同等以上 |
| class D | (測定要) | V15 同等以上 |
| class E | (測定要) | V15 同等以上 |
| class X | (測定要) | V15 同等以上 |

### 1-2. hit rate (paper trade)

| metric | V15 実績 | V20 GO 条件 |
|--------|---------|------------|
| trio 7 点 hit rate | 33-50% (条件別) | V15 同等以上 |
| winner_top1 (R 内 1 位 当たり率) | 32.8% | **≥ 33%** |
| top3 inclusion rate | 65-72% | V15 同等以上 |

### 1-3. ROI (paper trade、 仮想 ¥)

| 期間 | V15 paper (並行) | V20 paper | GO 判定 |
|------|----------------|-----------|---------|
| 6/8-6/14 (week 1) | 記録 | 記録 | 観察 |
| 6/15-6/21 (week 2) | 記録 | 記録 | 中間判定 |
| 6/22-6/28 (week 3) | 記録 | 記録 | 中間判定 |
| 6/29-6/30 (week 4 部分) | 記録 | 記録 | **最終判定** |
| 全期間 30 日 ROI | baseline | **≥ V15 + 5pt** | GO |

### 1-4. shift (LIVE 推論時の bias)

| metric | V15 baseline | V20 GO 条件 |
|--------|-------------|------------|
| BT → LIVE shift | 8-12x | **≤ 12x** |
| BT vs LIVE AUC delta | -2pt 以内 | -3pt 以内 OK |

---

## 2. 比較期間 (6/8-6/30)

### 2-1. paper trade 設定

| 項目 | V15 (継続) | V20 (paper) |
|------|-----------|-------------|
| 投票 strategy | 案B改 7 点 | 案B改 7 点 |
| 投資 / R | ¥700 | ¥700 (仮想) |
| max / 日 | 3 R | 3 R |
| 戦略⑦ | 適用 | 適用 |
| 実投票 | YES (production) | NO (paper) |

### 2-2. 並行運用 logic

```
06:00 AM: 両 model で予測実行 (paper)
08:00 AM: V15 結果 → Discord (production)
08:00 AM: V20 結果 → log (paper、 通知なし)
22:00 PM: 結果照合、 V15 vs V20 metrics 比較 → daily report
```

---

## 3. 統計的有意性検定

### 3-1. AUC 差の検定

- **DeLong test** (paired AUC test)
- p < 0.05 で V20 > V15 有意差確認

### 3-2. ROI 差の検定

- **bootstrap CI 95%** (1,000 resample)
- V20 ROI 95% CI が V15 ROI 95% CI を上回れば GO

### 3-3. sample size 評価

- 30 日 paper で N ≥ 60 R 想定
- N < 30 R なら判定 延期 (paper +14 日)

---

## 4. GO 判定 (V20 投入条件、 全 5 項目 PASS)

| # | 条件 | 閾値 |
|---|------|------|
| 1 | WF AUC | ≥ 0.895 |
| 2 | LIVE retro winner_top1 | ≥ 33% |
| 3 | shift (BT → LIVE) | ≤ 12x |
| 4 | paper ROI 30 日 | ≥ V15 + 5pt |
| 5 | LEAK 監査 | PASS (Session #51 + KKA) |

★ 5 項目 ALL PASS で 7/1 投入 ★
★ 1 項目でも FAIL なら 7/15 再判定 (paper +14 日) ★

---

## 5. NO-GO 時の対応

| 失敗 # | 対応 |
|--------|------|
| AUC 不足 | hyperparameter 再調整、 7/15 再判定 |
| winner_top1 不足 | shift 原因特定、 LIVE retro 拡張 |
| shift 過大 | LIVE feature 不一致 audit、 BT/LIVE diff 調査 |
| ROI 不足 | 投票 strategy 微調整、 戦略⑦ 拡張検討 |
| LEAK | 該当 feature 完全除外、 再学習 |

★ NO-GO でも V15 production 継続 (損失なし) ★

---

## 6. 比較レポート出力 (毎日 22:00)

```
data/v20_paper_compare_YYYYMMDD.md
- daily summary (V15 vs V20 hit / ROI / pred 一致率)
- weekly summary (週次 cumulative)
```

---

## 関連

- [V20_BUILD_DETAILED_PLAN.md](V20_BUILD_DETAILED_PLAN.md)
- [V20_DEPLOYMENT_CHECKLIST.md](V20_DEPLOYMENT_CHECKLIST.md)
