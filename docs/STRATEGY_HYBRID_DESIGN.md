# hybrid 戦略 design (Session #82)

> Session #69 (7 vs 11 点 backtest) 拡張。 surface / 頭数別 differential 発見の活用 plan。
> 作成: 2026-05-09 (Session #82)

---

## 1. 背景 (Session #69 リファレンス)

Session #69 の 7 vs 11 点 backtest (280 R production score):

| 切り口 | 7 点 ROI | 11 点 ROI | delta | 判定 |
|--------|---------|----------|-------|------|
| 全体 | baseline | +2.28pt | 統計的同等 (P=56.2%) | NO-GO |
| 芝 | baseline | **+18.1pt** | 改善 | ✅ candidate |
| ダート | baseline | **-11.2pt** | 悪化 | ❌ 維持 |
| 15+ 頭 (条件 C) | baseline | **-11.4pt** | 悪化 | ❌ 維持 |

→ **一律 11 点は NG だが、 surface / 頭数別 hybrid は候補**

---

## 2. hybrid 戦略 確定案

### 2.1 投票 logic
```
if R == 重賞:
    skip (現状維持)
elif surface == "芝" and num_horses <= 14:
    bet 11 点 (★+18.1pt 改善期待★)
elif surface == "ダート":
    bet 7 点 (strict、 -11.2pt 悪化のため)
elif num_horses >= 15:
    bet 7 点 (strict、 -11.4pt 悪化のため)
else:
    bet 7 点 (default)
```

### 2.2 cell 分類

| cell | surface | 頭数 | 点数 | Session #69 effect |
|------|---------|------|------|-------------------|
| 1 | 芝 | <= 14 | 11 | +18.1pt ✅ |
| 2 | 芝 | >= 15 | 7 | -11.4pt 部分含む |
| 3 | ダート | <= 14 | 7 | -11.2pt 部分含む |
| 4 | ダート | >= 15 | 7 | -11.4pt + -11.2pt |

→ cell 1 のみ 11 点、 残り 3 cell は 7 点 維持。

---

## 3. 期待効果

### 3.1 ROI 試算

| 切り口 | 現状 (7 点) | hybrid | delta |
|--------|-----------|--------|-------|
| 芝 14 頭以下 | 125% | **143-145%** | +18-20pt |
| ダート (全頭数) | 125% | 125% | 0 |
| 芝 15+ 頭 | 125% | 125% | 0 |
| ダート 15+ 頭 | 125% | 125% | 0 |
| **全体平均** | **125%** | **130-135%** | **+5-10pt** |

### 3.2 累計 P/L 試算

仮定: 月 50 R 投票、 cell 1 (芝 14 頭以下) が 50% を占めると仮定。

- 現状: 月 +¥2,500
- hybrid: 月 +¥4,500-7,000
- 半年: +¥30,000-50,000 候補

→ 累計 +¥12,830 → 半年で +¥40,000-60,000 に到達候補。

---

## 4. 制約

### 4.1 サンプル不足
- Session #69 = 280 R 全体
- cell 1 (芝 14 頭以下) の cell サンプルは 280 × 50% = ~140 R
- 統計的有意性 確認には 200 R 以上 推奨

### 4.2 V18 trial との衝突
- 5/16 V18 trial (Session #74 で plan v5)
- V18 trial 中は hybrid 評価を保留
- V18 結果出るまで hybrid は paper trade のみ

### 4.3 V20 との同時投入 risk
- 7/1 V20 投入予定
- V20 + hybrid 同時で効果切り分け困難
- → V20 単独 1 ヶ月 → hybrid 追加 (Session #82 D 推奨)

---

## 5. 採用判定の前提条件

以下 5 件すべて 満たした時のみ hybrid 採用:

- [ ] cell 1 サンプル >= 200 R (現状 140 R、 6-7 月で達成)
- [ ] cell 1 P(hybrid > strict) >= 75%
- [ ] V20 単独 ROI 安定確認 (125-135%)
- [ ] 撤退余裕 >= +¥40,000
- [ ] Session #82 risk 分析 で red flag なし

---

## 6. 関連 doc

- `docs/HYBRID_BACKTEST_PLAN.md` — 拡張 backtest 設計
- `docs/HYBRID_DEPLOYMENT_PLAN.md` — 投入 plan
- `docs/HYBRID_RISK_ANALYSIS.md` — risk 分析
- Session #69 7 vs 11 点 backtest 結果 (history reference)

---

**結論**: hybrid 戦略は ★候補★ に格上げ。 7/1 V20 投入後、 1 ヶ月 V20 単独運用 → 8/1 hybrid 追加 判定。 5/9 〜 6/30 は data 蓄積期間。
