# Session #69: 三連複 7 点 vs 11 点 backtest (★リーク完全防止★)

**作成**: 2026-05-09 (Session #69, ユーザー 5/9 終了 21:00 過ぎ)
**branch**: dev/audit-backtest (5 commits 追加)
**main 不変**: ✅ HEAD 8fc4e13b 保持
**V15 production 不変**: ✅ 完全 read-only audit

---

## 1. ユーザー指摘 → 実装

> 「今スコアを改めて出すとリークが入りそう」 (★完全に正しい★)

完全反映:
- ✅ V15 model 再 inference 一切なし
- ✅ data/v18/session69_horse_scores.csv (V15 final-model 再 inference) は LEAK と判定し不使用
- ✅ data/daily_predictions/ (production saved score) のみ使用
- ✅ top1-3 + top4-6 (trio_bets 出現回数で leak-free 復元) のみ評価
- ✅ top7-10 は使用せず (production 未保存 = 再 inference 必須 = LEAK risk)

---

## 2. backtest 設計

### 2.1 期間 + sample

- 期間: **3/14 - 4/25 (8 開催日、 280 R)**
- 累計 hours: 約 56 時間 (1 開催日 = 平均 7h)
- skip: 4/26 (結果未照合), 5/9 (5/9 21:00 未照合)

### 2.2 formation

| 名称 | 構成 | 点数 | 投資/R |
|------|------|-----|--------|
| 7 点 (現状 V15 baseline) | top1-{top2,3}-{top3-6} | 7 | 700 円 |
| 11 点 (本 Session) | top1 軸 box5 (10) + top2-top3-top4 (1) | 11 | 1,100 円 |

→ user prompt の "top4-10 11点" spec は LEAK risk のため変更
→ top1-6 のみ使用で 11 点 拡張 (代替 設計)

### 2.3 真配当 (jra_payouts.csv 由来)

- daily_results.csv の trio_payout は 7点 hit 時のみ非0 → counterfactual NG
- jra_payouts.csv (真の 三連複 配当) を join → 任意 formation の payout 算出可

---

## 3. 結果

### 3.1 メイン結果

| 指標 | 7 点 | 11 点 | Δ |
|------|------|-------|---|
| n_R | 280 | 280 | - |
| 投資 | 192,400 円 | 302,000 円 | +109,600 円 |
| 払戻 | 170,280 円 | 274,170 円 | +103,890 円 |
| 収支 | -22,120 円 | -27,830 円 | **-5,710 円** |
| ROI | 88.50% | **90.78%** | **+2.28pt** |
| hit | 76 (27.14%) | 97 (34.64%) | +21 (+7.50pt) |
| 平均配当 | 2,241 円 | 2,826 円 | +585 円 |
| EV/R | -79 円 | -99 円 | -20 円/R |

### 3.2 統計検定

- bootstrap n=2000, ΔROI 95% CI: **[-17.55, +24.24] pt**
- P(11点 > 7点 in ROI): **56.2%**
- → **CI が 0 跨ぎ → 統計的同等 (有意差なし)**

### 3.3 内訳

**条件別**:
- A (8-14頭/良-稍重): 11点 +6.0pt
- C (15+頭/良-稍重): 11点 **-11.4pt** ★ (多頭数で 7点 優位)
- D (1200-1400m): 11点 +6.5pt

**馬場別**:
- 芝 (131R): 11点 **+18.1pt** ★
- ダート (149R): 11点 **-11.2pt** ★

→ 大きな surface / 条件 別偏差あり

---

## 4. 結論 + 判定

### 4.1 結論

★ **11 点 投入 NO-GO 確定 (280 R 時点)** ★

判定 NG 5 項目:
1. ΔROI +2.28pt < 閾値 +5pt
2. CI95 lower bound -17.55pt < 0
3. P(11>7) 56.2% < 70%
4. EV/R Δ -20 円/R (悪化)
5. 投資効率 -5.2%

### 4.2 推奨 plan

| 期間 | 戦術 |
|------|------|
| 5/9 (今日) | 新潟 12R ¥700 (絶対遵守) |
| 5/16 | V18 trial (sib_exp GO 確率 50-65%) |
| 5/16 - 5/末 | V18 GO なら V18 7点運用、 NO-GO なら V15 7点継続 |
| 5/末 - 6/月 | +200-400R 蓄積、 surface 別 hybrid 検討 |
| 6/末 - 7/末 | 1,000R 累積で 再 backtest (最終結論) |
| 6/月以降 (V20) | production score を top10 まで保存 → 真の "top4-10 11点" 検証可 |

---

## 5. 5 commits (dev/audit-backtest)

| # | commit | 内容 |
|---|--------|------|
| 1 | 1d22172a | Session #69 A+B: data audit + leak-free 7vs11 backtest logic |
| 2 | ea008b23 | Session #69 C: 280R 期間限定 backtest 実行 + 真配当 join 修正 |
| 3 | 03e2fffd | Session #69 D: 投入判定 + 推奨 plan |
| 4 | (本 commit) | Session #69 E: doc 統合 + summary |

---

## 6. 投資保護 確認

| item | 値 |
|------|----|
| main HEAD | 8fc4e13b (不変) |
| V15 model file | 不変 (read-only) |
| predict_core.py | 不変 |
| daily_predict.py | 不変 |
| app.py | 不変 |
| schtasks 41 件 | 不変 |
| 5/9 投票方針 | 新潟 12R ¥700 (不変) |
| 累計収支 | +13,530 円 |
| 撤退余裕 | +63,530 円 (撤退ライン -50K) |

→ ✅ V15 production 完全保護
→ ✅ 5/9 投票 不変
→ ✅ ★ リーク完全防止 (production score 限定) ★

---

## 7. 関連ファイル

| file | 内容 |
|------|------|
| `data/v18/session_69_data_audit.md` | A: data source 確定 + リーク risk audit |
| `tools/sanrenpuku_7vs11_backtest_clean.py` | B: leak-free 7vs11 logic 実装 |
| `data/v18/session_69_per_race.csv` | C: 280 R 全 detail (raw) |
| `data/v18/session_69_metrics.json` | C: 集計 metrics + bootstrap CI |
| `data/v18/session_69_backtest_results.md` | C: 結果分析 (条件別/馬場別/date別) |
| `data/v18/session_69_recommendation.md` | D: 投入判定 + 推奨 plan |
| `data/v18/session_69_summary.md` | E: 本 doc (Session 全体 summary) |
