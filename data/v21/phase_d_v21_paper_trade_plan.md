# Phase D-4 — V21 paper trade plan (6/1-6/30)

date: 2026-05-16
session: Terminal D (Phase D)

---

## 1. 目的

- V21 (V15 + 動画 30 features stacking) の **実 race ROI を 4 週で確認**
- V15 production と 並行 shadow 比較
- 6/30 GO/no-go 判定 → 7/1 V21 投入候補

---

## 2. 前提 (paper trade engine 既存資産)

| file | 役割 | 状態 |
|---|---|---|
| `tools/backtest_engine.py` (Phase 17) | 30 年 backtest engine、 actual logs と 戦略比較 | 動作確認済 (10 日 sample、 5/10) |
| `data/v22/paper_trade_*` | V22 RL paper trade 出力 | 既存、 V21 と並行可 |
| `data/v22/paper_trade_detail_all.csv` | 全 R 詳細 (V15 base) | 累積中 |
| `data/v22/paper_trade_summary_all.csv` | 戦略別 ROI 集計 | 累積中 |

→ V21 paper trade は **同じ engine を 拡張** し、 V21 score 列を 追記する形で実装。

---

## 3. 6/1-6/30 schedule (週次)

| 週 | 期間 | 内容 | 出力 |
|---|---|---|---|
| Week 1 | 6/1 (土)〜6/7 (金) | V21 paper trade **start** (shadow only、 投票なし、 score 記録のみ) | `data/v21/paper_trade_week1.csv` |
| Week 2 | 6/8 (土)〜6/14 (金) | Week 1 mid-eval、 重賞 sub-model 投入、 paper trade 継続 | `data/v21/paper_trade_week2.csv`、 `data/v21/eval_week1.md` |
| Week 3 | 6/15 (土)〜6/21 (金) | 06_平場 + 少頭数 sub-model 投入、 paper trade | `data/v21/paper_trade_week3.csv`、 `data/v21/eval_week2.md` |
| Week 4 | 6/22 (土)〜6/28 (金) | 投票 candidate 増加 trial (V21 復帰 R を 段階的に open) | `data/v21/paper_trade_week4.csv`、 `data/v21/eval_week3.md` |
| Judge | 6/29 (土)〜6/30 (日) | 4 週 統合 + 6/30 GO/no-go 最終 判定 | `data/v21/v21_go_nogo_report_6_30.md` |

---

## 4. 各週 詳細

### Week 1 (6/1-6/7) — shadow eval start

- V21 meta-LGB 学習 (使用 data: 2023-2025、 動画 coverage 必要)
- 重賞 R のみ V21 score を 算出、 投票なし (shadow 記録のみ)
- daily 出力: V15 score / V21 score / 結果 / 配当
- ★ V15 production 100% 並行運用、 投資額 0 変動 ★

### Week 2 (6/8-6/14) — 重賞 sub-model 投入

- 重賞 G1/G2/G3 で V21 sub-model 起動
- paper trade 詳細: bet=V21 で 仮想 投票 + ROI 試算
- mid-eval (6/14 日): V21 重賞 paper ROI / winner_top1 計算
- ★ GO 条件: 重賞 paper WF AUC ≥ 0.90 + paper ROI ≥ 130% (V21 限定) ★

### Week 3 (6/15-6/21) — 06_平場 + 少頭数 投入

- V21 06_平場 sub-model + 少頭数 sub-model paper trade 開始
- 戦略⑦ 除外 R の **paper 復帰 ROI** 検証
- mid-eval (6/21 日): sub-model 別 ROI / winner_top1 / shift

### Week 4 (6/22-6/28) — 投票 candidate 増加 trial

- 4 週 通算 投票 candidate 増加 仮想 trial
- V15 並行 比較 で **V21 vs V15 paper ROI 差** を 計算
- mid-eval (6/28 日): V21 - V15 ROI ≥ +10pt か?

### Judge (6/29-6/30) — 6/30 GO/no-go 判定

- 4 週 統合 報告
- GO/no-go 判定 (★ 全 criteria 達成 必須 ★)

---

## 5. ★ 6/30 GO/no-go 判定 基準 ★

| criterion | 閾値 | 計算式 |
|---|---|---|
| 1. WF AUC (V21 meta paper) | ≥ 0.90 | meta-LGB の 6/1-6/28 paper 期間 WF AUC |
| 2. paper ROI 全体 | ≥ 130% | 戦略⑦込み V21 (4 週 通算) |
| 3. V21 vs V15 paper ROI | V21 ≥ V15 + 10pt | 同期間 並行 比較 |
| 4. winner_top1 | ≥ V15 + 2pt | 1 着的中率 (paper、 同期間) |
| 5. shift | ≤ 12x | V21 (paper) の 上位 vs 結果順 shift 比 |
| 6. LEAK 監査 | PASS | 動画 features に POST-RACE leak ないこと verify |
| 7. fallback path 動作 | PASS | 動画 features 全 default で V21 = V15 完全一致 確認 |

→ ★ 1 つでも FAIL → no-go → V15 単独継続 ★ (V21 投入は 7/15 以降 に持ち越し)

---

## 6. cron / schtasks (paper 期間)

| task | timing | 状態 |
|---|---|---|
| V15 production (既存 DailyPredict / RaceAutoNotify) | 既存通り | ★ 不変 ★ |
| V21 paper shadow (新規) | 朝 09:00 / 夜 22:00 | 6/1 新規登録予定 (本 Phase D 範囲外) |
| paper trade 集計 (新規) | 毎晩 23:30 | 6/1 新規登録予定 |

★ V21 paper 期間中 も V15 schtasks は 1 行も変更しない ★

---

## 7. risk と 撤退 line

| risk | 対策 |
|---|---|
| V21 < V15 paper ROI が継続 | 6/30 no-go、 V21 投入 中止 (累計 +13,530 円 維持) |
| 動画 coverage 不足 (50% 未満) | sub-model 拡張 中止、 fallback path で V15 維持 |
| 動画 features に POST-RACE leak 発見 | LEAK 監査 で 即除外、 paper やり直し |
| paper trade engine bug | actual logs から 再生成 + V22 engine と相互 verify |

撤退 line (CLAUDE.md 投資保護 規程 不変):
- 累計 -50,000 円 (現在 +13,530 円、 余裕 +63,530 円)
- 単日 ROI < 50% / 累計 -10k / 累計 -50k の 3 段階

---

## 8. 期待値 (★ 想定であって 実測ではない ★)

paper trade 開始前 の 想定:

| metric | V15 (実測) | V21 paper (★ 想定 ★) |
|---|---|---|
| 全体 ROI | 119.2% (戦略⑦込み 140%+) | 130-150% |
| winner_top1 | ~31% | +2-5pt |
| 重賞 ROI | unknown (sample 少) | +20-40pt (動画 features の最大効能 領域) |

→ 実測は 6/29-6/30 で 確定。 想定 ≠ 達成 を 明確に。

---

## 9. fabrication 防止

- 本 doc の数値 (期待 AUC / ROI / +XX pt 等) は **設計時 想定**、 paper trade 実測ではない
- 6/30 判定後の 報告 file (`v21_go_nogo_report_6_30.md`) が **実測 SoT**
- V21 投入の 約束は **6/30 paper trade 結果 達成 後** にのみ発生

---

## 10. 完了条件 (本 Phase D 段階)

- [x] V15 inference review doc (`phase_d_v15_inference_review.md`)
- [x] V21 architecture design (`phase_d_v21_architecture_design.md`)
- [x] inference orchestrator skeleton (`tools/v21/predict_core_v21.py`)
- [x] 戦略⑦除外 R 対応 logic (`phase_d_strategy_7_excluded_handling.md`)
- [x] paper trade 統合 plan (本 doc)
- [x] skeleton compile + self-test PASS

次 step (6/1+、 本 Phase D 範囲外):
- [ ] V21 meta-LGB 学習 script 実装 (`train/train_v21_meta_lgb.py`)
- [ ] paper trade engine 拡張 (V21 score 列 追記)
- [ ] paper trade schtasks 新規登録 (6/1)
