# 夜-2: N6 EV 目的変数 re-eval (final、 軽量化、 2026-05-17)

## 0. 結論

- **採用 verdict**: **NO-GO** (目的変数変更しない、 現行 top3 維持)
- **推奨目的変数**: **現行 top3_binary 維持**
- **過去 ROI 396.2% 真贋**: **leak ではない が INFLATED ESTIMATOR** (formula 起因の系統的過大)
- **V15 ensemble retrain は ★ 行わず ★ paper eval のみ** (CLAUDE.md 絶対遵守)
- **所要時間**: 約 4 分 (60 min hard limit の 6% 以内)

## 1. 3 目的変数 比較 (WF 3-fold 2023/2024/2025、 LGB 単独 軽量、 GPU)

LGB single (★ FT/IR 抜き ★、 num_boost_round=250 / early_stop=20、 leaf=63、 lr=0.07)。
AUC は全 target で **top3 binary を eval target に固定** (apples-to-apples)。

### 1.1 Full feature (145)

| 目的変数 | WF AUC mean | Brier | ROI_inflated (o1*o2*o3*20) | ROI_actual (実配当 mean) | Hit rate | LEAK 監査 |
|---|---|---|---|---|---|---|
| **top3 (V15 現行)** | **0.8686** | 0.1143 | 84.6% | **290.6%** | 17.2% | ✅ |
| win (1着) | 0.8590 | 0.1614 | 80.6% | 265.0% | 16.2% | ✅ |
| ev_weighted | 0.8655 | 0.1162 | 98.2% | 282.9% | 17.1% | ✅ |

**ev_weighted delta vs top3**: AUC **-0.0032**、 ROI_actual **-7.7pt**、 hit -0.1pt
→ ★ EV 重み付けは AUC・ROI 共に top3 を上回らない ★

**win delta vs top3**: AUC **-0.0096**、 ROI_actual **-25.6pt**
→ ★ 1 着 binary は trio 7 点には不向き (top3 識別力が落ちる) ★

### 1.2 Leak-free feature (137、 odds/pop 系 8 features 除外)

除外 features: `prev_odds_log, oz_tansho_base_log, oz_fukusho_base_log, oz_base_pop_rank, odds_change_rate, pop_rank_change, odds_sharp_drop, paci_ninki_idx`

| 目的変数 | WF AUC mean | ROI_inflated | ROI_actual | Hit rate |
|---|---|---|---|---|
| leakfree_top3 | 0.8667 | 80.9% | **299.5%** | 17.1% |
| leakfree_win | 0.8566 | 79.4% | 265.1% | 16.0% |
| leakfree_ev_weighted | 0.8643 | 94.5% | 295.1% | 17.4% |

### 1.3 LEAK 監査

| metric | full | leakfree | drop |
|---|---|---|---|
| top3 AUC | 0.8686 | 0.8667 | **+0.0019** |
| top3 ROI_actual | 290.6% | 299.5% | **-8.9pt (leak-free が高い)** |
| ev_weighted AUC | 0.8655 | 0.8643 | +0.0012 |
| ev_weighted ROI_actual | 282.9% | 295.1% | -12.2pt (leak-free が高い) |

**監査結果**: ✅ PASS
- odds/pop 8 features の AUC 寄与は **+0.0019 のみ** (極小)
- 実 ROI は leak-free の方が **高い** (人気馬偏重のバイアスが消えて配当が伸びる)
- V15 production が odds_log を含むのは Pattern B 側 (LIVE、 投票締切前で odds_log は固定 odds 既知前提) → リーク扱い ではない が、 学習データの過剰適合 risk あり

## 2. 過去 ROI 396.2% (Phase 2、 3 月) 真贋

### 2.1 出所
- File: `tools/validation_2_target_variable.py` (3 月実行)
- ROI 計算 source: `backtest_central_leakfree.estimate_payouts()` の
  `trio_pay = max(100, int(o1 * o2 * o3 * 20))` formula
- Same eval で:
  - place_current (現行 target): **368.8%**
  - ev_weighted: **396.2%**
  - delta = **+27.4pt** (= +7.4% relative)

### 2.2 真贋判定: **INFLATED ESTIMATOR** (leak でも fabrication でもない、 formula 起因の系統的過大)

| 根拠 | 詳細 |
|---|---|
| Phase 2 内部整合 | place_current の同 estimator が 368.8%、 ev_weighted advantage は +27.4pt のみ。 「396% という絶対値」が EV 重みの効果ではなく formula 起因の inflation |
| CLAUDE.md Phase 2b/3 既知 | "推定 ROI 式 `o1*o2*o3*20` が実配当の **約 2x**" (CLAUDE.md "過去の失敗から学んだ教訓" 表) |
| Phase 3 実配当 検証 | trio 真 ROI = **225.8%** [CI 198-264%] → 368.8% / 1.63 ≒ 226 で説明可、 396.2% も同比率で実は ~243% 相当 |
| 本 eval cross-check | top3 で roi_inflated=84.6% vs roi_actual=290.6% (= 3.4x 関係、 ただし買い目 logic 違うので絶対比は使えない)。 ev_weighted の inflated→actual 比は ほぼ同等 (98.2 → 282.9 = 2.88x) → **inflation は target 間で systematically 同方向**、 ev 加重 で advantage 反転しない |

### 2.3 真の ev_weighted の advantage

- Phase 2 報告: +27.4pt (inflated)
- 本 eval (実配当): **-7.7pt** (ev_weighted が劣る)
- → ★ 「ev_weighted が +27pt 良い」 は formula bias の artefact、 実配当では top3 と互角 か やや悪い ★

## 3. 採用判定 (CLAUDE.md ベースライン gate)

| 条件 | 値 | 判定 |
|---|---|---|
| AUC 維持 0.8939 ± 0.001 (V15 ベースライン) | ev_weighted の delta vs top3 = -0.0032 (本 eval scale)、 production scale でも 悪化 想定 | ❌ |
| paper ROI 改善 | ROI_actual delta = -7.7pt | ❌ |
| T4 LEAK 監査 gate PASS | full vs leak-free AUC drop 0.0019、 leak-free の ROI が同等以上 | ✅ |
| 全条件 PASS で GO | 3/3 必要、 実際 1/3 | ❌ |

**最終 verdict**: **NO-GO**
- 目的変数を ev_weighted / win に変える根拠なし
- 現行 top3_binary の維持を推奨
- V15 ensemble retrain (FT/IR 込み) は本 eval では実行しない (paper eval のみ、 CLAUDE.md 絶対遵守)

## 4. eval の制約と honest 補記

| 項目 | 制約 |
|---|---|
| Architecture | LGB single (V15 は 4-ens: LGB+XGB+FT+IR)。 絶対値 (0.8686) は V15 production (0.8939) と直接比較不可 |
| Fold 数 | 3 (2023/2024/2025) のみ。 V15 は 6-fold (2020-2025)。 60 min budget 厳守のため短縮 |
| 買い目 logic | 単純 pred top3 == actual top3 (順不同)。 V15 production は 7-point trio formation で hit rate 高い (= 絶対 ROI も高い)。 **絶対値 比較不可**、 **target 間 relative delta のみ 信頼** |
| ROI_actual の倍率 bias | jra_payouts.csv は course mojibake のため date+race_num で 10x mean merge → absolute は粗い。 ただし target 間 比較には十分 (全 target で同じ bias) |
| Boost rounds | num_boost_round=250 / early_stop=20 (V15 は 1000 / 50)。 軽量化で fold あたり 11s に圧縮 |

## 5. 5/24+ 次 phase 候補

ev_weighted の本格検証が必要なら 別 phase で:
- **P2-1 v15.2 fine-tuning** (★ 夜-3 結果 依存 ★) で V15 architecture (4-ens、 6-fold、 1000 round) のまま target だけ ev_weighted に変更して再学習
- ★ ただし本 eval で advantage マイナス確認のため 優先度 低 ★

## 6. 出力ファイル

- `data/subtask16_ev_eval_final.json` (eval 詳細、 36 数値)
- `data/subtask16_ev_eval_ckpt.json` (fold-level checkpoint)
- `logs/subtask16_ev_eval_final.log` (実行 log、 全 fold AUC/ROI 数値)
- `tools/subtask16/eval_ev_objective.py` (eval script、 軽量化 final)
- 本 doc

## 7. 完了 status

- 60 min hard limit: ✅ (実 ~4 分)
- fold-level checkpoint: ✅ (data/subtask16_ev_eval_ckpt.json)
- partial 出力 ready: ✅ (incremental save、 各 target で JSON 更新)
- V15 production .pkl.gz / cumulative_results.csv 改変: なし
- git commit / push: なし (★ 親集中 ★)
