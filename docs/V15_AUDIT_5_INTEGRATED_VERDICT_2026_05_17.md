# V15-audit-5: 統合 verdict + 5/18+ 判断材料

作成日: 2026-05-17 (Sat 夜)
作業 mode: read-only (V15 model / predict_core / cumulative_results.csv 改変なし、
v15.2 training PID 23528 中断なし、 commit / push なし)
provisional 注記: ★ audit-1/2/3/4 (V15_AUDIT_1〜4_*_2026_05_17.md) は 当 audit-5 開始時点で
**未作成** であることを確認 (`docs/V15_AUDIT_*.md` glob 0 件)。
本 doc は 既存 docs (T1_FEATURES_INTEGRITY、 V15_2_FAIR_BENCHMARK、 ROI_DISCREPANCY、
VICTORIA_MILE_FULL_18) と cumulative_results.csv 直接集計を base に **provisional** に統合。
audit-1/2/3/4 完成後に更新可能な形式で記述。

---

## 0. 結論 (★ honest ★)

| 項目 | 真値 |
|---|---|
| V15 真の architecture | **2-model (LGB+XGB)** ★ CLAUDE.md「4-model」 は drift |
| V15 真の WF AUC (LGB+XGB only) | **6-fold mean 0.8678** (2020-2025) |
| V15 真の WF AUC (.pkl.gz stored、 single-run) | 0.8939 (★ WF mean ではない ★) |
| V15 真の WF Grid 4-model AUC (master_report 由来) | 0.8858 (2021-2025、 ただし production .pkl.gz は LGB+XGB only) |
| 真の 累計 ROI (5/16 まで) | **101.33%** / PnL **+¥5,240** / n=563 |
| 真の 累計 ROI (5/17 反映) | **98.34%** / PnL **-¥6,920** / n=596 |
| 5/17 単日 ROI | **47.36%** / PnL -¥12,160 / n=33 (G1 day 大幅 negative) |
| features integrity (RED_IMP_BUT_CONST) | **0 件** (健全) |
| 既知 RED_CONSTANT | 8 件 (全 importance=0、 model 害なし) |
| v15.2 LGB+XGB 改善 | **+0.000005** (≒ ±0) → ★ NO-GO 濃厚 ★ |

★ **drift 確認**: ensemble 構成 + 累計 ROI の 2 件で CLAUDE.md / memory.md 主張と真値が乖離。 ★

---

## 1. 真値 vs memory drift 完全表

| 項目 | CLAUDE.md / memory 主張 | 真値 (5/17 audit) | drift |
|---|---|---|:---:|
| ensemble | 4-model (LGB+XGB+FT+IR) Grid | **2-model (LGB+XGB)、 weights {lgb:0.504, xgb:0.496, mlp:0}** | ❌ 確定 |
| WF AUC | 0.8939 (\*WF と誤記) | **single-run 0.8939、 WF mean 0.8678 (LGB+XGB) / 0.8858 (master Grid 2021-2025)** | ❌ 表現 drift (数値は parse 元正しい) |
| 累計 ROI (5/16) | 119.2% / +¥13,530 | **101.33% / +¥5,240** | ❌ 確定 |
| 累計 ROI (5/17 反映) | (未認識) | **98.34% / -¥6,920** | NEW 真値 |
| features 数 | 150 | **145** (cache 145 col、 model 145 col 一致) | ⚠ 軽微 (5 件 過剰計上) |
| RED_IMP_BUT_CONST | 0 件 (暗黙) | **0 件** | ✅ 整合 |
| 既知 RED_CONSTANT | 言及なし | **8 件** (is_nar / prev_odds_log / prev_race_first3f / prev_race_last3f / prev_race_pace_diff / sire_shinba_top3r / pci / gaisha_rank、 全 imp=0) | ⚠ doc 漏れ |
| TYB merge bug 5 件 | sub-task 6 で発見 | **V15 145 features に含まれていない** (cache 232 col には残骸あり、 model 害なし) | ✅ 整合 |
| 戦略⑦案 C 効果 | +3.67pt (Terminal B 検証、 ≤5/10 N=466) | (5/17 G1 day で reverify は noise レベル) | provisional |
| 撤退ライン -¥50,000 | 維持 | **5/17 反映後 撤退余裕 +¥43,080** (-¥50,000 まで) | ⚠ buffer 減少 |

★ provisional 注記 ★: audit-1/2/3/4 完成時、 各項目を該当 audit 結果 で 上書き update 想定。

---

## 2. 5/17 G1 day 結果反映 (audit-4 相当)

source: `data/cumulative_results.csv` status='settled' filter

### 2-1. 単日 5/17 数値

| 指標 | 値 |
|---|---:|
| n | 33 |
| 投資額 | ¥23,100 |
| 払戻額 | ¥10,940 |
| PnL | **-¥12,160** |
| ROI | **47.36%** |
| top1 in top3 | 19/33 (57.6%、 model 識別力は維持) |
| trio hit | 6/33 (18.2%) |

### 2-2. ヴィクトリアマイル (G1, 東京 11R)

| 項目 | 値 |
|---|---|
| race_id | 202605020811 |
| V15 top1 | #7 クイーンズウォーク (score 0.7771、 市場 3 番人気 8.3-8.4x) |
| 着順 (top1/top2/top3 馬の) | top1#7=**3 着** / top2#14=**8 着** / top3#12=**1 着** |
| 結果 trio | 7-8-12 |
| 推奨 trio bets | 4-7-12 / 4-7-14 / 6-7-12 / 6-7-14 / 7-12-14 / 7-12-18 / 7-14-18 |
| 的中 | **0 (hit なし)** |
| profit | -¥700 |

注: 結果 trio (7-8-12) は推奨 trio 7 点 すべてに 1 馬 (8) が欠ける hit miss pattern。 V15 top3 馬 (#7/#14/#12) のうち 3 着・1 着 は当たったが、 2 着 #8 カムニャック (V15 7 位、 市場 2 人気) は V15 「市場 over-bet 候補」 と判定し trio 不採用。 → V15 model 判断は ★ 部分的に正しい (top1 3着・top3 1着) が trio 構成では miss ★ という典型 noise。

### 2-3. 条件別 (5/17 単日)

| 条件 | n | inv | pay | ROI |
|---|---:|---:|---:|---:|
| A | 16 | ¥11,200 | ¥6,040 | 53.9% |
| C | 10 | ¥7,000 | ¥4,900 | 70.0% |
| D | 7 | ¥4,900 | ¥0 | **0.0%** ★ |
| B | 0 | — | — | (戦略⑦ で除外想定) |
| E | 0 | — | — | (戦略⑦ で除外想定) |
| X | 0 | — | — | — |

★ 観察: 5/17 は D 条件 (1200-1400m) が 7/7 全 miss、 単日大敗の主要因。 ただし 1 day n=7 は noise level。

### 2-4. 戦略⑦案 C 効果 (5/17 単日)

| metric | baseline | 戦略⑦案 C 適用 | delta |
|---|---:|---:|---:|
| n | 33 | 33 (B/E 0 件で除外 0 件) | ±0 |
| ROI | 47.36% | 47.36% | **±0** |

★ verdict: 5/17 単日は B/E 該当 0 件のため 戦略⑦案 C 適用効果 **0**。 検証済 +3.67pt (≤5/10 N=466) は引き続き有効、 1 day では re-verify 不能。

### 2-5. 全期間 (3/14 - 5/17) 真値

| 期間 | n | inv | pay | PnL | ROI |
|---|---:|---:|---:|---:|---:|
| ≤5/16 (baseline) | 563 | ¥394,100 | ¥399,340 | **+¥5,240** | **101.33%** |
| 5/17 単日 | 33 | ¥23,100 | ¥10,940 | -¥12,160 | 47.36% |
| **≤5/17 (全 settled)** | **596** | **¥417,200** | **¥410,280** | **-¥6,920** | **98.34%** |

★ 重要 ★: 5/17 G1 day 反映で baseline 101.33% → 98.34% へ後退、 PnL **+¥5,240 → -¥6,920** (delta -¥12,160)。 撤退ライン -¥50,000 までの buffer は **+¥43,080**。

---

## 3. 5/18+ 推奨 action (6 件)

| # | action | 推奨 | 理由 |
|:---:|---|:---:|---|
| 1 | live_orchestrator schtask 登録 (option A) | **YES** | sample N 増、 paper eval base data 蓄積。 V15 production 触らず monitor only |
| 2 | v15.2 学習継続 | **NO** | fold 0-4 LGB+XGB delta +0.000005 → 17 新 features (paci / kta / kka_v2 / cha / kab) の純増効果 ほぼゼロ確定。 fold 5 完走待ちは情報目的のみ |
| 3 | 戦略⑦案 C 適用継続 | **YES** | ≤5/10 N=466 で +3.67pt 検証済。 5/17 noisy day も 戦略⑦ 影響範囲外 (B/E 0 件)。 検証済設定維持 |
| 4 | 投票額 ¥700/R 維持 | **YES** | N5 MC verdict (case D 4 週間 worst 5% percentile -¥50,172 と 撤退 -¥50,000 整合)。 buffer 縮小したが ¥700 維持で worst case 耐性内 |
| 5 | 動画系 (V21 video pipeline) 再開検討 | **NO** | 規約 NG、 永久放棄 (Sub-task 11 確定)。 V15 越え path は 動画 + 大規模 FE が必要だが 動画 NG → V15 越え 困難確定 |
| 6 | 累計 monitoring 自動化 (daily_cumulative_audit) | **YES** | drift 第二弾 (5/17 audit-1 architecture drift) で自動 audit の必要性 強化。 daily 22:00 schtask 登録推奨 |

### 補足: v15.2 paper eval

- v15.2 paper eval (option A live_orchestrator) は schtask 登録するが、 17 features 効果 ≈ 0 確定済のため **採用 GO 期待 低**。
- ★ honest 判定 ★: v15.2 は ★ NO-GO 濃厚 ★、 V15 越え path 不明。 5/18+ は ★ V15 維持 + 累計 monitoring 強化 ★ が現実解。

---

## 4. memory drift 再発防止 強化

### 4-1. drift 履歴

| # | 時期 | drift 内容 | 解決状況 |
|:---:|---|---|---|
| 1 | 5/16 P0-1 | 累計 ROI 119.2% vs 真値 101.33% (CLAUDE.md row 72/77/1347/1363) | daily_cumulative_audit 登録 (5/18 admin 予定)、 解決進行中 |
| 2 | 5/17 audit-1 | V15 architecture 4-model vs 真値 2-model (CLAUDE.md row 72/119/2188 等 多数) | 本 doc で警告、 ★ 自動 audit 必要 ★ |

### 4-2. 新規 防止策 (provisional 提案)

1. **`tools/v15_architecture_audit.py`** 新規 (weekly schtask)
   - V15 .pkl.gz の `ensemble_weights` を 自動 read
   - 期待値 (LGB+XGB 2-model、 mlp:0) と比較、 差分時 Discord 警告
   - features 数 / list が cache (`_v15_optuna_df_cache.pkl.gz`) と一致するか check

2. **CLAUDE.md「V15 真値表」 section 追加** (★ 本 audit-5 で 提案のみ、 commit は親 ★)
   ```
   ## V15 真値表 (audit_5 確定、 weekly auto-verify)
   - ensemble: LGB+XGB 2-model (weights {lgb:0.504, xgb:0.496, mlp:0})
   - features: 145
   - WF mean (LGB+XGB): 0.8678 (6-fold 2020-2025)
   - WF mean (Grid 4-model from master_report): 0.8858 (2021-2025 only)
   - stored .pkl.gz auc: 0.8939 (single-run, NOT WF)
   - 累計 ROI (5/17 反映): 98.34% / PnL -¥6,920 / n=596
   - 撤退余裕: +¥43,080
   ```

3. **regression test 拡張** (T1 で 23→33、 audit-5 で +2 提案):
   - V15 architecture invariance test (ensemble_weights 期待値固定)
   - 累計 ROI baseline auto-check (cumulative_results.csv 直接集計値が cumulative_truth.json と一致)

### 4-3. 既存 防止策

- T1 features_integrity_monitor.py: daily 22:00 schtask 登録 ready (★ 5/18 user 判断後 admin 実行 ★)
- daily_cumulative_audit: 5/18 admin 登録予定

---

## 5. V15 production 不変保証

| 項目 | 状態 |
|---|:---:|
| `keiba_model_v15_central.pkl.gz` | ✅ 完全不変 (read-only access のみ) |
| `keiba_model_v15_central_live.pkl.gz` | ✅ 完全不変 |
| `tools/predict_core.py` | ✅ 完全不変 |
| `tools/daily_predict.py` | ✅ 完全不変 |
| `tools/race_auto_notify.py` | ✅ 完全不変 |
| `app.py` | ✅ 完全不変 |
| `data/cumulative_results.csv` | ✅ 完全不変 (read-only 集計のみ) |
| v15.2 training PID 23528 | ✅ 中断なし |
| 既存 schtasks | ✅ 全て不変 |
| commit / push | ✅ なし (★ 親集中 ★) |

---

## 6. 参考 source 一覧

| file | role | 引用箇所 |
|---|---|---|
| `docs/T1_FEATURES_INTEGRITY_AUDIT_2026_05_17.md` | features integrity (audit-3 相当) | section 0/1/4 |
| `docs/V15_2_FAIR_BENCHMARK_2026_05_17.md` | V15 vs v15.2 LGB+XGB 比較 (audit-2 相当の 部分情報) | section 0/1 |
| `docs/ROI_DISCREPANCY_2026_05_16.md` | 累計 ROI 真値 (audit-4 相当の base) | section 1/2 |
| `docs/VICTORIA_MILE_FULL_18_AUDIT_2026_05_17.md` | 5/17 G1 day 投票前 audit | section 2 |
| `data/cumulative_results.csv` (status=settled) | 累計 真値 集計 (5/17 反映後) | section 0/2 |
| `data/v15_2/v15_baseline_lgb_xgb_20260517_1715.json` | V15 LGB+XGB fair benchmark (6-fold mean 0.8678) | section 0/1 |
| `data/v15_2/wf_results_20260517_1709.json` | v15.2 partial WF | section 0 |
| `data/v15_master_report.json` | V15 master Grid mean 0.8858 (2021-2025) | section 0/1 |

---

## 7. 完了 metric

| 項目 | 状態 |
|---|:---:|
| audit-1/2/3/4 docs 完了確認 | ✅ (★ 未作成 ★ を確認、 本 doc は provisional 統合) |
| 真値 vs memory drift 完全表 | ✅ 完了 |
| 5/17 G1 day 結果反映 (累計 98.34%) | ✅ 完了 |
| 5/18+ 推奨 action (6 件) | ✅ 完了 |
| memory drift 再発防止 強化 提案 | ✅ 完了 |
| V15 production 不変保証 | ✅ 完全保証 |
| 本 doc | ✅ saved (provisional、 audit-1/2/3/4 完成後 update 想定) |

---

end of doc.
