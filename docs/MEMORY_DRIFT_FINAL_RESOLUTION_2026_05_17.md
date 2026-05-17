# memory drift 5 件 全訂正 final resolution (2026-05-17)

date: 2026-05-17
type: documentation correction (read-only audit + docs 訂正のみ)
status: ★ Sub-task D 完了 ★

## 0. 概要

5/16-5/17 audit で 5 件の memory drift 確定 + 全訂正。 V15 production は完全不変、 v15_full / v15.2 training も中断していない。 destructive op / git commit / push は一切実施せず、 docs 訂正のみで真値統一。

## 1. drift 一覧

| # | item | 旧値 (drift) | 真値 (5/17 後) | 確定 audit | resolved |
|---|------|--------------|---------------|-----------|----------|
| 1 | 累計 ROI | 119.2% | **98.34%** (n=596、 CI [66.33%, 138.05%] 100% 含む = 統計的有意 勝ち なし) | V15-audit-4 | ✅ |
| 2 | architecture | 4-model (LGB+XGB+FT+IR) | **LGB+XGB 2-model production** (mlp=None、 FT/IR は .pkl 未保存、 WF 評価専用) | V15-audit-1 | ✅ |
| 3 | WF AUC | 0.8939 | **0.8678** (LGB+XGB genuine WF 6-fold) / 0.8858 (Grid 4-model 5-fold) ※ 0.8939 は LGB train-set self-eval (in-sample LEAKY) | V15-audit-2 | ✅ |
| 4 | features count | 150 | **145** (booster)、 Pattern B features list 150 だが predict_core.py L2162-2163 で booster 入力後 truncate | V15-audit-1 / V15-audit-3 | ✅ |
| 5 | formation record | 記録あり (暗黙) | ★ **race-time formation 永久喪失** ★ (race_auto_notify.py 独立予測 → Discord 送信のみ、 不揮発化なし)。 trio_bets_str は AM 8:00 morning prediction のみ | data-audit-3 | ✅ |

## 2. 累計 PnL 関連 (drift 1 / 2 派生)

| 旧値 | 値 | 出典 |
|---|---|---|
| +¥13,530 | drift (旧 CLAUDE.md、 4/27-5/6 snapshot 残存と推定) | claim 不能 |
| +¥5,240 | 5/16 P0-1 真値 (n=563、 ≤5/16) | docs/ROI_DISCREPANCY_2026_05_16.md |
| **¥-6,920** | ★ **5/17 V15-audit-4 真値** (n=596、 ≤5/17) ★ | docs/V15_AUDIT_4_CUMULATIVE_ROI_5_17_2026.md |

撤退ライン -¥50,000 までの余裕: **¥43,080** (現累計 ¥-6,920 から)。

## 3. 経緯

| 時刻 | event | 影響 |
|---|---|---|
| 5/16 evening | drift 1 (ROI/PnL) 発覚 (P0-1) | CLAUDE.md / 21+ docs 真値統一 (P0-1 → 101.33% / +¥5,240) |
| 5/17 朝 | drift 1 再 verify + 戦略⑦案 C 適用 | 5/17 G1 day 投票 33 R 全件 (案 C strict 適用) |
| 5/17 夜 | drift 2-5 全 verify (V15-audit-1〜5 + data-audit-1〜4) | architecture / WF AUC / features / formation 真値確定 |
| 5/17 夜後半 | 全訂正 + 新規 doc (本 doc) | docs/data/v21 全 drift 箇所 真値統一 (Sub-task D) |

## 4. 訂正済 file 一覧 (Sub-task D 内)

| file | 訂正内容 |
|------|----------|
| `CLAUDE.md` | Session #89 block 追加、 model 概要 / model 詳細 / ベースライン / 投資保護 / 月利想定 全 drift 訂正 |
| `docs/SYSTEM_MASTER_2026_05_16.md` | Executive summary / model table / SWOT / critical 注意事項 / file 構成 / 投資保護 全 drift 訂正 |
| `data/v21/inventory_5_16/A_features_full.md` | model table / pipeline / data 表 / 強み / mini-map / checklist 全 drift 訂正 |
| `data/v21/inventory_5_16/B_market_research.md` | 当 system 基準 / 現状機能 / strength 全 drift 訂正 |
| `data/v21/inventory_5_16/C_persona_swot.md` | header 真値 update / §0 評価前提 / ペルソナ 2 / ペルソナ 4 / SWOT S1+S4 / W1 / SO4 / 改善案 全 drift 訂正 |
| `data/task_outcomes/baseline_v15.json` | 完全 rewrite (真値 5/17 反映、 history 含む、 memory_drift_resolved 6 件記載) |
| `docs/MEMORY_DRIFT_FINAL_RESOLUTION_2026_05_17.md` | ★ 本 doc (新規) ★ |

注: 過去 docs (docs/TYB_*, docs/P0_5_*, data/v21/session_5_16_*, docs/SUBTASK_*) 等は 5/16 P0-1 で既に真値 update 済 (docs/MEMORY_DRIFT_FIX_LOG_2026_05_16.md)。 本 Sub-task D では 5/17 audit で 確定した 残る 4 drift (architecture / WF AUC / features / formation) を中心に訂正。

## 5. 再発防止策

| 策 | 担当 | 状態 |
|---|---|---|
| `tools/daily_cumulative_audit.py` (commit 5972f8f0) | drift 1 (ROI/PnL) 自動検出 | 稼働中 |
| T1 `features_integrity_monitor` | drift 4 (features count) 自動検出 | 稼働中 |
| T4 `leak_audit_automation` (docs/T4_LEAK_AUDIT_AUTOMATION_2026_05_17.md) | drift 2 (architecture) leak detection | 稼働中 |
| `race_notify_log v2` (Sub-task C) | drift 5 (formation record) 解決 | 5/18+ 開始予定 |
| `v15_full` case D 学習 (Sub-task B) | drift 2 (architecture) 真値反映 + FT/IR 有効化 +0.018 AUC 検証 | 進行中 (本 task 中断なし) |

## 6. 5/18+ 認識

- 真値統一済 (Sub-task D で 6 file + 新規 doc)
- V15 真の architecture = **LGB + XGB 2-model production**
- 改善 path = v15_full で FT+IR 有効化 (+0.018 AUC、 5/24+ paper eval で検証)
- formation 真の record = race_notify_log v2 で 5/18+ 開始
- 累計 PnL は ¥-6,920 で 「赤字に転落」、 但し CI [66.33%, 138.05%] 100% 含む = 偶然範囲内
- 撤退余裕 ¥43,080 / 中間アラート (累計 -¥10,000) まで ¥3,080 余裕しかない → 5/24 単日 ROI<50% で 警報レベル

## 7. V15 production 不変保証 ✅

| 項目 | 状態 |
|------|------|
| `keiba_model_v15_central.pkl.gz` | 不変 (2026-04-08 23:32:37) |
| `keiba_model_v15_central_live.pkl.gz` | 不変 (2026-04-08 23:32:38) |
| `tools/predict_core.py` | 不変 |
| `tools/daily_predict.py` | 不変 |
| `tools/race_auto_notify.py` | 不変 |
| `data/cumulative_results.csv` | 不変 (read-only audit のみ) |
| v15.2 / v15_full training | 中断なし (独立 process) |
| schtasks 7 件 | 不変 |
| git commit / push | ★ 親集中 (本 Sub-task では実施せず) ★ |

## 8. honest 自己評価

| 項目 | 評価 |
|------|------|
| fabrication | 0 (全数値は audit-1〜4 + V15-audit-1〜5 + data-audit-1〜4 から正確に引用) |
| 「想定」 vs 「実測」 区別 | 明確 (genuine WF / stored .pkl.auc / Grid 4-model 5-fold を別 row で記載) |
| drift 訂正 出典明示 | ✅ (全 drift 箇所に audit 出典追記) |
| destructive op | 0 |
| V15 production 不変 | ✅ |
| commit/push | 親集中 (本 task では実施せず) |

---

★ honest 厳守、 V15 production 完全不変、 drift 5 件全訂正、 真値統一完了 ★
