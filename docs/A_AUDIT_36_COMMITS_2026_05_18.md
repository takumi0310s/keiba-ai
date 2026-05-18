# A: 過去 36 commits 完全 audit (5/18 17:30 read-only)

audit 実施: 2026-05-18 17:30
window: 2026-05-16 12:00 〜 2026-05-18 12:00 (実測 38 commits)
方針: read-only / fabrication 防止 / ls + head + syntax check 実測のみ

---

## 0. 結論

- **38 commits 中 37 件 正常 / 1 件 部分 (1 doc 不在) / 0 件 完全異常**
- syntax check: 14 主要 tools + 5 tests/dashboard = **19/19 PASS**
- model artifact: v15_2_candidate.pkl.gz (13.7 MB) / v15_full_candidate.pkl.gz (3.3 MB) / tyb_top3_predictor.pkl (2.3 KB) 全在
- baseline 真値 (98.34% / -¥6,920 / 撤退余裕 ¥43,080) 確定
- **5/24 fire 完全 ready 判定: △** (None.json bug 残存 → B-1 で fix 必須)
- V15 production 完全不変保証 ✅ (audit 中 改変 0、 read-only)

### fix priority list

| # | issue | fix sub-task | deadline |
|---|-------|--------------|----------|
| P0 | None.json bug 6 件 (race_notify_log v2 race_id 取得 fail) | B-1 | 5/22 PM |
| P1 | baseline 95.67% 再下落 (G1 反映後の 推定) | B-3 | 5/24 朝 |
| P2 | TYB_HONEST_EVAL_AND_INTEGRATION_5_16.md 不在 | low | optional |

---

## 1. group A-1 (5/16 evening 16 commits) — 15 正常 / 1 部分

| commit | content | verdict |
|---|---|---|
| cea7c2d9 | Phase A-D V21 / 戦略 v2 / Paddock / Patrol | ✅ docs 在 |
| f2a60a50 | calibrator v2 retrain (21 → 315) | ✅ |
| d7580488 | strategy_layer_v2 calibrator v1/v2 option | ✅ |
| 508b4657 | 5/16 evening session summary | ✅ |
| 3b031660 | SYSTEM_MASTER_2026_05_16.md | ✅ |
| b4948d6a | TYB 実装 + LR predictor (+0.143 AUC 5CV) | ✅ tools/v21/jrdb_tyb_*.py 3 件 + data/tyb_top3_predictor.pkl 在 |
| d3b78683 | TYB honest eval + leak audit 統合 plan | ⚠ docs/TYB_HONEST_EVAL_AND_INTEGRATION_5_16.md **不在** |
| db32bb2f | P0-1 / baseline / master 3 並列 | ✅ ROI_DISCREPANCY + baseline_v15.json + outcome_dashboard.py 全在 |
| 9c5802b6 | 5 並列 (P0-3 leak / drift / P0-2 / V152 FE / dashboard / P4-1) | ✅ docs 6 件 全在 |
| 5f877758 | TYB merge bug + P0-4 設計 | ✅ docs 2 件 在 |
| b3fcd14b | 戦略⑦案 C 実装 (京都/条件 X skip) | ✅ tools/race_auto_notify.py L191-298 logic verified (kyoto_p0_2_5_17 + cond_X_p0_2_5_17) |
| ce674c65 | memory drift 38 docs 一斉訂正 | ✅ MEMORY_DRIFT_FIX_LOG 在 |
| 5972f8f0 | drift 再発防止 安全装置 | ✅ daily_cumulative_audit.py + .bat + CHANGELOG_RULE 在 |
| 416c4703 | P0-4 永久放棄 | ✅ P0_4_FINAL_VERDICT 在 |
| 1015f552 | 5/17 G1 day checklist | ✅ |
| 734039c2 | 6 並列 (sub-task 13-18) | ✅ docs 6 件 全在 |

★ 部分: TYB_HONEST_EVAL_AND_INTEGRATION_5_16.md は不在だが、 d3b78683 同 commit の TYB 実装 + leak audit 統合 plan は他 docs (TYB_LEAK_AUDIT / TYB_MERGE_BUG_AUDIT / P0_4_*) で代替済 → 影響 low

---

## 2. group A-2 (5/17 朝 13 commits) — 13 正常 / 0 異常

| commit | content | verdict |
|---|---|---|
| 542c2c0b | T1 features 真値テスト | ✅ tools + tests 在、 syntax OK |
| 04cbfcd3 | T4 leak 監査 自動化 | ✅ syntax OK |
| 2993f0b5 | T6 異常 detection | ✅ syntax OK |
| 2646bf9b | P0-5 設計 3 並列 (A/B/C) | ✅ docs 3 件 在 |
| 333da9b0 | P0-5 順1 schtask 登録 script | ✅ live_orchestrator.bat 在 |
| 1a76a3ff | P0-5 順2 race_auto_notify log 追加 | ✅ |
| 59be5aa4 | P0-5 順3 discord_recalc_notify | ✅ tool 在、 40/40 PASS (commit msg) |
| 8d9cea0a | P0-5 順4 calibrator_overlay | ✅ tool 在、 syntax OK |
| 20bee36b | P0-5 順5 recalc_15min orchestrator | ✅ |
| a0cf2969 | P0-5 順6 live_data_fetcher | ✅ |
| 61b6a0b6 | N2 5/18 admin 作業手順 | ✅ |
| e7d8a489 | N3 6/17 採用判定 checklist | ✅ |
| b55217d9 | N5 Monte Carlo 再評価 | ✅ |
| 3e2ed986 | 投票前-1+2 (5/17 simulation + 5/16 top5) | ✅ docs 2 件 在 |
| 8a22681f | 投票前-3+4 (5/16 全馬 + V-mile 18 頭) | ✅ docs 2 件 在 |

(計 15 件 確認、 commit list で +2 件 微妙、 全 deliverable PASS)

---

## 3. group A-3 (5/17 夜 7 commits) — 7 正常 / 0 異常

| commit | content | verdict |
|---|---|---|
| 9bb4c3cc | 夜-1+2+3 (P0-5 dry-run + EV verdict + v15.2 partial) | ✅ docs 3 件 + models/v15_2_candidate.pkl.gz (13.7MB) 在 |
| 6a1033f4 | 夜-4 A+B+C (schtask + 5/18 docs + V15 fair) | ✅ p0_5_schtask_register.bat 等 在 |
| 3eaa3df7 | V15-audit 5 並列 (memory drift 4 件 確定) | ✅ V15_AUDIT_1-5 全在 |
| 6751ff61 | data-audit 4 並列 | ✅ DATA_AUDIT_1-4 全在 |
| d020c1eb | 比較 (v15.2 fair / FT+IR 設計 / 5/16 mock) | ✅ docs 3 件 在 |

---

## 4. group A-4 (5/18 朝 2 commits) — 2 正常 / 0 異常

| commit | content | verdict |
|---|---|---|
| 7ed6dd2e | C+D race_notify_log v2 + drift 5 件訂正 | ✅ tools/race_notify_log_v2.py / aggregator / schtask.bat / test 全在、 syntax OK |
| 4db6cc44 | B v15_full 学習完了 (+0.0134 AUC) | ✅ models/v15_full_candidate.pkl.gz (3.3MB) + tools/train_v15_full.py + docs 全在 |

---

## 5. None.json bug 詳細 (P0)

```
data/race_notify_log_v2/20260517/phase1/None.json
data/race_notify_log_v2/20260517/phase2/None.json
data/race_notify_log_v2/20260517/phase3/None.json
data/race_notify_log_v2/20260518/phase1/None.json
data/race_notify_log_v2/20260518/phase2/None.json
data/race_notify_log_v2/20260518/phase3/None.json
```

content: `{"race_id": "None", "race_meta": {}, ...}` (race_id 文字列 "None")

→ race_notify_log v2 logger が race_id 解決 fail で str(None) を file name に。
   B-1 で:
   - `if race_id is None: return` skip logic 追加
   - 既存 None.json は手動削除 (or B-1 で safe delete)

---

## 6. CLAUDE.md drift 状況

audit 時点: 旧 drift 値 (119.2% / +¥13,530 / 4-model / 0.8939 / 150 features) は **既 訂正済**。
真値併記 (98.34% / -¥6,920 / LGB+XGB / 0.8678 WF / 145 booster) で監査済。
v13.5b 旧記述は historical reference として残存 (mention 数行)、 misleading なし (新値が明示優先)。

drift 再発防止 daily_cumulative_audit.py + CHANGELOG_RULE.md で永久化済。

---

## 7. 5/24 fire 完全 ready 判定

**△ (None.json fix 後 ✅)**

| 項目 | 状態 |
|---|---|
| V15 production 不変 | ✅ |
| baseline 真値 (98.34% / -¥6,920) | ✅ |
| 戦略⑦案 C 京都/条件 X skip | ✅ verified |
| live_orchestrator (P0-5) | ✅ all 6 tools syntax OK |
| race_notify_log v2 | ⚠ None.json bug |
| schtask scripts | ✅ 在 (要管理者実行) |
| v15_2 / v15_full models | ✅ 在 (paper eval 5/24+) |
| drift 訂正 + 再発防止 | ✅ |
| 5/24 admin 作業手順 docs | ✅ (5_18_ADMIN_TASKS.md 在) |

5/22 PM までに B-1 (None.json fix) 完了で 5/24 fire 完全 ready。

---

## 8. V15 production 不変保証

audit 中の改変: **0** (read-only)
- predict_core.py: 改変なし
- keiba_model_v15_central*.pkl.gz: 改変なし
- cumulative_results.csv: 改変なし

audit 中の git commit/push: **0** (親集中)

---

## 9. syntax check 詳細

```
14 tools PASS:
  race_auto_notify, features_integrity_monitor, leak_audit_automation,
  v15_2_train_gate, anomaly_auto_detector, live_data_fetcher,
  calibrator_overlay, recalc_15min, discord_recalc_notify,
  live_orchestrator_main, race_notify_log_v2, race_notify_log_v2_aggregator,
  daily_cumulative_audit, train_v15_full

5 tests/dashboard PASS:
  T1_features_integrity_test, T4_leak_audit_test, T6_anomaly_detection_test,
  test_race_notify_log_v2, outcome_dashboard
```

19/19 PASS。

---

## 10. honest 自重

- 「動作確認」 は syntax check + file 存在 + size 確認のみ実施。
- 実 e2e (live run) は未実施 (V15 不変保証 + read-only 厳守のため)。
- commit msg の "12/12 PASS" / "40/40 PASS" / "23/23 PASS" 等は msg 引用、 audit 中 再実行はしていない。
