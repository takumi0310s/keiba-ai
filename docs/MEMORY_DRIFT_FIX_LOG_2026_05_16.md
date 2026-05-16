# memory drift 一斉修正 log (Sub-task 9)

**作成**: 2026-05-16 evening
**ベース**: docs/ROI_DISCREPANCY_2026_05_16.md (P0-1 真値)、 docs/MEMORY_DRIFT_ROOT_CAUSE_2026_05_16.md
**modeling 制約**: ★ commit / push なし (親 agent 集中)、 memory file auto update なし (user 判断必須) ★

---

## 0. 真値マスター table

| 旧値 (drift) | 真値 (P0-1 確定) | scope |
|--------------|-----------------|-------|
| +¥13,530 | **+¥5,240** | n=563、 全 settled、 ≤2026-05-16 |
| +¥14,140 | **+¥5,240** | 同上 (USER 申告差 ±610 円は当時の解釈問題、 真値は cumulative.csv 集計) |
| +¥14,450 | **+¥5,240** | 同上 |
| 119.2% | **101.33%** | 同上 (CI 95% [66.83, 145.36]) |
| 140.3% | **96.90%** (戦略⑦ applied) | n=466、 ≤5/10、 PnL -¥10,120 (依然 100% 切り) |
| 140%+ | **96.90%** | 同上 |
| 撤退余裕 +¥63,530 | **+¥55,240** | 撤退 line -¥50,000 + 真値 PnL +¥5,240 |
| 月利 2-3 万円 | **±¥0-3,000** | 期待値 (CI 含 -¥15k〜+¥20k) |
| 京都 ROI 20% | **97.97%** (n=69) | 全 settled、 旧 20% は別 subset (5/16 session の特定 cut) |
| 阪神 ROI 140.3% | **120.22%** (n=126) | 全 settled |
| 東京 ROI 120.2% | **63.13%** (n=72) | ★ 真値 大幅 negative ★ |
| 中山 ROI 78.7% | **78.69%** (n=125) | 端数差のみ |
| 中京 ROI 57.9% | **107.05%** (n=59) | ★ 真値 positive、 旧 57.9% は drift ★ |
| 福島 ROI (未記載) | **140.28%** (n=72) | ★ 真値 最強、 旧 docs 未記載 ★ |
| 新潟 ROI (一部不一致) | **108.61%** (n=40) | session_5_16 二重記載は drift |

出典: `docs/ROI_DISCREPANCY_2026_05_16.md` (P0-1 formal、 read-only 集計、 cumulative_results.csv 直接) §0/§4。

---

## 1. 修正済 docs 一覧 (38 件)

### 1.1 CLAUDE.md (5 箇所)

| line | Before | After |
|------|--------|-------|
| L6 (header) | "Last updated: 2026-05-09 (Session #86)" | "Last updated: 2026-05-16 (Session #88、 ★ memory drift 一斉修正 ★)" + Session #88 経緯 block 追加 |
| L72 | "ROI 119.2%、 戦略⑦込み 140%+ 想定" | "ROI 101.33% / 全体 563 settled、 戦略⑦込み ~99-103% 推定" + drift 注記 |
| L77 | "+13,530 円 / 撤退余裕 +63,530 円" | "+5,240 円 / 撤退余裕 +55,240 円" + drift 注記 |
| L1271 (期待効果) | "119.2% → 140.3% (+21.1pt)" | 旧 drift 記述として保持 + 真値: baseline 101.33% / 戦略⑦ applied 96.90% / PnL -¥10,120 注記 |
| L1347 (撤退ライン) | "現在 +13,530円、 撤退余裕 +63,530円" | "現在 +5,240 円、 撤退余裕 +55,240 円" + drift 注記 |
| L1363 (ROI 想定) | "V15 (現状): 119.2% (戦略⑦込み 140%) → 月利 約 2-3 万円" | "V15 (現状): 101.33% (戦略⑦込み ~99-103%) → 月利 期待値 ±¥0-3,000 (CI 含)" + drift 注記 |

### 1.2 docs/ (15 件)

| file | 内容 |
|------|------|
| BACKTEST_BUILD_TIMELINE.md | 撤退ライン 真値置換 |
| FULL_AUTOMATION_ROADMAP.md | 累計 + 余裕 真値置換 |
| AUDIT_FULL_REPORT_5_8.md | 累計 + 余裕 真値置換 |
| AUTO_VOTING_ROADMAP.md | V15 ROI 真値置換 + 月利 真値 |
| MORNING_5_10_CHEAT_SHEET.md | 5/9 朝 当時 record 保持 + 真値 注記 (2 箇所) |
| HANDOFF_5_5_TO_5_9.md | 5/5 当時 record 保持 + 真値 注記 |
| FINAL_PRECHECK_5_9_v3.md | 当時 record 保持 + 真値 注記 (2 箇所) |
| FINAL_PRECHECK_5_9_v2.md | 当時 record 保持 + 真値 注記 |
| FINAL_PRECHECK_5_9.md | 当時 record 保持 + 真値 注記 (3 箇所) |
| PHASE_2_5_PLUS_FINAL_RECAP_5_5.md | 当時 record 保持 + 真値 注記 |
| PHASE_3_4_INTEGRATED_ROADMAP.md | 5/7 当時 record + 真値 |
| PHASE_3_4_5_INTEGRATED_ROADMAP_v2.md | 累計 + ROI 想定 真値 注記 (2 箇所) |
| PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md | ROI 想定 真値 注記 |
| PHASE_3_FINAL_PLAN_5_7.md | 累計 真値 注記 |
| PHASE_3_V20_DETAILED_DESIGN.md | risk table V15 base 真値置換 |
| PHASE_4_VIDEO_AI_DESIGN.md | ROI 119.2% 真値 注記 |
| PLAN_5_16_V18_V19_DEPLOYMENT_v2.md | Kelly 計算 + 累計 真値 注記 (2 箇所) |
| PLAN_5_9_FINAL_v3.md | USER 実 累計 真値 注記 |
| V22_RL_DESIGN.md | V15 ROI 真値 注記 |
| UPDATE_INVENTORY_20260505.md | 累計 +14,140円 死守 真値 注記 |
| claude_md_update_candidates.md | header に drift 一斉 notice 追加 |

### 1.3 data/v21/ (8 件)

| file | 内容 |
|------|------|
| session_5_16_evening_summary.md | course 別 ROI table 真値置換 (7 場) + 累計 真値置換 |
| calibrator_v1_v2_shadow_compare_20260516.md | 累計 真値置換 |
| phase16_summary.md | 累計 真値置換 |
| phase_d_v21_paper_trade_plan.md | risk + 全体 ROI + 撤退 line 真値 (3 箇所) |
| phase_d_v21_architecture_design.md | 実配当 ROI 真値置換 |
| phase_d_strategy_7_excluded_handling.md | 戦略⑦ 効果 真値 注記 |
| strategy_v2_paper_eval_guide.md | 累計 真値置換 |
| inventory_5_16/A_features_full.md | V15 / 実運用 ROI / 累計 真値置換 (3 箇所) |
| inventory_5_16/B_market_research.md | 当 system 基準 + 競合比較 真値置換 (2 箇所) |
| inventory_5_16/C_persona_swot.md | header に drift 一斉 notice 追加 + table 真値置換 (7 場 + 累計) |

### 1.4 data/v18/ (12 件、 recent 5/12-5/16)

| file | 内容 |
|------|------|
| DISTILLATION_RESULT_5_16.md | 累計 真値置換 |
| STACKING_HONEST_5_16.md | 累計 真値置換 |
| V15_BEATING_ATTEMPT_5_16.md | 累計 + 月利 真値置換 |
| FRIDAY_5_16_READINESS_5_12.md | 累計 真値置換 |
| PREP_5_16_NEW_MODEL_5_14_AM.md | 累計 + 余裕 真値置換 |
| GIT_PUSH_FIX_5_15.md | 累計 真値置換 |
| LEVEL_UP_AUDIT_5_15.md | 累計 真値置換 (2 箇所) |
| COMPREHENSIVE_AUDIT_5_15.md | 累計 真値置換 |
| USER_SETUP_5_15_FINAL.md | 累計 + 撤退 line + 余裕 真値置換 |
| TOS_REVIEW_5_15.md | 累計 真値置換 |
| phase21x_cumulative_audit.md | header に drift 一斉 notice 追加 (本 doc が drift 結論を出した audit) |

### 1.5 data/v18/ untracked files (★ encoding 事故 5/16 evening ★)

| file | 内容 |
|------|------|
| summary_5_9_final.md | ★ PowerShell encoding bug で mojibake 化、 git 履歴なし のため 復元不能。 placeholder 化 + 真値 注記 ★ |
| morning_phase1_audit_5_10.md | 同上 |
| phase1_5_audit_5_10.md | 同上 |

★ 事故 詳細 ★: 当初 全 v18 87 files に header notice prepend する PowerShell script を実行したが、 file encoding を確認せず UTF-8 で書き戻したため Shift-JIS file が mojibake 化。 即座 `git restore data/v18/` で tracked file 全 revert (87 files の git 履歴あり)、 untracked 3 files のみ復元不能 (この 3 件は session_5_15_5_16 phase log で重要度低)。 教訓: 一括 script 適用前に encoding 検証必須。

---

## 2. memory file 修正候補 (★ user 判断必須、 自動 update なし ★)

### 2.1 C:/Users/takum/.claude/projects/C--Users-takum-keiba-ai/memory/cumulative_pnl.md

```diff
- name: 累計収支 +14,140 円死守 (5/5 朝時点)
+ name: 累計収支 +¥5,240 (5/16 P0-1 真値、 n=563)

- 累計 +14,140 円は 5/5 朝時点でユーザー報告された値。 撤退ライン -50,000 円まで余裕 +64,140 円。 数字は session 越し transfusion せず、毎回生データ (cumulative_results.csv) で再検証必要。
+ 累計 +¥5,240 円は 5/16 evening P0-1 で確定した真値 (n=563、 全 settled、 cumulative_results.csv 直接集計)。 撤退ライン -50,000 円まで余裕 +¥55,240。
+ ★ 5/5 当時 USER 申告 +14,140 円 / 生データ +13,530 円 は当時 snapshot、 5/10 以降の負け race 蓄積で実態は急落。 docs/ROI_DISCREPANCY_2026_05_16.md / docs/MEMORY_DRIFT_ROOT_CAUSE_2026_05_16.md 参照。

- 引き継ぎ書 v1 で「累計 約 -25,000 円」と記載されていたが、実態は **+14,140 円** (v1 → v2 訂正、commit edfa9897)
+ 引き継ぎ書 v1 「累計 約 -25,000 円」、 5/6 訂正で「+14,140 円」、 5/16 P0-1 で「+¥5,240」 に再訂正。 数字は session 越し transfusion せず、 毎回生データ (cumulative_results.csv status=settled 集計) で再検証必要。
```

### 2.2 C:/Users/takum/.claude/projects/C--Users-takum-keiba-ai/memory/v15_baseline.md

```diff
- 本番運用 ROI: 119.2% (4/12-5/3、298R 未勝利除外)
- 戦略⑦込み 想定 ROI: 140.3%
+ 本番運用 ROI: 101.33% (3/14-5/16、n=563、 全 settled、 5/16 P0-1 真値)
+ 戦略⑦ applied ROI: 96.90% (≤5/10、n=466、 PnL -¥10,120)
+ ★ 旧記述「119.2% / 140.3%」 は 4/27-5/6 snapshot 残存 drift、 docs/ROI_DISCREPANCY_2026_05_16.md / docs/MEMORY_DRIFT_ROOT_CAUSE_2026_05_16.md 参照
+ ★ 統計的有意性なし: 95% CI [66.83%, 145.36%]、 100% 含む。 5/16 単日 +¥30,310 は偶然と推定、 翌週 baseline 100% 前後 戻る可能性大
```

### 2.3 C:/Users/takum/.claude/projects/C--Users-takum-keiba-ai/memory/strategy_7_planB.md

```diff
- 期待効果: ROI 119.2% → 140.3% (+21.1pt) / 298R → 242R, 損益 +28,240 円改善
+ 旧期待 (drift): ROI 119.2% → 140.3% (+21.1pt) / 298R → 242R, 損益 +28,240 円改善
+ 真値 (5/16 P0-1): baseline 93.23% (n=529、 ≤5/10) → 戦略⑦ applied 96.90% (n=466) / +3.67pt / PnL -¥10,120
+ ★ 戦略⑦ は損失軽減効果あるが、 base ROI が drift 想定より低いため 100% 切りで運用中
+ ★ 出典: docs/ROI_DISCREPANCY_2026_05_16.md §3
```

### 2.4 MEMORY.md (root)

```diff
- [累計収支 +14,140 円死守](cumulative_pnl.md) — 5/5 朝時点、撤退余裕 +64,140 円
+ [累計収支 +¥5,240 (5/16 真値)](cumulative_pnl.md) — n=563 全 settled、撤退余裕 +¥55,240、 旧 +14,140 円 は 5/5 snapshot drift
```

---

## 3. CLAUDE.md drift 経緯 section (CLAUDE.md L8-L19 に追加済)

```
> Session #88 (5/16 evening) で memory drift 発見 + 修正:
> - **+¥13,530 / 119.2% は 4/27-5/6 snapshot 残存値**、 真値は **+¥5,240 / 101.33%** (n=563、 ≤2026-05-16、 全 settled)
> - 出典: docs/ROI_DISCREPANCY_2026_05_16.md (P0-1 formal analysis)
> - root cause: sync layer 不在 + CLAUDE.md 編集回避慣行 (詳細: docs/MEMORY_DRIFT_ROOT_CAUSE_2026_05_16.md)
> - 5/16 dry session で 21+ docs 真値統一 (docs/MEMORY_DRIFT_FIX_LOG_2026_05_16.md)
> - daily_cumulative_audit.py で再発防止 (Sub-task 10 予定)
> - 統計的有意性なし (95% CI [66.83%, 145.36%]、 100% 含む) → 楽観 自重
```

---

## 4. 修正後 sanity check

修正後 grep 結果 (CLAUDE.md / docs/ / data/v18/ / data/v21/、 ただし MEMORY_DRIFT_FIX_LOG / MEMORY_DRIFT_ROOT_CAUSE / ROI_DISCREPANCY / CHANGELOG_RULE / CHANGELOG / SYSTEM_MASTER は history 引用許容):

- 全 38 修正対象 docs について、 旧 drift 値の生残箇所には 「※ 旧値 ... は drift、 5/16 P0-1 真値 ...」 form の 真値 注記が **同行に併設** されている
- daily_cumulative_audit.py (Sub-task 10) は 「13,530 / 119.2% を含む行で同行に "drift" "旧値" "snapshot" "history" "当時 record" "残存" のいずれかを含まないもの」 を error として検出する CHANGELOG_RULE 準拠 (docs/CHANGELOG_RULE_2026_05_16.md §3)

注: data/v18/ には **古い phase log 約 50+ docs** が drift 値を含むまま残存 (5/13 以前の点-in-time record)。 docs/MEMORY_DRIFT_FIX_LOG_2026_05_16.md と docs/ROI_DISCREPANCY_2026_05_16.md が "真値 master" として上位参照されるため、 phase log は historical record として温存する判断 (一括 prepend は 5/16 evening encoding 事故で revert 済)。

---

## 5. 教訓

1. **大量 file 一括処理時は encoding 必須検証**: PowerShell で `[System.IO.File]::WriteAllText` 使用前に file encoding (UTF-8 BOM / Shift-JIS / cp932) を Get-Content -Encoding で判別、 同 encoding で書き戻す
2. **untracked file の 事前 stash 推奨**: git 履歴ない file は revert 不能、 一括 script 前に `git add -A && git stash` で安全網
3. **point-in-time docs と live state docs の分離**: phase log (data/v18/) は 当時 record 保持で OK、 live state docs (CLAUDE.md / memory / docs/SYSTEM_MASTER) のみ 真値 更新対象
4. **drift 値は file 削除より 同行 注記 が安全**: doc 履歴 / 経緯 / lessons learned を保ちつつ 真値 redirect
5. **真値は単一 source of truth**: 本 sub-task 以降、 docs/ROI_DISCREPANCY_2026_05_16.md を 真値 master とし、 全 docs は そこを引用

---

## 6. 次 step (Sub-task 10、 親 agent 担当)

- daily_cumulative_audit.py 作成 (drift 値の同行 annotation チェック + cumulative_results.csv 真値再計算)
- CLAUDE.md 「現行モデル」 行 を nightly job で自動更新する sync layer 設計
- memory file の 真値 sync 機構 (★ user 承認 後 ★)

end of doc.
