# 全機能 audit prompt (148h+ マラソン後)

**作成**: Session #87 (2026-05-09 23:30)
**用途**: 148h+ マラソン (Session #1-87) 完了後の 全機能 不具合 audit
**所要時間**: 1.5-2 h
**実施タイミング**: 5/10 朝確認後 (時間あれば) または 5/11 平日

---

## prompt 全文 (audit 実施時 そのまま貼る)

```
GW Phase 2.5+ マラソン 148h+ 完了後の 全機能 不具合 audit。
完了 Sessions: #1-87 (88 sessions + AUDIT-1)。
read-only 中心、 V15 production 完全不変、 不具合あれば list 化のみ。

【背景】
ユーザー (れんはす) 5/10-11。
148h+ マラソンで 多数機能を追加・修正。
production 反映前の 最終 audit。

main HEAD: [現在最新]
累計: +¥14,140 (撤退余裕 +¥64,140)

【作業】 7 領域、 1.5-2 h

A. schtasks 全 50 件 健全性確認 (15分)
A1. schtasks /query /fo csv | wc -l = 50+ 確認
A2. 各 task の next run time / last run time 確認
A3. last run result が 0x0 (success) であること
A4. disable / pause 状態の task list 化
A5. 異常な task (last run > 7 days) の list

B. predict_core.py 健全性 (15分)
B1. import 確認 (features_v15_new 等)
B2. v15 model file 存在 (keiba_model_v15_central_live.pkl.gz)
B3. v15 feature count = 150 確認
B4. dist_cat 5bins 確認
B5. version_gate future_proof 確認

C. data/ ファイル 健全性 (20分)
C1. cumulative_results.csv 行数 + 直近 30 日 NaN 率
C2. daily_predictions/ 直近 7 日分 存在
C3. daily_results/ 直近 7 日分 存在
C4. discord_dedup_state.json 健全性 (timestamp 直近)
C5. v18/ 直近成果物 list (Session #71/#72 ファイル)
C6. 不要な temp ファイル の有無

D. tools/ scripts 動作 ping (15分)
D1. 主要 script の python -m py_compile (構文チェック)
   - tools/daily_predict.py
   - tools/race_auto_notify.py
   - tools/predict_core.py
   - tools/save_all_horse_scores.py (Session #71)
D2. import エラー の有無
D3. 廃止予定 script の list (使われていない script)

E. docs/ 整合性 (15分)
E1. INDEX.md と 実 doc 数 の整合
E2. 直近 Session #80-87 の doc 全存在確認
E3. リンク切れ の有無

F. test suite 実行 (10分)
F1. python tests/regression_test.py 実行
F2. 17 tests PASS 確認
F3. FAIL あれば list 化

G. 集約 report + Discord 1 通 (15分)
G1. 全 audit 結果を 1 ファイル集約
   docs/AUDIT_FULL_REPORT_5_10.md
G2. 不具合 list (0 件 ならその旨)
G3. Discord 通知 (audit 完了 / 不具合 N 件)

【完了基準】
✅ A: schtasks 50 件 確認
✅ B: predict_core.py 健全
✅ C: data/ 健全
✅ D: tools/ ping
✅ E: docs/ 整合
✅ F: test 17/17 PASS
✅ G: report + Discord

【絶対遵守】
🔴 NEVER:
- predict_core / daily_predict / app.py 変更
- V15 model 変更
- schtasks 既存 50 件 変更
- destructive git op (reset --hard / push --force)
- 不具合の自動修正 (list 化のみ)

🟢 OK:
- read-only command (status / log / cat)
- 構文チェック
- test 実行
- audit report 作成 (docs/ に出力)
- Discord 1 通

【優先順位】
1. 重大不具合 (V15 production 影響あり) → 即時報告
2. 中重要不具合 (production 影響なし) → list 化
3. 軽微不具合 (cosmetic) → list 化のみ

優秀に頼む。
完了したら 「audit 完了、 重大 X 件 / 中 Y 件 / 軽微 Z 件」 と明確に。
```

---

## 注意事項

- **read-only 中心**: 不具合修正は **別 session** で対応
- **list 化のみ**: 重大不具合のみ即時 user 確認
- **V15 production 完全不変**: production 反映 前の最終チェック

---

## audit 後の対応

| 結果 | 対応 |
|------|------|
| 重大 0 件 / 中 0 件 / 軽微 任意 | OK、 5/16 V18 trial へ |
| 重大 1 件+ | 即時別 session で修正 |
| 中 5 件+ | 5/12-14 平日に修正 session |
| 軽微 のみ | 5/22 Sprint 6 着手前に整理 |

---

## 関連

- [MORNING_5_10_PROMPT.md] — 5/10 朝確認 (30-45 分)
- [V20_BUILD_DETAILED_PLAN.md] — 5/22 Sprint 6 開始 (audit 後)
- [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md] — 全体 timeline
