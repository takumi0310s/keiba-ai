# 5/10 (日) 朝起床時 確認 prompt

**作成**: Session #87 (2026-05-09 23:30)
**用途**: 5/10 (日) 朝 起床時に Claude Code に貼る確認 prompt
**所要時間**: 30-45 分

---

## prompt 全文 (起床時 そのまま貼る)

```
おはよう。 5/10 (日) 朝。 GW Phase 2.5+ マラソン 148h+ 完了後の 初の
本番運用日。 V15 案B改 strict 投票 + Session #71/#72 初稼働日。
全機能の動作状態を確認、 不具合あれば即報告、 V15 投資保護 完全。

【背景】
ユーザー (れんはす) 5/10 朝 起床。
昨日 5/9 23:00 まで 148h+ マラソン完了。
完了 Sessions: #1-87 (88 sessions + AUDIT-1)
main HEAD: [前日最新]
累計: +¥14,140 (撤退余裕 +¥64,140)

【今日の予定 fire】
06:30 morning_checklist (既存)
08:00 DailyPredict_0800 (既存、 V15 production) ★最重要★
08:45 RaceAutoNotify_Sun_0845 (既存、 案B改候補通知)
09:30 ★SaveAllHorseScores_0930 (Session #71、 全馬 score 初稼働)★
14:00 投票候補確定通知
14:00-15:30 PAT 投票 (案B改 strict、 該当 R で ¥700)
17:00 cumulative monitor
18:00 DailyResults
20:30 1 day summary

【disable 中】
PreRacePredict_Watchdog_5_9 (Session #78 で DISABLE)
→ 全馬通知 (補助) は 5/10 動作なし
→ 5/15 V18 trial 直前 re-enable 予定

【作業】 5 領域、 30-45 分

A. schtasks 動作確認 (10分)
A1. 06:30 morning_checklist 実行ログ確認
   - logs/morning_checklist_20260510.log 存在
   - exit code 0
A2. 08:00 DailyPredict 実行ログ確認 ★最重要★
   - logs/daily_predict_20260510.log 存在
   - data/daily_predictions/20260510.csv 生成
   - V15 model ロード成功 ("v15 Pattern B" log)
A3. 08:45 RaceAutoNotify 実行確認
   - 案B改 strict 投票候補 Discord 到達
A4. 09:30 SaveAllHorseScores 初稼働確認 ★Session #71 初日★
   - data/v18/all_horse_scores_20260510.csv 生成
   - 全頭分の row 数 (該当 R 全頭)

B. csv 確認 (Session #71 初稼働) (5分)
B1. data/v18/all_horse_scores_20260510.csv
   - row 数 ≥ 該当 R の合計頭数
   - score column NaN 率 < 5%
   - top1 score 合理範囲 (0.5-0.95)

C. Discord 通知 audit (10分)
C1. dedup 動作確認
   - data/discord_dedup_state.json 確認
   - 重複通知 0 件
C2. 各時刻の通知到達確認
   - 06:30 / 08:00 / 08:45 / 09:30 全到達
C3. 異常通知の有無
   - error / FAIL keyword 検出 0 件

D. V15 投資保護 確認 (5分)
D1. predict_core.py 変更なし
   - git diff HEAD~10 -- tools/predict_core.py = empty
D2. daily_predict.py 変更なし
D3. app.py 変更なし
D4. schtasks 既存 50 件 変更なし
   - schtasks /query で count 確認

E. 集約 + Discord 1 通 (5分)
E1. 5/10 朝確認 結果 1 通 集約
E2. Discord 通知 (5/10 朝確認 完了 / 全機能 OK or NG list)

【完了基準】
✅ A: schtasks 動作確認
✅ B: csv 確認 (Session #71 初稼働)
✅ C: Discord 通知 audit
✅ D: V15 投資保護 確認
✅ E: 集約 + Discord 1 通

【絶対遵守】
🔴 NEVER:
- predict_core / daily_predict / app.py 変更
- V15 model 変更
- schtasks 既存 50 件 変更
- destructive git op (reset --hard / push --force)

🟢 OK:
- 確認 commands (read-only)
- Discord 1 通

優秀に頼む。
完了したら 「5/10 朝確認 完了、 [全機能 OK / 不具合 N 件]」 と明確に。
```

---

## 注意事項

- **本番運用日**: V15 案B改 strict 7 点 ¥700/R を 該当 R で 実投票
- **撤退ライン**: 累計 -¥50,000 (現状 +¥14,140、 余裕 +¥64,140)
- **取り返し禁止** (損切り後 翌日へ持ち越さない)
- **PreRacePredict 全馬通知 は disable 中** (Session #78)、 補助通知なし

---

## 関連

- [PRE_RACE_PREDICT_HARDCODE_REMOVAL.md] (Session #78)
- [V20_BUILD_DETAILED_PLAN.md] (Sprint 6 5/22 開始)
- [AUDIT_FULL_PROMPT.md] (全機能 audit、 1.5-2h バージョン)
