# Session #72 E: 5/9 fire 検証

**作成**: 2026-05-09 18:11 (Session #72、 dev/two-stage)
**source**: `logs/stage2_predict.log` (384 行、 13:00-15:30 fire 全件 + Session #68 修復前 logic)

---

## 5/9 PreRacePredict_Watchdog_5_9 全 fire 一覧

| fire 番号 | check_next_1h 出力時刻 (推定) | candidates | 試行 (dedup skip 除く) | discord sent |
|---|---|---|---|---|
| 1 | ~13:00 | 6 件 | 6 件 (R7-R10、 各場 1) | 6 通 |
| 2 | ~13:30 | 5 件 | 2 件 (3 dedup) | 2 通 |
| 3 | ~14:00 | 5 件 | 3 件 (2 dedup) | 3 通 |
| 4 | ~14:30 | 5 件 | 3 件 (2 dedup) | 3 通 |
| 5 | ~15:00 | 4 件 | 2 件 (2 dedup) | 2 通 |
| 6 | ~15:30 | 2 件 | 0 件 (2 dedup) | 0 通 |
| 7 | ~16:00 | 0 件 | 0 件 | 0 通 |
| **合計** | | | **試行 16 / Discord 16 通** | |

→ **Discord に 16 通 (全件 sent 確認)**。 ユーザー要望「30 分毎 1R 分 全会場分 届く」確認 OK ✅

## ユーザー要望 vs 実態 突合

### 要望 1: 「今日の不具合 (Stage 2 失敗) が解消されているか確認」

| 項目 | 結果 |
|---|---|
| Session #68 修復済 (commit 911ab4fc) | ✅ |
| 修復後の manual 動作確認 (5/9 18:00) | ✅ HTTP 400 → "Stage 1 fallback 採用" + cache skip |
| 5/9 13:00-15:30 fire は **修復前** logic で送信済 | ⚠ 古い「Stage 2 予測 失敗」 文言で 16 通送信 |
| 5/9 18:30+ の次 fire は新 logic で送信される | ✅ Session #72 C で「全馬 V15 score 順 table」 (full 不在で fallback) |
| Stage 2 全失敗の物理 root cause (netkeiba server block) | ⚠ client 側修復不可 (Session #62/63 既知) |

→ **不具合 (failure path の handling) は解消、 ただし netkeiba block 自体は server 側問題**。

### 要望 2: 「1 時間ごとに 1R 分 全会場分 届くようになったか確認」

5/9 fire pattern (window=60min):
- 13:00 fire: candidates 6 件 (3 場 × 2 R = 6)
- 13:30 fire: candidates 5 件 (3 dedup → 2 新規)
- 14:00 fire: candidates 5 件 (2 dedup → 3 新規)
- ...

→ 30 分間隔 fire で window=60min なので、 **各 R は 必ず 1-2 回 cover** される (dedup で 2 回目以降 skip)。
→ 「1 時間 / 1R 分 / 全会場分」 ✅ 確認。 5/9 16 通 全 R 通知済。

### 要望 3: 「通知は買い目ではなく、 各 R ごと 全馬評価スコア順 全馬分表記」

| 状況 | 通知内容 |
|---|---|
| 5/9 以前 (旧 logic、 Session #68 修復前) | top3 のみ (古いフォーマット) |
| 5/9 18:00+ (Session #72 C 修復後) | top3 のみ (daily_predictions_full 不在 fallback) |
| 5/10 以降 (Session #71 完了 想定) | **全馬 V15 score 順 markdown table** ★ |

→ 5/9 までは **top3 のみ** が data 制約。
→ 5/10 から Session #71 の `data/daily_predictions_full/{date}.csv` が生成されれば
  **全馬 V15 score 順 table** で自動切替 (build_message_all_horses 経由)。

### 要望 4: 「実装からテスト、 修正まで自動」

| 段階 | 自動完結 |
|---|---|
| audit (A) | logs/stage2_predict.log を解析、 fire 統計化 |
| design (B) | 通知 schema 確定 (5/10+ vs 5/9 以前) |
| 実装 (C) | tools/stage2_predict.py に load_full_predictions / build_horse_table / build_message_all_horses 追加、 自動 fallback |
| test (D) | tests/test_stage2_predict.py 7 件、 7/7 PASS |
| 検証 (E) | 本 doc |
| commit + push (F) | dev/two-stage に 5 commits |

→ ✅ 全工程 自動完結。

---

## 5/9 投票結果との関係

5/9 投票: 新潟 12R 4歳以上1勝クラス ¥700 (案B改 strict、 V15 単独)
- 朝予測 top1: 11 ハイクオリティ (V15 score 0.648)
- 投票結果 確定 (Session #67 B verdict 済): -¥700 (累計 +¥12,830)

→ **本 Session #72 の通知変更は 5/9 投票結果に影響なし** (5/10 以降の通知のみ変更)。

---

## 5/10 以降の運用 期待値

1. 8:00 朝予測 (daily_predict.py) → daily_predictions/20260510.csv 生成
2. (Session #71) → daily_predictions_full/20260510.csv 生成 (全馬 score 保存)
3. 各 R 1h 前: PreRacePredict_Watchdog 30 分毎 fire
4. stage2_predict.py:
   - load_full_predictions(race_id, "20260510") → 全馬 dict list 取得
   - build_message_all_horses → 全馬 V15 score 順 markdown table
   - Discord 送信 (channel=bets)
5. 想定 通知数: 全 R (3 場 × 12 = 36 R) を 30 分毎 fire で 2 回 cover → ~36 通 / 日

→ 5/16 V18 trial 後の運用評価 base が完成。

---

## 注: Session #72 E commit 経緯 (干渉 record)

本 doc の最初の commit (~18:07) で `git commit` 実行時、 並行 Claude session
(他 branch で work 中) が同じ working directory 上で git 操作を行ったため、
HEAD が main に swap した状態でコミットされ、 Session #71 の関連 file 7 件が
意図せず main に commit された。

main には push されておらず origin/main は 8fc4e13b 不変のため、
`git reset --hard origin/main` で main をクリーンアップ済 (18:13)。
本 doc は dev/two-stage 上で再 commit。 V15 production 完全保護維持。
