# Phase 5 緊急 5/10 通知 score 表示 修正

date: 2026-05-10 ~12:30
session: Phase 5 (caveman mode、 Opus 4.7)
goal: ユーザー画像報告 = Stage 2 通知 score 列「-」 / 順位「?」 修正

---

## root cause: case A (CSV column 名称不一致)

`tools/stage2_predict.py:load_full_predictions()` が
`data/daily_predictions_full/20260510.csv` を読込み時、
column 名前で **mismatch**:

| 用途 | 期待 (build_horse_table) | 実 CSV 列名 (Session #71 出力) |
|------|------|------|
| V15 score | `score` | **`V15_score`** |
| ranking | `horse_rank` | **`rank_in_race`** |
| 馬番 | `umaban` | **`horse_num`** |

→ `r.get("score")` は None → "-" 表示
→ `r.get("umaban")` は None → "?" 表示
→ `sort_values("horse_rank")` も "score" も列なし → sort 適用されず

馬名 / オッズ は両方 一致 (horse_name / odds) → 表示 OK だった。

---

## 修正

`tools/stage2_predict.py:load_full_predictions()` で CSV 読込み直後に
`rename_map` で 3 列を統一:

```python
rename_map = {}
if "V15_score" in sub.columns and "score" not in sub.columns:
    rename_map["V15_score"] = "score"
if "rank_in_race" in sub.columns and "horse_rank" not in sub.columns:
    rename_map["rank_in_race"] = "horse_rank"
if "horse_num" in sub.columns and "umaban" not in sub.columns:
    rename_map["horse_num"] = "umaban"
if rename_map:
    sub = sub.rename(columns=rename_map)
```

→ 既存 build_horse_table() / sort logic 不変。

---

## 動作確認 (R5 東京 202605020605)

```
| 順 | 馬番 | 馬名 | V15 score | 単勝オッズ |
| 1 | 16 | ビービーアジャイル | 0.725 | 3.3 |
| 2 | 5 | ミスターキャンベラ | 0.693 | 7.4 |
| 3 | 9 | イーグルロック | 0.642 | 6.9 |
| 4 | 10 | イルカンダ | 0.464 | 40.8 |
...
| 16 | 2 | ランブリングマン | 0.153 | 43.8 |
```

✅ 順位 1-16 表示
✅ score 0.725 → 0.153 降順 sort
✅ 馬番 / オッズ 一致
✅ V15 top1 = 16 ビービーアジャイル (CSV 通り)

---

## 投票方針 (修正なし)

既存 logic 維持 (`stage2_predict.py:437-439`):
- Stage 2 通知は **学習用、 投票推奨ではない**
- 投票は **朝予測 (Stage 1) の trio_bets に従う**
- 累計 +13,530 円 死守

→ Phase 5 は score 表示 fix のみ、 戦略変更なし。
→ ユーザー要望「直前 30 分予測で 体重最大 AUC」 は Stage 2 比較欄で確認。
→ 朝候補 8 R は 既存 race_auto_notify.py + strategy⑦ filter 経由で確定済。

---

## 次 fire timing (5/10)

PreRacePredict_Watchdog_5_9 (30 分毎):
- 12:00 fire ✅ (R5 東京 + R6 京都 / 新潟 等 既 fire、 score 表示 bug 状態)
- 12:30 fire → 修正後 code で R7 (13:00) 等 cover
- 13:00 fire → R7 / R8
- 13:30 fire → R8 (14:00 京都 / 14:15 東京) 等 ★ 14:00 投票判断 直前 ★

→ 既 fire 済 R は cache dedup で skip。 必要なら `--force --race-id` で再 fire 可。

---

## V15 投資保護

✅ V15 model 不変
✅ predict_core / daily_predict / app.py 不変
✅ tools/stage2_predict.py の 通知部分 (column rename) のみ
✅ schtask 不変
✅ 累計 +¥14,140 維持
