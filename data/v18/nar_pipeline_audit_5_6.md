# NAR pipeline 動作確認 (Session #31 C)

**作成**: 2026-05-06 PM
**目的**: 5/12 NAR paper 開始の前提確認

---

## 1. NAR 5 task status (5/6 PM)

| Task | LastRun | LastResult | 判定 |
|------|---------|-----------|------|
| Keiba-NarMidDayCalendar | 2026/05/06 13:00:01 | 0 | ✅ OK (60 races scrape) |
| Keiba-NarDailyScrape | 2026/05/06 16:30:01 | 0 | ✅ OK |
| **Keiba-NarDailyPredict** | 2026/05/06 17:00:01 | **1** | ❌ **FAIL** |
| Keiba-NarLiveOddsRefresh | 2026/05/05 19:00:00 | 0 | (未発火、19:00 まで待ち) |
| Keiba-NarDailyResults | 2026/05/05 21:30:01 | 0 | (未発火、21:30 まで待ち) |

---

## 2. NarDailyPredict 17:00 失敗 真因

`logs/nar_daily_predict_20260506.log` より:

```
[NAR v4] AUC=0.8145 features=22
Traceback (most recent call last):
  File "C:\Users\takum\keiba-ai\tools\predict_nar.py", line 336, in <module>
    main()
  File "C:\Users\takum\keiba-ai\tools\predict_nar.py", line 314, in main
    ranked = predict_one_race(sub, meta, model, config)
  File "C:\Users\takum\keiba-ai\tools\predict_nar.py", line 177, in predict_one_race
    feat = encode_one(r, **race_meta, jockey_stats=jockey_stats, default_horse_weight=default_hw)
  File "C:\Users\takum\keiba-ai\tools\predict_nar.py", line 104, in encode_one
    pop_rank = int(row.get('pop_rank', 99) or 99)
ValueError: invalid literal for int() with base 10: '--'
```

→ shutuba CSV の `pop_rank` 列に `'--'` (発走前で未確定の馬) が含まれており int() で失敗。

---

## 3. fix (本セッション)

`tools/predict_nar.py` L104:
```python
# Before:
pop_rank = int(row.get('pop_rank', 99) or 99)

# After (Session #31 C fix):
try:
    pop_rank = int(row.get('pop_rank', 99) or 99)
except (ValueError, TypeError):
    pop_rank = 99
```

→ `'--'` 等の不正値で fallback、try/except でも safe。

### 動作確認

```bash
$ python tools/predict_nar.py --shutuba-csv data/nar_today_shutuba_20260506.csv
[NAR v4] AUC=0.8145 features=22
... (657 行予測完了)
=== trio 3 点 ===
  1. 1-2-8
  2. 1-2-10
  3. 1-8-10
```

→ **fix 完了**、5/6 60 races の予測成功。

---

## 4. 5/12 paper 開始 前提条件 final check

| 条件 | status |
|------|--------|
| scrape_nar_today.py | ✅ (commit eeb48e45) |
| scrape_nar_results.py | ✅ (commit eeb48e45) |
| predict_nar.py | ✅ (本セッション fix で 100%) |
| nar_daily_pipeline.bat | ✅ Session #28 で実装 |
| schtasks 5 件 | ✅ admin 登録済 |
| nar_all_races.csv | 5/2 取得済 (1 年 stale だが学習済) |
| keiba_model_nar_v4.pkl | ✅ (167 KB) |

→ **5/12 paper 開始 GO** (本日の fix で blocker 解消)。

---

## 5. 残課題 (5/12 までに任意で改善)

| 項目 | 詳細 | 優先度 |
|------|------|--------|
| nar_all_races.csv 2025-06〜backfill | データ 1 年 stale | 🟢低 (paper 開始には不要) |
| jockey_stats CSV stale | 5/5 柏記念で確認、JOCKEY_OVERRIDE_JRA で補完 | 🟢低 |
| LiveOddsRefresh 19:00 動作確認 | 5/6 19:00 後の log で audit | 🟡中 |
| Results 21:30 動作確認 | 5/6 21:30 後の log で audit | 🟡中 |

---

## 6. 5/12 paper 開始フロー

```
5/12 (火) 13:00 NarMidDayCalendar → 60 races scrape
5/12 (火) 16:30 NarDailyScrape    → 出馬表 + 前夜オッズ
5/12 (火) 17:00 NarDailyPredict   → V4 model で予測
5/12 (火) 19:00 NarLiveOddsRefresh → live odds 更新
5/12 (火) 21:30 NarDailyResults   → 結果照合
```

paper 期間 5/12-5/15 で 4 日分 蓄積、5/16 (土) で V18/V19 と並んで GO/no-go 判定。

---

## 7. 結論

NarDailyPredict 17:00 失敗の真因 (`pop_rank='--'` ValueError) を **本日 fix 完了**。
5/12 paper 開始の blocker 解消、自動運用準備完了。
