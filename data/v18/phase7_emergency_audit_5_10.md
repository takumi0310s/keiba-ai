# Phase 7 緊急: 5/10 残 R + 5/11+ 全予想 audit

date: 2026-05-10 14:30
session: Phase 7 (Opus 4.7、 caveman mode)
goal: Phase 6 修正の波及確認 + 5/11+ risk 払拭

---

## A. 5/10 残 R 確認 (Phase 6 修正後)

### 投票候補 (¥1,400 残)
| 候補 | 真値 (Phase 6 修正後) | Stage 2 fire 期待 |
|---|---|---|
| 京都 R11 平城京S | **15:25** | 14:30/15:00 fire window で cover ✓ |
| 京都 R12 4歳上2勝C | **16:10** | 15:00/15:30/16:00 fire window で cover ✓ |

### Stage 2 schtask fire schedule (30 min 毎)
- 14:30 fire window 14:30-15:30 → 京都R10 14:50 / 京都R11 15:25 / 新潟R11 15:15 / 東京R10 15:01
- 15:00 fire window 15:00-16:00 → 京都R11 15:25 / 京都R12 16:10 (start) / 東京R11 15:40
- 15:30 fire window 15:30-16:30 → 京都R12 16:10 / 東京R11 15:40 / 東京R12 16:30 / 新潟R12 16:01
- 16:00 fire window 16:00-17:00 → 京都R12 / 東京R12 / 新潟R12

→ **投票候補 京都R11 / R12 ともに 30+ min 前 fire 期待**、 体重統合 OK。

---

## B. 5/11+ 発走時刻 source audit

### tools/stage2_predict.py RACE_START_TIMES (元 static dict)
| date | hardcoded entries 数 |
|------|------|
| 5/9 (土) | 約 30 R (Session #65 A 当時の 11R anchor base) |
| 5/10 (日) | 35 R (Phase 6 真値 patch) |
| 5/11+ | **0 R** (未追加) |

→ **5/16/17 (来週末) で fire 不可 risk** を発見。

### tools/race_auto_notify.py
- L50: `https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}`
- 朝 8:45 通知 / 5min 前通知 で **dynamic 取得** ✓
- 任意 date OK

### tools/daily_predict.py
- L179: 同 race_list_sub URL で **dynamic 取得** ✓
- 任意 date OK

### data/daily_predictions/{date}.csv
- 発走時刻 column 不在 (race_id / course / race_num / race_name / condition / num_horses 等のみ)
- daily_predict.py が in-memory で start_time 使用、 csv には保存せず

### data/daily_predictions_full/{date}.csv (Session #71 9:30)
- 発走時刻 column 不在 (同上)

→ **stage2_predict.py だけが static、 他は dynamic** = bug の局所性 確認

---

## C. 5/9 vs 5/10 同一性 audit

### 開催曜日 schedule 違い (実証)
| 場 R | 5/9 (土) | 5/10 (日) | ズレ |
|---|---|---|---|
| 京都 R8 | 14:00 | 13:45 | -15 min |
| 京都 R11 | 15:30 | 15:25 | -5 min |
| 京都 R12 | 16:00 | 16:10 | +10 min |
| 東京 R8 | 14:15 | 13:55 | -20 min |
| 東京 R11 | 15:45 | 15:40 | -5 min |
| 新潟 R12 | 16:10 | 16:01 | -9 min |

→ JRA 開催日の発走 schedule は**日々 変動 (3-20 min)**、 同 anchor 流用 NG。
→ 5/16 / 5/17 / 5/18 も 当日 schedule 取得必要。

---

## D. dynamic 取得化 ★ 即実装 ★ (Phase 7 で適用)

### 実装 (commit 本 commit)
`tools/stage2_predict.py` に `_load_today_dynamic_times(date_str)` 追加:
- netkeiba race_list_sub から real-time fetch
- `RACE_START_TIMES` に merge (override)
- 1 process 内で 1 回のみ (DATE 単位 cache)
- 失敗時は static fallback (static dict 維持)

`races_in_next_window()` の呼び出し直前で fetch trigger:
```python
def races_in_next_window(window_min: int = 60, now: datetime | None = None) -> list[str]:
    now = now or datetime.now()
    _load_today_dynamic_times(now.strftime("%Y%m%d"))  # Phase 7
    ...
```

### 動作確認
```
$ python -c "import stage2_predict as s2; s2.races_in_next_window(60)"
[dynamic] netkeiba race_list_sub: date=20260510, races=36, updated=0
next 1h races: 10
```
→ netkeiba 36 R load 成功、 Phase 6 真値と一致 (updated=0)

### 効果
- ✅ 5/10 残 R: dynamic load で **常に真値** 使用、 hardcode のズレ risk 消滅
- ✅ 5/16 / 5/17 / 5/18 (来週末): hardcode 不要、 当日 netkeiba から自動取得
- ✅ 任意 date 対応 (race_auto_notify と同 source 統一)
- ✅ static fallback 維持 (netkeiba block 時も 5/9 hardcode で一部動作)

---

## E. 5/11-5/15 平日 (開催無し)

| date | 曜日 | 開催 |
|------|------|------|
| 5/11 (月) | 平日 | なし |
| 5/12 (火) | 平日 | なし |
| 5/13 (水) | 平日 | なし |
| 5/14 (木) | 平日 | なし |
| 5/15 (金) | 平日 | なし |

→ 中央競馬 平日無し、 stage2 fire しても netkeiba race_list 空 = 自然 skip。
→ 5/13 (火) の dynamic 化追加実装作業も本 Phase 7 で 完了済。

---

## F. 来週末 (5/16-5/18) 動作確認 plan

5/16 (土) 朝 8:00 fire 時:
1. netkeiba race_list_sub?kaisai_date=20260516 から 真値 fetch
2. `RACE_START_TIMES` に merge (5/9, 5/10 hardcode に override されない 5/16 race_ids 追加)
3. PreRacePredict_Watchdog 8:30 fire 以降、 真値 base で 30 min 前予測
4. Stage 2 通知 真値 timing で fire

→ **5/16 で動作確認**、 ズレ無く投票判断可能。

---

## V15 投資保護

✅ V15 model 不変
✅ predict_core / daily_predict / app.py 不変
✅ schtask 不変
✅ tools/stage2_predict.py に dynamic fetch 関数追加 (既存 static dict 維持、 fallback)
✅ 累計 +¥14,140 維持

---

## 結論

| 項目 | 状態 |
|------|------|
| 5/10 残 R (京都R11 / R12) | ✅ Stage 2 fire timing 整合 (Phase 6 + 7 修正) |
| 5/11-5/15 平日 | ✅ 影響なし (開催無し) |
| 5/16-5/18 来週末 | ✅ dynamic 取得化で **fire 不可 risk 払拭** |
| 全予想体制 | ✅ source 統一 (netkeiba race_list_sub) |
