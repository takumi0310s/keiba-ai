# Phase 4 緊急 5/10 当日実装 audit

date: 2026-05-10
session: Phase 4 (caveman mode、 Opus 4.7)
goal: ユーザー 4 大要望 5/10 当日 即実装

---

## 要件 1: 京都 通知 修正

### 現状 (修正前)
`tools/race_auto_notify.py:183-186`:
```python
if course_str == '京都':
    print(f"    [STRATEGY7] Skip 京都: データ蓄積待ち")
    return
```
→ 京都 R = 予測 fire 無し + 通知 無し (戦略⑦ filter で完全 skip)

### ユーザー要望
- 京都 予測 fire (= 通知含む)
- 案B改候補抽出時のみ 京都 除外 (手動)

### 修正
race_auto_notify.py:183-186 京都 filter 削除。 戦略⑦ filter は 06_特別 + 条件E + 条件B の 3 つに縮小。

→ 京都 R も 5min 前 通知 + V15 score + 買い目 Discord 送信。
→ ユーザー が 案B改 strict 投票時 京都 を手動除外。
→ 「データ蓄積待ち、 5/11 以降に再評価」 元コメント撤去 (Phase 4 で正式変更)。

### 影響
- 5/10 14:00 以降 R で 京都 通知 復活
- 戦略⑦ ROI シミュレーション値変動 (旧 +21.1pt → 京都込みで再算定必要 = 5/16 plan)

---

## 要件 2: Stage 2 (30 分前予測) 5/10 enable

### 現状 (修正前)
- `tools/stage2_predict.py`: main に 不存在 (dev/two-stage 専用)
- `tools/race_day_weight_features.py`: main に 不存在
- `Keiba-PreRacePredict_Watchdog_5_9` schtask: DISABLED
- pre_race_predict_runner.bat: stub (stage2_predict.py 不存在で no-op exit 0)

### 修正
1. `git checkout dev/two-stage -- tools/stage2_predict.py tools/race_day_weight_features.py` (Session #78 ae81ebf0 反映済 file)
2. RACE_START_TIMES に 5/10 race_ids 追加 (5/9 entries 維持 + 5/10 同 time slot 仮定):
   - 京都 5/10: 202608030601-12 (kaiji 3 day 6)
   - 東京 5/10: 202605020601-12 (kaiji 2 day 6)
   - 新潟 5/10: 202604010401-12 (kaiji 1 day 4、 R4 不在)
3. `schtasks /Change /TN Keiba-PreRacePredict_Watchdog_5_9 /Enable` → Status=Ready, Next Run=2026/05/10 12:00

### 動作確認
```
$ python tools/stage2_predict.py --check-next-1h --no-discord --date 20260510
[check_next_1h] window=60min, candidates=10
=== Stage 2 predict 202608030605 (京都 R5) ===
... (予測 phase 入り)
```

→ 5/10 12:00 schtask fire で 30 分前予測 開始期待。
→ 14:00 以降 R で 体重統合 Stage 2 通知 期待。

### Schtask trigger 詳細
- StartBoundary: 2026-05-09T13:00:00
- Repetition: PT30M (30 分毎)
- Duration: P29DT4H (29 日 4 時間 = 6/7 17:00 まで継続)
- Enabled: True

---

## 要件 3: 朝一 全頭スコア (9:30 → 9:00)

### 現状 (修正前)
- `Keiba-SaveAllHorseScores_0930`: 9:30 fire (Sat/Sun)
- 5/10 9:30 既 fire 済 (Last Run 2026/05/10 9:30:00)

### 修正
`schtasks /Change /TN Keiba-SaveAllHorseScores_0930 /ST 09:00`
→ Next Run 2026/05/16 9:00:00

### 注意
- DailyPredict_0800 完了想定 ~8:56 → 9:00 fire は marginal (4 分 buffer)
- 5/16 からは 9:00 fire (今週末は既 9:30 fire 済、 影響なし)
- 名前は `_0930` のまま (実態と乖離するが schtask rename 避ける)

---

## 要件 4: 5/10 当日 動作確認

### 状態
- ✅ 京都通知: 14:00 以降 R で復活 (race_auto_notify.py 修正済)
- ✅ Stage 2: 12:00 schtask fire 期待、 14:00 以降 R 30 分前予測 fire
- ✅ 朝一全頭スコア: 9:30 既 fire 済 (Session #71 動作中)、 5/16 から 9:00
- ✅ Discord 通知: notify.py main 既存 (修正不要)

### 14:00 投票判断
- 8:45 朝候補 (体重未統合): 8 R
- 13:30+ 30 分前予測 (体重統合 Stage 2): 候補 update 期待
- 最終投票判断: 30 分前予測 base 推奨

---

## V15 投資保護 確認

✅ V15 model file 不変 (`keiba_model_v15_*.pkl.gz` 触らず)
✅ predict_core.py 不変
✅ daily_predict.py 不変
✅ app.py 不変
✅ 既存 dev branch 不変 (file checkout のみ、 commit せず)
✅ 累計 +¥14,140 維持

---

## ファイル変更 list

| file | 変更 |
|------|------|
| tools/race_auto_notify.py | 京都 filter 削除 (line 183-186) |
| tools/stage2_predict.py | 新規追加 (dev/two-stage より) + RACE_START_TIMES 5/10 entries 追加 |
| tools/race_day_weight_features.py | 新規追加 (dev/two-stage より) |
| schtasks: PreRacePredict_Watchdog_5_9 | DISABLED → ENABLED |
| schtasks: SaveAllHorseScores_0930 | 9:30 → 9:00 (次回 5/16) |

---

## Discord 通知

完了後 1 通送信 (#updates channel、 dedup 適用)。
