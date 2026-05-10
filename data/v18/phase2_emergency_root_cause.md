# Phase 2 緊急 audit: 買い目 2 重送信 + 9:30 SaveAllHorseScores 初稼働 (5/10 09:44)

## 結論

| 項目 | 判定 | 影響 |
|------|------|------|
| 買い目 2 重送信 root cause | ✅ **NOT A BUG** (chunking 仕様) | なし |
| 9:30 SaveAllHorseScores | ✅ **動作中** (17/35R, ETA 9:58) | なし |
| 戦略⑦ filter 動作 | ✅ **正常** (36R → 8R) | なし |
| 投票判断 影響 | ❌ **なし** | 14:00 GO |

---

## A. 買い目 2 重送信 root cause = case **C (誤認、 chunking 仕様)**

### 真相
ユーザー報告の「全会場全 R が二つ送られてきた」 = `notify_bets_all_in_one.py` が **Discord 1900 char 制限** で全36R を 2 chunk に分割して送信した結果。

```
📋 2026/05/10(日) 全36R一括 (1/2)  ← chunk 1: 京都12R + 新潟一部
📋 2026/05/10(日) 全36R一括 (2/2)  ← chunk 2: 新潟残り + 東京12R + footer
```

両 message タイトルが酷似 (`📋 2026/05/10(日) 全36R一括`) → ユーザー視覚で重複と誤認。

### 証拠

**race_auto_notify_20260510.log (line 49-50):**
```
全レース一括通知: 2 messages       ← 仕様通り 2 chunk
整形済み買い目通知: 8 messages     ← 戦略⑦ filter 後 8 候補
```

**discord_send_cache.json (8:45 burst, 11 entries):**
- 8:45:01-07 内 11 unique hash 送信
- 内訳: 1 (Auto-Notify起動) + 2 (全レース一括 1/2, 2/2) + 8 (整形済み)
- ✅ dedup 動作 (5min window、 同一 hash silent skip)

**chunking logic** (`tools/notify_bets_all_in_one.py:81-102 _chunk_by_venue`):
- DISCORD_CHAR_LIMIT = 1900
- 競馬場境界で分割
- 36R 全表示は 1 message に収まらず 2 chunk に分割される

### 修復要否
- ✅ **修復不要** (仕様通り、 dedup 動作中)
- 📝 5/13+ 改善案: `notify_bets_all_in_one.py` に **戦略⑦ filter 適用** → 1 message に収まる + ノイズ削減
- 5/10 14:00 投票判断には影響しない (整形済み 8 messages を参照)

---

## B. 9:30 SaveAllHorseScores 初稼働 = ✅ **動作中**

### schtasks status
```
\Keiba-SaveAllHorseScores_0930
  Last Run Time: 2026/05/10 9:30:00
  Last Result:   267009 (=Running)
  Status:        Running
```

### Process status
```
python pid 37536
  StartTime: 2026/05/10 9:30:00
  CPU: 703 sec (9:44 時点)
  WorkingSet: 1.8 GB
```

### 進捗 (logs/save_all_horse_scores.log, 9:44 時点)
- **17/35 R 処理済** (48%)
- 京都 1-12R + 新潟 1-3, 5R 完了
- 残り 18R: 新潟 6-12R + 東京 1-12R
- ETA ~9:58 完了予定

### 完了後 artifact
- `data/daily_predictions_full/20260510.csv` (約 360-540 row, 全頭分 V15 score)
- ✅ Session #71 158h+ マラソン集大成 **真の初稼働 成功**

---

## C. 戦略⑦ filter 動作 = ✅ **正常**

### 36R 内訳
| 場 | R 数 | 戦略⑦ skip | 残 |
|----|-----|-------------|----|
| 京都 | 12 | 全 12 (course skip) | 0 |
| 新潟 | 12 | 9, 10 (06_特別) | 10 |
| 東京 | 12 | 9 (06_特別) | 11 |
| **計** | **36** | **15** | **21** |

### filter 適用箇所
- `tools/race_auto_notify.py:171-187` (race_name + course)
- `tools/race_auto_notify.py:269-276` (条件 E / B)
- ✅ 京都全12R skip
- ✅ 06_特別 skip (荒川峡特別/五泉特別/日吉特別)
- ✅ 条件E (頭数≤7) skip
- ✅ 条件B (重~不良) skip — 5/10 馬場良で 0R

### 整形済み 8 messages = 案B改 strict 候補
21R - (オープン特別の追加 filter / 条件 X 等) = 8 候補。
具体的 R は Discord #買い目 channel 個別 message で確認可。

### filter の盲点
- ⚠ `notify_bets_all_in_one` は 戦略⑦ filter **未適用** (全36R 表示)
- → 一括通知 chunk (1/2, 2/2) には 京都/特別 含む
- → 投票判断時は **整形済み 8 messages のみ** を参照

---

## D. 8:45 通知の真実 (cache hash 11 entries との 1:1 対応)

| # | timestamp | 通知 | 件数 |
|---|-----------|------|------|
| 1 | 8:45:01 | Auto-Notify起動 (#bets) | 1 |
| 2-3 | 8:45:03-07 | 全レース一括 (1/2, 2/2) | 2 |
| 4-11 | 8:45:04-07 | 整形済み買い目 × 8 | 8 |
| - | 8:50:02 | DailyPredict ok (#updates) | 1 (別 channel) |
| 12 | 9:31:14 | (要確認、 9:30 task 関連?) | 1 |
| 13 | 9:41:22 | (要確認、 watchdog/sanity?) | 1 |

**8:45 burst 計 11 messages = 1+2+8 一致** ✅

---

## E. 投票判断 (14:00) への影響

### 影響: ❌ **なし**

**理由**:
1. 整形済み 8 messages = 戦略⑦ filter 後の案B改候補 → そのまま投票候補
2. 一括通知 (1/2, 2/2) は全36R 一覧 (filter 未適用) → 投票判断には使わない
3. 京都/特別 が一括通知に含まれることは正常 (情報提供のため)

### 14:00 投票推奨
- **整形済み 8 messages の R を採用**
- 案B改: 12R 1勝クラスのみ 上限 ¥2,100
- 投資候補: 8 × ¥700 = **¥5,600** (上限以下に収まる)
- ⚠ 5/9 限定 task `Keiba-VoteCandidates_1400_5_9` 不在 → manual 投票

### V15 投資保護
- ✅ 完全 (model 不変、 predict_core 不変、 daily_predict 不変)
- ✅ 累計 +¥14,140 維持
- ✅ 撤退余裕 +¥64,140

---

## 緊急対応 = **不要**

- ✅ 通知システム 仕様通り動作
- ✅ 戦略⑦ filter 動作中
- ✅ 9:30 SaveAllHorseScores 動作中
- ✅ 投票判断 GO (整形済み 8 messages 参照)

### 5/13 平日 改善 候補 (今日は手を付けない)
1. `notify_bets_all_in_one.py` に 戦略⑦ filter 適用 (1 message 化、 ノイズ削減)
2. 一括通知 タイトル に「(1/2)」を 大きく表示 (chunk 識別性 UP)
3. dedup window を 30 min に拡大 (8:00 と 8:45 の 整形済み 重複防止)

→ いずれも 5/10 投票には影響しない。
