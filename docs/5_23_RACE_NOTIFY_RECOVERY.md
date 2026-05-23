# 5/23 race_auto_notify Recovery

**確認時刻**: 2026-05-23 09:53〜09:54  
**結果**: 再起動成功 ✅

---

## race_auto_notify 生存状態 (09:53)

| 項目 | 状態 |
|------|------|
| python プロセス | **なし** (停止確認) |
| 停止原因 | log 末尾の `^C` = プロセス中断 (schtask timeout または手動停止) |
| 停止タイミング | 08:45〜09:40 の間 (個別 R エントリなし) |

---

## 既通知 / 未通知 R (09:53 時点)

| カテゴリ | R 数 | レース |
|---------|------|-------|
| 通知ミス (発走済) | **2 R** | 新潟1R 09:45 / 東京1R 09:55 |
| 残り (再起動で対応) | **34 R** | 京都1R 10:05 〜 京都12R 16:30 |

**通知ミス 2R は投票不可** (発走時刻通過済のため)

---

## 再起動結果 ✅

**コマンド**:
```bash
C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe -u tools\race_auto_notify.py >> logs\race_auto_notify_20260523.log 2>&1 &
```

**起動 log**:
```
Race Auto-Notify: 20260523
Found: 36 races

  新潟1R 09:45: already passed  ← 正しく skip
  東京1R 09:55: already passed  ← 正しく skip
  京都1R 10:05: notify at 10:00 (in 6min)
  ...
  京都12R 16:30: notify at 16:25 (in 391min)

Active timers: 34  ✅ (36 - 2 passed = 34)
全レース一括通知: 2 messages
整形済み買い目通知: 8 messages
Waiting for races... (Ctrl+C to stop)
```

**プロセス確認 (09:54)**:
```
python.exe    28500  Console  1  108,824 K
```
→ PID 28500 で稼働中 ✅

---

## 残り R 買い目通知

| 時刻 | レース | 通知 |
|------|-------|------|
| 10:00 | 京都1R | ⏰ 次の通知 (6 分後) |
| 10:15 | 新潟2R | — |
| 10:25 | 東京2R | — |
| ... | ... | — |
| 16:25 | 京都12R | — |

**34 R 全て戦略⑦案 C + C4 フィルタ適用後に通知**。

---

## 戦略⑦案 C + C4 適用確認

race_auto_notify.py は変更していない。起動後のフィルタロジックは前回と同一:
- 戦略⑦案 C: 06_特別 / 京都 / 条件E / 条件B / 条件X (non-graded) 除外
- C4: 条件A × 1600-1800m 除外

整形済み買い目通知 8 messages = フィルタ後の通知数 (前回起動と同数 → 正常)

---

## 総括

| 項目 | 結果 |
|------|------|
| race_auto_notify 停止原因 | 不明 (^C、schtask timeout 疑い) |
| 再起動 | **成功** |
| 通知ミス | **2 R** (新潟1R + 東京1R) |
| 残り R 通知 | **34 R 対応可** |
| 初回通知 (10:00 京都1R) | 再起動後 6 分以内 |
| V15 production | **不変** |

---

*Recovery: 2026-05-23 09:54 | V15 production 不変*
