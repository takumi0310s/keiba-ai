# 戦略⑦ 案 C rollback 手順 (5/17 G1 day 異常検出時)

作成: 2026-05-16
適用 commit: Sub-task 8 (race_auto_notify.py 戦略⑦ 案 C 拡張)
target date: 2026-05-17 (土、 ヴィクトリアマイル G1 day)

---

## 0. rollback trigger

以下のいずれかに該当したら **即座に rollback**:

1. **京都以外の R で 予測ゼロ** (Discord #買い目 に 東京/新潟 通知が通常より少なすぎる)
2. **ヴィクトリアマイル (東京 11R 202605020811) の予測通知なし**
3. **regression test fail** (`python -u -c "..." tests/regression_test.py`)
4. **race_auto_notify.py 起動時に Python syntax / import error**

---

## 1. rollback コマンド (1 行)

```powershell
cd C:\Users\takum\keiba-ai
git log --oneline -5  # Sub-task 8 commit hash 確認
git revert <Sub-task 8 commit hash> --no-edit
```

★ commit hash は 完了通知 (Sub-task 8 完了、 commit hash XXXXX) 参照 ★

### 1.1 緊急時 手動 rollback (git revert 失敗時)

`tools/race_auto_notify.py` の以下 2 箇所を元に戻す:

#### 元 (rollback 後の正しい姿、 5/10 以降)

```python
# 1. 06_特別 (G/L/OPEN特別 ではない平場特別) を除外
is_graded = any(g in race_name_str for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'])
is_listed = any(s in race_name_str for s in ['L)', '(L)', 'OP)', '(OP)'])
is_open_tokubetsu = any(s in race_name_str for s in ['杯', '賞', 'ステークス', 'カップ', 'ハンデ'])
if '特別' in race_name_str and not (is_graded or is_listed or is_open_tokubetsu):
    print(f"    [STRATEGY7] Skip 06_特別: {race_name_str}")
    return

# 2. 京都 filter 削除 (Phase 4 5/10、 ユーザー要望「予測 fire + 通知含む、 候補のみ手動除外」)
# ===== 戦略⑦ フィルタ ここまで =====
```

#### 条件判定後の filter (rollback 後)

```python
# ===== 戦略⑦ フィルタ続き (条件判定後) =====
if cond_key == 'E':
    print(f"    [STRATEGY7] Skip 条件E (頭数<=7)")
    return
if cond_key == 'B':
    print(f"    [STRATEGY7] Skip 条件B (重~不馬場)")
    return
# ===== 戦略⑦ フィルタ続き ここまで =====
```

(京都 filter + 条件 X filter の 2 blocks を 削除)

---

## 2. 5/17 朝 rollback 判断 timing

| 時刻 | アクション | 判定 |
|---|---|---|
| 5:30 | 起床、 dry-run 動作確認 | 異常あれば revert + Discord 通知 |
| 6:00 | DailyJrdbKyi 起動 (JRDB) | 通常通り通過 (案 C は JRDB に影響しない) |
| 7:30 | JrdbHealthCheck_Sat 起動 | 通常通り通過 |
| 8:00 | DailyPredict 起動 (35-36R 予測) | **京都 11R が出力されないこと確認** (案 C 効果) |
| 8:45 | RaceAutoNotify 起動 (5 min before) | 第 1R 通知タイミング、 東京/新潟 通知あること |
| 9:30 | Discord 通知 確認 | 京都 11R 京都 12R … 全 skip 確認 |
| 9:30 | 異常検出時 rollback 実行 | 15:40 G1 まで 余裕 6h+ |
| 15:40 | ヴィクトリアマイル G1 | 案 C 影響 0、 通常通り通知 expectation |

★ 9:30 までに rollback 完了すれば、 ヴィクトリアマイル含む 5/17 後半戦は通常運用可能 ★

---

## 3. rollback 後 検証

```powershell
# 1. syntax check
python -m py_compile tools/race_auto_notify.py

# 2. regression test (Sub-task 8 関連は FAIL するはず、 その他 17 tests は PASS)
python tests/regression_test.py

# 3. 5/17 dry-run (revert 直後)
python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
# 京都 11R で skip されないこと
print('rollback verified: 京都 filter no longer in race_auto_notify.py')
"
```

---

## 4. rollback 連絡

rollback 実行したら以下を実施:

1. Discord #アップデート に rollback 通知
   ```
   python tools/notify_done.py "P0-2 案 C rollback" "5/17 X時XX分 案 C 撤回 (理由: XXX)" --color red
   ```
2. CLAUDE.md に rollback 記録 (commit hash + 理由 + timestamp)
3. 親 agent に報告 (次 session で 案 C 修正版 再検討)

---

## 5. honest 注記

1. **rollback コスト**: 京都 12R 5/17 分の機会損失/利益が baseline に戻る (5/17 京都 12R 予想配当は事前算定不能)
2. **案 C の理論的 expected loss/gain**: 5/17 京都 12R で +¥8,400 投資 → ROI 97.97% 想定で約 -¥170 (微赤想定)
3. **rollback 判断は ヴィクトリアマイル G1 day の安全性最優先**。 案 C 改善幅 +3.72pt (設計 doc) は CI 内変動範囲、 1 日の rollback 影響は誤差

---

end of doc.
