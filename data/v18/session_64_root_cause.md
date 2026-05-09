# Session #64 B: root cause 特定

## 1. 真の原因 — tools/process_watchdog_v2.py の 2 件 logic bug

### Bug 1: process_match の path separator 不一致

```python
# tools/process_watchdog_v2.py L66, L74 (修正前)
process_match='tools\\daily_predict.py',     # Windows backslash
process_match='tools\\race_auto_notify.py',  # 同上
```

実プロセスの cmdline:
```
python -u tools/daily_predict.py --resume     # forward slash
python -u tools/race_auto_notify.py           # 同上
```

`process_alive()` は PowerShell CIM で `CommandLine -like '*tools\\X.py*'` 比較 → **常に False**。

5/9 12:35 時点で 12 件の race_auto_notify.py プロセスが running していたが、 watchdog はそれらを完全に見落とし → MISSING 判定。

### Bug 2: COMPLETED ケースを MISSING に誤分類

```python
# L233-238 (修正前)
if alive and not stale:    status = 'ALIVE'
elif alive and stale:      status = 'STALE'
else:                      status = 'MISSING'   # ← ここに (alive=False, stale=False) が入る
```

ワンショット daily_predict.py の典型 lifecycle:
1. fire (8:00 schtask 経由)
2. ~30 分 実行
3. 終了 (process 消滅、 ログは新鮮)
4. 5 分後 watchdog: `alive=False (Bug 1 もあって正確判定不可)`、 `stale=False (mtime < 60min)`
5. → MISSING 判定 → restart_target() → 再起動

これが **5 分毎に永続 loop**。 daily_predict --resume mode は CSV を見て 34 R skip → cookie check →「20260509 のレース見つからず」→ ただし「整形済み買い目通知送信: 8 messages」を fire (Discord spam の正体)。

## 2. なぜ今日 (5/9) 顕在化したか

- daily_predict は通常 8:00 schtask で 1 回 fire → 完了 → watchdog は 60 min stale を待ってから再起動判定
- しかし 5/9 9:30+ で daily_predict_watchdog_restart_20260509.log が誰かに touch されて mtime が新鮮化 (おそらく 8:56 終了時の追記)
- 11:48 watchdog: alive=False (Bug 1), stale=False (mtime 8:56 → 12:48 > 60min) ← あれ、 stale=True になるはず

→ 詳細は `logs/watchdog_v2_20260509.log` に `mtime=1778294589` 等の epoch あり、 `2026-05-09 11:43` 付近。 watchdog 自身が restart で書き出した log の mtime を「daily_predict のログ」として拾っている可能性 (`daily_predict_watchdog_restart_*.log` も glob `daily_predict*.log` にマッチ → 自己 mtime ループ)。

→ Bug 3 候補: glob が watchdog 自身の restart log を含む → restart した瞬間 mtime fresh → 次回 stale=False → また restart。 自己持続 loop。

## 3. なぜ 5/9 だけか

- 前日まで 5/8 23:00 NightlySanity が 翌日タスクのプリチェックで何かを触り、 たまたま log mtime を 30 min 圏外に押し出していた？
- 5/9 朝 daily_predict が 8:56 完了 (8:00 fire + 56 min) → 11:48 watchdog で stale_sec=3600 に対して mtime=8:56 → diff 約 172 min → stale=True のはず
- それでも MISSING に flow した = `(not alive and stale)` = MISSING — そして restart → restart log を 11:48 に書く → 次 watchdog 11:53 で mtime 11:48 → stale=False (5min) → COMPLETED 扱いになるはずだったが Bug 2 で MISSING → 再 restart …

→ 5/9 朝の 8:56 終了タイミング + 11:48 watchdog 起動の組み合わせが trigger。 1 回 restart loop に入ると Bug 2 の自己持続で延々続く。

## 4. 確定 root cause

**Bug 2 (COMPLETED 誤分類) が主因**、 Bug 1 (path) が副因 (アライブな race_auto_notify の存在も拾えず加速)。 Bug 3 (自己 mtime loop) は Bug 2 解消で結果的に消える (restart しなくなれば log mtime も自然 stale 化)。

→ 修正方針: `(not alive and not stale)` → `COMPLETED` 状態を新設し restart skip。 同時に path separator 修正。

main 不変 (dev/training-poc 専用 commit のみ)。 V15 投資保護 維持。
