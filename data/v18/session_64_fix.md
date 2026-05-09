# Session #64 C: 修正実装

## 1. 即時 出血止め (12:38 完了)

### 1-1. Lingering process kill (12:36)

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*race_auto_notify.py*' -or
                 $_.CommandLine -like '*daily_predict.py*' } |
  Stop-Process -Force
```

12 件の race_auto_notify.py プロセスを kill。 daily_predict.py は当時 idle で対象なし。

### 1-2. Kill-switch file (Admin 不要)

`\ProcessWatchdog` schtask の `/Change /DISABLE` は Access denied (Admin 必要)。
代替として code-level kill switch を実装:

```python
# tools/process_watchdog_v2.py main() 冒頭
kill_switch = os.path.join(BASE_DIR, 'data', 'v18', 'process_watchdog_v2.kill')
if os.path.exists(kill_switch):
    print(f"[watchdog_v2] kill-switch active ({kill_switch}) → no-op exit")
    return
```

`data/v18/process_watchdog_v2.kill` を touch 済。 12:43 以降の ProcessWatchdog fire は no-op で抜ける。

## 2. logic bug 修正 (恒久対応)

### 2-1. Bug 1 path separator

```diff
-process_match='tools\\daily_predict.py',
+process_match='tools/daily_predict.py',
-process_match='tools\\race_auto_notify.py',
+process_match='tools/race_auto_notify.py',
```

cmdline は POSIX slash で起動されるため (subprocess.Popen 経由)、 一致条件もそれに揃える。

### 2-2. Bug 2 COMPLETED 状態追加

```diff
 if alive and not stale:
     status = 'ALIVE'
 elif alive and stale:
     status = 'STALE'
+elif (not alive) and (not stale):
+    # Session #64 fix: 直近に正常終了したワンショット
+    status = 'COMPLETED'
 else:
     status = 'MISSING'
```

run_once() の continue 条件を拡張:
```diff
-if r['status'] == 'ALIVE':
+if r['status'] in ('ALIVE', 'COMPLETED'):
     continue
```

→ COMPLETED は restart も Discord 警告も発火しない。 真に MISSING (process なし、 ログも stale_sec 以上古い) の時のみ restart。

## 3. 動作確認

```
$ python -c "import sys; sys.argv=['p','--once']; sys.path.insert(0,'tools'); \
             import process_watchdog_v2; process_watchdog_v2.main()"
[watchdog_v2] kill-switch active (.../process_watchdog_v2.kill) → no-op exit
```

OK。

## 4. 残課題 (今日は触らない)

- `\ProcessWatchdog` schtask の正常化:
  - 5/16 (土) 朝に Admin で `schtasks /Change /TN "\ProcessWatchdog" /ENABLE` 必要なし (Disable してない)
  - kill-switch file を delete すれば即時 fix 版 logic で再有効化
- daily_predict.py の「整形済み買い目通知送信: 8 messages」の dedup
  - 「20260509 のレース見つからず」シナリオで messages を送らない条件追加
  - Session #65 候補
- race_auto_notify.py の `Found: 0 races` 早期 exit 後 process 残存
  - 別 scope、 Session #65 で別途調査

## 5. 安全性チェック

- main / V15 model file / predict_core.py / app.py / daily_predict.py / race_auto_notify.py: 全て触っていない
- 変更は `tools/process_watchdog_v2.py` (1 file) と `data/v18/process_watchdog_v2.kill` (新規) のみ
- Session #61 で登録した 9 件 schtasks (vote_candidates 14:00 / verdict×6 / cumulative 17:00 / summary 20:30) には影響なし
- 5/9 投票方針 不変 (新潟 12R ¥700 のみ)

## 6. 効果検証

next ProcessWatchdog fire = 12:43 で kill-switch hit → no-op。 13:00 以降の watchdog log で `[watchdog_v2] kill-switch active` 行を確認すれば正常停止。
