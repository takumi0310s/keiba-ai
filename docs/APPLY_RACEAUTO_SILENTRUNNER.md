# RaceAutoNotify console-kill 対策 適用チェックリスト(管理者1回)

> **目的**: per-race 通知タスク(RaceAutoNotify_Sat/Sun)の途中死亡(5/30 の `^C` console-kill → 早い6R取りこぼし)を根絶する。
> **対策**: 起動を可視コンソール直起動 → 隠し窓(`wscript silent_runner.vbs`)に変更(DailyPredict と同方式 = console-kill 免疫)。
> **性質**: 本番タスクの**起動様式のみ**変更。bat・予測ロジック・通知経路の中身は不変。可逆。
> **期限**: 次の開催 = **土曜 6/6** まで。間に合わなくても朝 8:00 DailyPredict の全R買い目通知が保険。

---

## 0. 事前検証(Claude 実施済 2026-06-05)

非開催日(6/5 金)に、スケジューラと同一の起動方式で一時起動して確認済み:

```
wscript.exe "C:\Users\takum\keiba-ai\tools\silent_runner.vbs" "C:\Users\takum\keiba-ai\race_auto_notify.bat"
```

- ✅ silent_runner 経由で bat 起動・ログ形式は直起動(5/31)と**完全一致**。
- ✅ `End (exit=0)` 行まで出力 = **正常完走**(5/30 の console-kill 時はこの行が欠落していた)。
- ✅ cwd 正常(相対 `logs/` に出力)・env 正常(UTF-8/unbuffered)・ExitCode=0 伝播・プロセス残留なし。
- ✅ 非開催日で **誤通知なし**(`No races today` で即終了、Discord を呼ばない)。
- silent_runner は `Run(arg, 0, True)` = 隠し窓(0)+完了待ち(True)で結果コードを正しく返す。

---

## 1. 適用(★管理者権限が必要 / RunLevel=Highest★)

`Win+X` →「**ターミナル(管理者)**」または「Windows PowerShell(管理者)」を開き、1行実行:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\apply_raceauto_silentrunner.ps1
```

成功すると Sat/Sun 両方に次が出る:

```
OK RaceAutoNotify_Sat: Execute=[wscript.exe] Args=["...\silent_runner.vbs" "...\race_auto_notify.bat"]
OK RaceAutoNotify_Sun: Execute=[wscript.exe] Args=["...\silent_runner.vbs" "...\race_auto_notify.bat"]
DONE. RaceAutoNotify Sat/Sun は隠し窓起動(console-kill免疫)になりました。
```

> 注: セッション経由の UAC 昇格は繰り返しキャンセルされたため、**手動で管理者ターミナルを開く**方式が確実。

---

## 2. 適用後の検証

管理者でなくてよい。以下で Action が `wscript.exe` に変わったか確認:

```powershell
foreach ($n in 'RaceAutoNotify_Sat','RaceAutoNotify_Sun') {
  $t = Get-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\'
  "{0}: Execute=[{1}] Args=[{2}]" -f $n, $t.Actions[0].Execute, $t.Actions[0].Arguments
}
```

期待: `Execute=[wscript.exe]` / `Args=["...\silent_runner.vbs" "...\race_auto_notify.bat"]`。

(任意・即時動作確認)非開催日なら誤通知ゼロで安全なので、§0 のコマンドを1回流して `logs\race_auto_notify_<today>.log` に `End (exit=0)` が出れば OK。

---

## 3. 失敗時 / 元に戻す(revert・要管理者)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\revert_raceauto_silentrunner.ps1
```

→ 元の `Execute=[C:\Users\takum\keiba-ai\race_auto_notify.bat]`(可視コンソール直起動)に戻る。

| 症状 | 対処 |
|------|------|
| `Access is denied` | 管理者ターミナルでない。Win+X → ターミナル(管理者)で再実行。 |
| `The system cannot find the file specified` | `-TaskPath '\keiba-ai\'` が必要(script内で指定済。手打ち時は注意)。 |
| 適用後 per-race が動かない | revert で元に戻し、`logs\race_auto_notify_*.log` を確認。 |

---

## 4. 背景(なぜ起きたか)

- 5/30 は **8:45 定刻起動していた**(「起動遅れ」は誤診)。ログ1行目が 8:45 Start。
- その後 `^C`(コンソール制御イベント = Ctrl+C / ウィンドウ閉じ / ログオフ)で**途中死亡** → 10:55 手動再起動(= LastRunTime)→ 早い6R の 5分前タイマーが消滅し取りこぼし。
- 可視コンソールに紐付く直起動が唯一の脆弱点。隠し窓化で根治。
- 詳細: `docs/SESSION_LEAK_AUDIT_S2B.md` §4.2 / メモリ `keiba-raceautonotify-console-kill`。
