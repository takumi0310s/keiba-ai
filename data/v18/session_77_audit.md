# Session #77 A: silent_runner.vbs Line 24 audit

## file: tools/silent_runner.vbs (24 lines)

```vbs
Option Explicit
Dim shell, batPath, i, argLine

If WScript.Arguments.Count = 0 Then
    WScript.Echo "Usage: silent_runner.vbs <batfile> [args...]"
    WScript.Quit 1
End If

batPath = WScript.Arguments(0)
argLine = """" & batPath & """"

For i = 1 To WScript.Arguments.Count - 1
    argLine = argLine & " """ & WScript.Arguments(i) & """"
Next

Set shell = CreateObject("Wscript.Shell")
WScript.Quit shell.Run(argLine, 0, True)   ' <-- Line 24
```

## Line 24 logic

`shell.Run(argLine, 0, True)` で wrapped batPath を起動。
- argLine 例: `"C:\Users\takum\keiba-ai\pre_race_predict_runner.bat" "--check-next-1h"`
- intWindowStyle=0 (hidden)、 bWaitOnReturn=True (同期)
- batPath 物理欠如 → ERROR_FILE_NOT_FOUND (80070002) raise → Windows Script Host popup

## 38 schtasks が silent_runner 経由

major group:
- 5/9 限定 (Verdict, Cumulative, VoteCandidates, Summary、 計 8 件)
- 週末 daily (Morning, MorningWeightCheck, MultiStagePredict, JrdbRetry、 計 12 件)
- 平日 daily (DailyPredict, DailyResults, NarDaily, MorningDigest 等、 計 14 件)
- 30 分 watchdog (PreRacePredict_Watchdog_5_9, ProcessWatchdog、 計 2 件)
- 監視 (NightlySanity, JrdbHealthCheck, AM3/6/8FireCheck、 計 7 件)

## 結論

vbs 自体は健全。 Line 24 は呼び出し先 bat 物理欠如時に popup 出力。
root cause は schtask 設定 file path と main branch file 在中の不整合。
