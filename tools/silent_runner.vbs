' Silent batch runner
' Args: Arg(0) = bat ファイルパス
'       Arg(1..N) = 元の bat に渡す追加引数
' wscript.exe で起動するとコンソールウィンドウは表示されない (window-style 0)
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
' Run(command, intWindowStyle, bWaitOnReturn)
' 0 = hidden, True = wait for completion (タスクスケジューラの結果コードを正しく返す)
WScript.Quit shell.Run(argLine, 0, True)
