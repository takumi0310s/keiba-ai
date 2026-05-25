' JV-Link SE record fetcher (VBScript, 32-bit COM)
' Usage: C:\Windows\SysWOW64\cscript.exe tools\jv_read_se.vbs [fromdate] [maxrecords] [outfile]
' Example: C:\Windows\SysWOW64\cscript.exe tools\jv_read_se.vbs 20260301 500 data\jvlink\se_raw.txt

Option Explicit

Dim fromDate, maxRecs, outFile
fromDate = "20260301"
maxRecs  = 500
outFile  = "C:\Users\takum\keiba-ai\data\jvlink\se_raw.txt"

If WScript.Arguments.Count >= 1 Then fromDate = WScript.Arguments(0)
If WScript.Arguments.Count >= 2 Then maxRecs  = CInt(WScript.Arguments(1))
If WScript.Arguments.Count >= 3 Then outFile  = WScript.Arguments(2)

Dim fromTime
fromTime = fromDate & "000000"

WScript.Echo "[jv_read_se] from=" & fromDate & " maxRecs=" & maxRecs

' --- JV-Link 初期化 ---
Dim jv
Set jv = CreateObject("JVDTLab.JVLink")
Dim rcInit
rcInit = jv.JVInit("UNKNOWN/0.0")
WScript.Echo "[jv_read_se] JVInit rc=" & rcInit

If rcInit <> 0 Then
    WScript.Echo "[ERROR] JVInit 失敗 rc=" & rcInit
    WScript.Quit 1
End If

' --- JVOpen RACE ---
Dim nData, nFiles, lastTime
nData    = CLng(0)
nFiles   = CLng(0)
lastTime = ""
Dim rcOpen
rcOpen = jv.JVOpen("RACE", fromTime, CLng(1), nData, nFiles, lastTime)
WScript.Echo "[jv_read_se] JVOpen rc=" & rcOpen & " nData=" & nData & " nFiles=" & nFiles

If rcOpen < 0 Then
    WScript.Echo "[ERROR] JVOpen 失敗 rc=" & rcOpen
    jv.JVClose
    WScript.Quit 1
End If

' --- ファイル出力準備 ---
Dim fso, ts
Set fso = CreateObject("Scripting.FileSystemObject")
Set ts  = fso.CreateTextFile(outFile, True, False)

' --- JVRead ループ ---
Dim buff, fname, rcRead
Dim nTotal, nSE
nTotal = 0
nSE    = 0
Dim recType

Do While nSE < maxRecs
    buff  = ""
    fname = ""
    rcRead = jv.JVRead(buff, 2048, fname)

    If rcRead = 0 Then
        Exit Do
    ElseIf rcRead = -1 Then
        ' ファイル切替
    ElseIf rcRead < 0 Then
        WScript.Echo "[ERROR] JVRead rc=" & rcRead
        Exit Do
    Else
        nTotal = nTotal + 1
        If Len(buff) >= 2 Then
            recType = Left(buff, 2)
            If recType = "SE" Then
                ts.WriteLine buff
                nSE = nSE + 1
                If nSE Mod 100 = 0 Then
                    WScript.Echo "  SE: " & nSE & " / total: " & nTotal
                End If
            End If
        End If
    End If
Loop

ts.Close
jv.JVClose

WScript.Echo "[jv_read_se] 完了: SE=" & nSE & " total=" & nTotal
WScript.Echo "[jv_read_se] 出力: " & outFile
