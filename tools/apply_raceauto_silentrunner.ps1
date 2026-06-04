# RaceAutoNotify_Sat/Sun を 可視コンソール直起動 → 隠し窓(wscript silent_runner.vbs)に変更
# 目的: console-kill(Ctrl+C/閉じる/ログオフ)による途中死亡(5/30の取りこぼし)を根絶。
# 可逆: 元Action = race_auto_notify.bat 直接・引数なし。戻す場合は tools/revert_raceauto_silentrunner.ps1。
$ErrorActionPreference = 'Stop'
$vbs = 'C:\Users\takum\keiba-ai\tools\silent_runner.vbs'
$bat = 'C:\Users\takum\keiba-ai\race_auto_notify.bat'
$newArgs = '"' + $vbs + '" "' + $bat + '"'
$act = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument $newArgs
foreach ($n in 'RaceAutoNotify_Sat','RaceAutoNotify_Sun') {
  Set-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\' -Action $act | Out-Null
  $t = Get-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\'
  Write-Host ("OK {0}: Execute=[{1}] Args=[{2}]" -f $n, $t.Actions[0].Execute, $t.Actions[0].Arguments)
}
Write-Host "DONE. RaceAutoNotify Sat/Sun は隠し窓起動(console-kill免疫)になりました。"
