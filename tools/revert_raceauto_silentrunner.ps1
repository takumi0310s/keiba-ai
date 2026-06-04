# RaceAutoNotify_Sat/Sun を 元の可視コンソール直起動(race_auto_notify.bat)に戻す(緊急時用・要管理者)
$ErrorActionPreference = 'Stop'
$act = New-ScheduledTaskAction -Execute 'C:\Users\takum\keiba-ai\race_auto_notify.bat'
foreach ($n in 'RaceAutoNotify_Sat','RaceAutoNotify_Sun') {
  Set-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\' -Action $act | Out-Null
  $t = Get-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\'
  Write-Host ("REVERTED {0}: Execute=[{1}] Args=[{2}]" -f $n, $t.Actions[0].Execute, $t.Actions[0].Arguments)
}
