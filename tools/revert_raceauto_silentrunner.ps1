# Revert RaceAutoNotify_Sat/Sun back to the original visible-console launch (race_auto_notify.bat).
# Emergency use only. Requires administrator. Only the Action is changed; Trigger/Settings/Principal preserved.
$ErrorActionPreference = 'Stop'
$act = New-ScheduledTaskAction -Execute 'C:\Users\takum\keiba-ai\race_auto_notify.bat'
foreach ($n in 'RaceAutoNotify_Sat','RaceAutoNotify_Sun') {
  Set-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\' -Action $act | Out-Null
  $t = Get-ScheduledTask -TaskName $n -TaskPath '\keiba-ai\'
  Write-Host ("REVERTED {0}: Execute=[{1}] Args=[{2}]" -f $n, $t.Actions[0].Execute, $t.Actions[0].Arguments)
}
