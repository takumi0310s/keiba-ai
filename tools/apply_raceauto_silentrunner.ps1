# Switch RaceAutoNotify_Sat/Sun from visible-console launch to hidden window (wscript silent_runner.vbs).
# Purpose: prevent mid-run death from console-kill (Ctrl+C / window close / logoff) -- the 5/30 miss.
# Reversible: original Action = race_auto_notify.bat (no args). To revert: tools/revert_raceauto_silentrunner.ps1.
# Note: only the Action is changed. Trigger / Settings / Principal are preserved (Set-ScheduledTask -Action).
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
Write-Host "DONE. RaceAutoNotify Sat/Sun now launch via hidden window (console-kill immune)."
