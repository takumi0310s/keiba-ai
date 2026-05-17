# P0-5 Live Orchestrator PowerShell wrapper
# (★ 5/18+ admin schtask から fire される代替版 ★)

Set-Location 'C:\Users\takum\keiba-ai'

# log dir 確保
$logDir = 'data\live_orchestrator_log'
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# mock 安全運用 (5/24 以降 user 判断で 解除予定)
& python -u tools\live_orchestrator_main.py --mock --dry-run *>> data\live_orchestrator_log\stdout.log
