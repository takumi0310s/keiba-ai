# JV-Link SE record fetcher (32-bit PowerShell)
# Run with: C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -File tools\jv_read_se.ps1

param(
    [string]$fromDate  = "20260301",
    [int]   $maxRecs   = 300,
    [string]$outFile   = "C:\Users\takum\keiba-ai\data\jvlink\se_raw.txt"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$fromTime = $fromDate + "000000"

Write-Host "[jv_read_se] from=$fromDate maxRecs=$maxRecs"

$jv = New-Object -ComObject "JVDTLab.JVLink"
$rcInit = $jv.JVInit("UNKNOWN/0.0")
Write-Host "[jv_read_se] JVInit rc=$rcInit"

if ($rcInit -ne 0) {
    Write-Host "[ERROR] JVInit failed rc=$rcInit"
    exit 1
}

# JVOpen with [ref] for BYREF params
$nData   = [ref]([int]0)
$nFiles  = [ref]([int]0)
$lastTime = [ref]("")

$rcOpen = $jv.JVOpen("RACE", $fromTime, 1, $nData, $nFiles, $lastTime)
Write-Host "[jv_read_se] JVOpen rc=$rcOpen nData=$($nData.Value) nFiles=$($nFiles.Value)"

if ($rcOpen -lt 0) {
    Write-Host "[ERROR] JVOpen failed rc=$rcOpen"
    $jv.JVClose()
    exit 1
}

# Output file
$writer = [System.IO.StreamWriter]::new($outFile, $false, [System.Text.Encoding]::Default)

$nTotal = 0
$nSE    = 0

# JVRead with [ref] params
$buff   = [ref]("")
$size   = [ref]([int]2048)
$fname  = [ref]("")

while ($nSE -lt $maxRecs) {
    $rcRead = $jv.JVRead($buff, $size, $fname)

    if ($rcRead -eq 0)  { break }
    if ($rcRead -eq -1) { continue }
    if ($rcRead -lt 0)  { Write-Host "[ERROR] JVRead rc=$rcRead"; break }

    $nTotal++
    $rec = $buff.Value
    if ($rec.Length -ge 2 -and $rec.Substring(0,2) -eq "SE") {
        $writer.WriteLine($rec)
        $nSE++
        if ($nSE % 100 -eq 0) {
            Write-Host "  SE: $nSE / total: $nTotal"
        }
    }
}

$writer.Close()
$jv.JVClose()

Write-Host "[jv_read_se] Done: SE=$nSE total=$nTotal"
Write-Host "[jv_read_se] Output: $outFile"
