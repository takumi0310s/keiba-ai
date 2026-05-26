# JV-Link RA lap full fetch: RACE datatype → lap times TSV (confirmed offset=750)
# 2015-2019 full backfill for V20 training data (lap 0% fill in those years)
# Run with 32-bit PowerShell:
#   C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -File tools\jv_ra_lap_full_fetch.ps1

param(
    [string]$fromDate = "20150101",
    [string]$toDate   = "20191231",     # for logging only (JVOpen fetches all from fromDate)
    [string]$outDir   = "C:\Users\takum\keiba-ai\data\v20",
    [int]   $maxRecs  = 5000000
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$fromTime = $fromDate + "000000"
# For 2015-2019 only, use SETUP (option=3) to get historical data
# Then filter by date in Python processing step

Write-Host "[jv_ra_lap_full] from=$fromDate to=$toDate outDir=$outDir"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$jv = New-Object -ComObject "JVDTLab.JVLink"
$rcInit = $jv.JVInit("UNKNOWN/0.0")
Write-Host "[jv_ra_lap_full] JVInit rc=$rcInit"
if ($rcInit -ne 0) { Write-Host "[ERROR] JVInit failed"; exit 1 }

$nData = [ref]([int]0); $nFiles = [ref]([int]0); $lastTime = [ref]("")
# option=4 (same as probe script — option=1 returns -402 error with 0 RA records)
$rcOpen = $jv.JVOpen("RACE", $fromTime, 4, $nData, $nFiles, $lastTime)
Write-Host "[jv_ra_lap_full] JVOpen rc=$rcOpen nData=$($nData.Value) nFiles=$($nFiles.Value)"
if ($rcOpen -lt 0) { Write-Host "[ERROR] JVOpen failed rc=$rcOpen"; $jv.JVClose(); exit 1 }

$buff = [ref](""); $size = [ref]([int]32768); $fname = [ref]("")
$nTotal = 0; $nRA = 0; $nSkip = 0

$outFile = Join-Path $outDir "ra_laps_${fromDate}_${toDate}.tsv"
$writer = [System.IO.StreamWriter]::new($outFile, $false, [System.Text.Encoding]::UTF8)
$writer.WriteLine("race_date`tcourse_code`tkai`tnichi`trace_num`tlap_times`tn_laps`tsum_sec")

$lastReport = [System.DateTime]::Now

while ($nTotal -lt $maxRecs) {
    $rcRead = $jv.JVRead($buff, $size, $fname)
    if ($rcRead -eq 0)  { Write-Host "[jv_ra_lap_full] EOF"; break }
    if ($rcRead -eq -1) { continue }
    if ($rcRead -lt 0)  {
        if ($rcRead -eq -3) { $size = [ref]([int]65536); $nSkip++; continue }
        Write-Host "[ERROR] JVRead rc=$rcRead"; break
    }
    $nTotal++
    $rec = $buff.Value

    if ($rec.Length -lt 2 -or $rec.Substring(0, 2) -ne "RA") { continue }
    if ($rec.Length -lt 805) { continue }   # need at least offset 804 for 18 laps

    # DataKubun [2]: 1=new/8=deleted. Skip deleted (7=confirmed).
    # Actually DataKubun=7 means "成績確定" which is what we want
    $dkbn = $rec.Substring(2, 1)
    if ($dkbn -eq "0") { continue }   # skip empty

    # Race identifier from RA record (confirmed offsets from probe)
    $raceDate   = $rec.Substring(11, 8)   # YYYYMMDD
    $courseCode = $rec.Substring(19, 2)
    $kai        = $rec.Substring(21, 2)
    $nichi      = $rec.Substring(23, 2)
    $raceNum    = $rec.Substring(25, 2)

    # Filter: only 2015-2019 races
    $yr = [int]$raceDate.Substring(0, 4)
    if ($yr -lt 2015 -or $yr -gt 2019) { continue }

    # Parse laps from offset 751 (confirmed: byte at 750 is non-lap field), 3 chars each, max 18 laps
    $laps = @()
    $pos = 751
    while ($pos + 3 -le $rec.Length) {
        $chunk = $rec.Substring($pos, 3)
        if ($chunk -eq "000") { break }
        $v = 0
        if ([int]::TryParse($chunk, [ref]$v) -and $v -ge 90 -and $v -le 200) {
            $laps += $v
        } else {
            break
        }
        $pos += 3
        if ($laps.Count -ge 18) { break }
    }

    if ($laps.Count -lt 3) { continue }  # skip if too few laps

    $lapStr = $laps -join ","
    $sumSec = ($laps | Measure-Object -Sum).Sum
    $writer.WriteLine("$raceDate`t$courseCode`t$kai`t$nichi`t$raceNum`t$lapStr`t$($laps.Count)`t$sumSec")
    $nRA++

    $now = [System.DateTime]::Now
    if (($now - $lastReport).TotalSeconds -ge 30) {
        Write-Host "  RA(written)=$nRA total=$nTotal skip=$nSkip date=$raceDate"
        $lastReport = $now
    }
}

$writer.Close()
$jv.JVClose()

Write-Host "[jv_ra_lap_full] Done: RA=$nRA total=$nTotal skip=$nSkip"
Write-Host "[jv_ra_lap_full] Output: $outFile"
Write-Host ""
Write-Host "=== Next step ==="
Write-Host "Run: python train/build_ra_lap_features.py"
