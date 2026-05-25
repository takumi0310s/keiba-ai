# JV-Link SE bulk fetch (32-bit PowerShell)
# Fetches SE records from JVOpen("RACE") and outputs to data/jvlink/se_bulk/YYYYMM.tsv
# Run with: C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -File tools\jv_bulk_se_fetch.ps1

param(
    [string]$fromDate  = "20200101",   # start date YYYYMMDD
    [string]$toDate    = "",           # end date (for logging only; JVOpen fetches all from fromDate)
    [int]   $maxRecs   = 2000000,
    [string]$outDir    = "C:\Users\takum\keiba-ai\data\jvlink\se_bulk"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$fromTime = $fromDate + "000000"

Write-Host "[jv_bulk_se_fetch] from=$fromDate maxRecs=$maxRecs outDir=$outDir"

# Ensure output directory exists
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
    Write-Host "[jv_bulk_se_fetch] Created dir: $outDir"
}

$jv = New-Object -ComObject "JVDTLab.JVLink"
$rcInit = $jv.JVInit("UNKNOWN/0.0")
Write-Host "[jv_bulk_se_fetch] JVInit rc=$rcInit"
if ($rcInit -ne 0) { Write-Host "[ERROR] JVInit failed"; exit 1 }

$nData   = [ref]([int]0)
$nFiles  = [ref]([int]0)
$lastTime = [ref]("")

$rcOpen = $jv.JVOpen("RACE", $fromTime, 1, $nData, $nFiles, $lastTime)
Write-Host "[jv_bulk_se_fetch] JVOpen rc=$rcOpen nData=$($nData.Value) nFiles=$($nFiles.Value) lastTime=$($lastTime.Value)"
if ($rcOpen -lt 0) { Write-Host "[ERROR] JVOpen failed rc=$rcOpen"; $jv.JVClose(); exit 1 }

$buff  = [ref]("")
$size  = [ref]([int]2048)
$fname = [ref]("")

$nTotal = 0
$nSE    = 0
$nSkip  = 0

# Output: single TSV file with tab-separated fields
# Fields: horse_id, race_date, course_code, kai, nichi, race_num, umaban, finish_time, agari3f
$outFile = Join-Path $outDir "se_bulk_${fromDate}.tsv"
$writer = [System.IO.StreamWriter]::new($outFile, $false, [System.Text.Encoding]::UTF8)
$writer.WriteLine("horse_id`trace_date`course_code`kai`nichi`race_num`umaban`finish_time`agari3f")

$lastReport = [System.DateTime]::Now

while ($nSE -lt $maxRecs) {
    $rcRead = $jv.JVRead($buff, $size, $fname)

    if ($rcRead -eq 0)  { Write-Host "[jv_bulk_se_fetch] EOF"; break }
    if ($rcRead -eq -1) { $nSkip++; continue }
    if ($rcRead -lt 0)  { Write-Host "[ERROR] JVRead rc=$rcRead"; break }

    $nTotal++
    $rec = $buff.Value

    if ($rec.Length -lt 395) { continue }
    if ($rec.Substring(0, 2) -ne "SE") { continue }

    # Parse fields (1-indexed string positions = 0-indexed bytes)
    $year     = $rec.Substring(11, 4)   # bytes 11-14 YYYY
    $mday     = $rec.Substring(15, 4)   # bytes 15-18 MMDD
    $courseCode = $rec.Substring(19, 2)
    $kai      = $rec.Substring(21, 2)
    $nichi    = $rec.Substring(23, 2)
    $raceNum  = $rec.Substring(25, 2)
    $umaban   = $rec.Substring(28, 2)
    $horseId  = $rec.Substring(30, 10)

    # finish_time bytes 297-300 (0-indexed) = Substring(297, 4)
    $ftRaw = $rec.Substring(297, 4)
    $ft = ""
    try {
        $ftInt = [int]$ftRaw
        if ($ftInt -ge 600 -and $ftInt -le 2500) { $ft = ($ftInt / 10.0).ToString("F1") }
    } catch {}

    # agari3f bytes 390-392 = Substring(390, 3)
    $agRaw = $rec.Substring(390, 3)
    $ag = ""
    try {
        $agInt = [int]$agRaw
        if ($agInt -ge 290 -and $agInt -le 500) { $ag = ($agInt / 10.0).ToString("F1") }
    } catch {}

    $raceDate = $year + $mday
    $writer.WriteLine("$horseId`t$raceDate`t$courseCode`t$kai`t$nichi`t$raceNum`t$umaban`t$ft`t$ag")
    $nSE++

    # Progress every 30 seconds
    $now = [System.DateTime]::Now
    if (($now - $lastReport).TotalSeconds -ge 30) {
        Write-Host "  SE: $nSE / total: $nTotal / skip: $nSkip"
        $lastReport = $now
    }
}

$writer.Close()
$jv.JVClose()

Write-Host "[jv_bulk_se_fetch] Done: SE=$nSE total=$nTotal skip=$nSkip"
Write-Host "[jv_bulk_se_fetch] Output: $outFile"
