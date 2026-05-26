# JV-Link RA record probe: fetch RACE datatype, save raw RA text for byte-offset analysis
# Run with 32-bit PowerShell:
#   C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -File tools\jv_ra_lap_probe.ps1
#
# Output: data/v20/ra_probe_YYYYMMDD.txt  (raw RA records for offset analysis)
#         data/v20/ra_probe_sample.json   (parsed header + hex excerpt)

param(
    [string]$fromDate = "20150101",
    [string]$outDir   = "C:\Users\takum\keiba-ai\data\v20",
    [int]   $maxRA    = 500,    # stop after N RA records (for probing)
    [int]   $maxRecs  = 200000  # hard limit on total JVRead calls
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$fromTime = $fromDate + "000000"

Write-Host "[jv_ra_lap_probe] from=$fromDate maxRA=$maxRA outDir=$outDir"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$jv = New-Object -ComObject "JVDTLab.JVLink"
$rcInit = $jv.JVInit("UNKNOWN/0.0")
Write-Host "[jv_ra_lap_probe] JVInit rc=$rcInit"
if ($rcInit -ne 0) { Write-Host "[ERROR] JVInit failed"; exit 1 }

$nData  = [ref]([int]0); $nFiles = [ref]([int]0); $lastTime = [ref]("")
$rcOpen = $jv.JVOpen("RACE", $fromTime, 4, $nData, $nFiles, $lastTime)
Write-Host "[jv_ra_lap_probe] JVOpen rc=$rcOpen nData=$($nData.Value) nFiles=$($nFiles.Value)"
if ($rcOpen -lt 0) { Write-Host "[ERROR] JVOpen failed rc=$rcOpen"; $jv.JVClose(); exit 1 }

$buff = [ref](""); $size = [ref]([int]32768); $fname = [ref]("")

$nTotal = 0; $nRA = 0; $nSE = 0; $nSkip = 0

# Output file: one raw RA record per line
$outTxt  = Join-Path $outDir "ra_probe_${fromDate}.txt"
$writer  = [System.IO.StreamWriter]::new($outTxt, $false, [System.Text.Encoding]::UTF8)
$writer.WriteLine("# JV-Link RA raw records from RACE datatype, from=$fromDate")
$writer.WriteLine("# Each non-comment line = one RA record (raw text, trimmed)")

while ($nTotal -lt $maxRecs -and $nRA -lt $maxRA) {
    $rcRead = $jv.JVRead($buff, $size, $fname)
    if ($rcRead -eq 0)  { Write-Host "[jv_ra_lap_probe] EOF after $nTotal records"; break }
    if ($rcRead -eq -1) { continue }
    if ($rcRead -lt 0)  {
        if ($rcRead -eq -3) { $size = [ref]([int]65536); $nSkip++; continue }
        Write-Host "[ERROR] JVRead rc=$rcRead"; break
    }
    $nTotal++
    $rec = $buff.Value
    if ($rec.Length -lt 5) { continue }

    $rtype = $rec.Substring(0, 2)
    if ($rtype -eq "RA") {
        $writer.WriteLine($rec)
        $nRA++
        if ($nRA -le 3) {
            Write-Host "  [RA sample] len=$($rec.Length) head=$($rec.Substring(0, [Math]::Min(80, $rec.Length)))"
        }
    } elseif ($rtype -eq "SE") {
        $nSE++
    }

    if ($nTotal % 5000 -eq 0) {
        Write-Host "  total=$nTotal RA=$nRA SE=$nSE skip=$nSkip"
    }
}

$writer.Close()
$jv.JVClose()

Write-Host "[jv_ra_lap_probe] Done: RA=$nRA SE=$nSE total=$nTotal skip=$nSkip"
Write-Host "[jv_ra_lap_probe] Output: $outTxt"
Write-Host ""
Write-Host "=== Next step ==="
Write-Host "Run: python train/analyze_ra_offsets.py  (analyzes byte offsets in RA records)"
