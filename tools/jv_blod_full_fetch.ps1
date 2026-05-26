# JV-Link BLOD full fetch: HN (繁殖馬) + SK (産駒) records
# Builds 3代血統 sire/dam/bms distance×surface features
# Run with 32-bit PowerShell:
#   C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -File tools\jv_blod_full_fetch.ps1

param(
    [string]$fromDate = "19900101",   # BLOD = 全期間
    [string]$outDir   = "C:\Users\takum\keiba-ai\data\v20",
    [int]   $maxRecs  = 5000000
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$fromTime = $fromDate + "000000"

Write-Host "[jv_blod_full_fetch] from=$fromDate outDir=$outDir"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$jv = New-Object -ComObject "JVDTLab.JVLink"
$rcInit = $jv.JVInit("UNKNOWN/0.0")
Write-Host "[jv_blod_full_fetch] JVInit rc=$rcInit"
if ($rcInit -ne 0) { Write-Host "[ERROR] JVInit failed"; exit 1 }

$nData = [ref]([int]0); $nFiles = [ref]([int]0); $lastTime = [ref]("")
$rcOpen = $jv.JVOpen("BLOD", $fromTime, 3, $nData, $nFiles, $lastTime)
Write-Host "[jv_blod_full_fetch] JVOpen rc=$rcOpen nData=$($nData.Value) nFiles=$($nFiles.Value)"
if ($rcOpen -lt 0) { Write-Host "[ERROR] JVOpen failed rc=$rcOpen"; $jv.JVClose(); exit 1 }

$buff = [ref](""); $size = [ref]([int]32768); $fname = [ref]("")
$nTotal = 0; $nHN = 0; $nSK = 0; $nBT = 0; $nSkip = 0

# HN: 繁殖馬情報 (blood_num, horse_name, sex, birthday, sire_blood_num, dam_blood_num)
$hnFile = Join-Path $outDir "blod_hn.tsv"
$wrHN = [System.IO.StreamWriter]::new($hnFile, $false, [System.Text.Encoding]::UTF8)
$wrHN.WriteLine("blood_num`tsex`tbirthday`tsire_blood_num`tdam_blood_num`thorse_name")

# SK: 産駒登録 (sire_blood_num → offspring horse_id)
$skFile = Join-Path $outDir "blod_sk.tsv"
$wrSK = [System.IO.StreamWriter]::new($skFile, $false, [System.Text.Encoding]::UTF8)
$wrSK.WriteLine("sire_blood_num`toffspring_blood_num`toffspring_horse_id")

$lastReport = [System.DateTime]::Now

while ($nTotal -lt $maxRecs) {
    $rcRead = $jv.JVRead($buff, $size, $fname)
    if ($rcRead -eq 0)  { Write-Host "[jv_blod_full_fetch] EOF"; break }
    if ($rcRead -eq -1) { continue }
    if ($rcRead -lt 0)  {
        if ($rcRead -eq -3) { $size = [ref]([int]65536); $nSkip++; continue }
        Write-Host "[ERROR] JVRead rc=$rcRead"; break
    }
    $nTotal++
    $rec = $buff.Value
    if ($rec.Length -lt 5) { continue }

    $rtype = $rec.Substring(0, 2)

    if ($rtype -eq "HN") {
        # HN record: 繁殖馬
        # [0-1]=HN [2]=DataKubun [3-10]=MakeDate [11-18]=BloodNum(8) [19-54]=HorseName(36sjis)
        # [55]=sex [56-61]=birthday(YYYYMM) [62-69]=SireBloodNum [70-77]=DamBloodNum
        if ($rec.Length -ge 78) {
            $bn      = $rec.Substring(11, 8).Trim()
            $sex     = $rec.Substring(55, 1).Trim()
            $bday    = $rec.Substring(56, 6).Trim()
            $sireBN  = $rec.Substring(62, 8).Trim()
            $damBN   = $rec.Substring(70, 8).Trim()
            # horse_name is SJIS at 19-54 (36 bytes), skip for now
            $wrHN.WriteLine("$bn`t$sex`t$bday`t$sireBN`t$damBN`t")
            $nHN++
        }
    } elseif ($rtype -eq "SK") {
        # SK record: 産駒登録 (sire → offspring link)
        # [0-1]=SK [2]=DataKubun [3-10]=MakeDate [11-18]=SireBloodNum [19-26]=OffspringBloodNum [27-36]=HorseId(10)
        if ($rec.Length -ge 37) {
            $sireBN   = $rec.Substring(11, 8).Trim()
            $offBN    = $rec.Substring(19, 8).Trim()
            $horseId  = if ($rec.Length -ge 37) { $rec.Substring(27, 10).Trim() } else { "" }
            $wrSK.WriteLine("$sireBN`t$offBN`t$horseId")
            $nSK++
        }
    } elseif ($rtype -eq "BT") {
        $nBT++
    }

    $now = [System.DateTime]::Now
    if (($now - $lastReport).TotalSeconds -ge 30) {
        Write-Host "  HN=$nHN SK=$nSK BT=$nBT total=$nTotal skip=$nSkip"
        $lastReport = $now
    }
}

$wrHN.Close(); $wrSK.Close()
$jv.JVClose()

Write-Host "[jv_blod_full_fetch] Done: HN=$nHN SK=$nSK BT=$nBT total=$nTotal skip=$nSkip"
Write-Host "[jv_blod_full_fetch] HN -> $hnFile"
Write-Host "[jv_blod_full_fetch] SK -> $skFile"
Write-Host ""
Write-Host "=== Next step ==="
Write-Host "Run: python train/build_blod_full_features.py"
