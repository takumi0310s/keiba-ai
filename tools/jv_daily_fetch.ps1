# JV-Link daily fetch (32-bit PowerShell / SysWOW64 必須)
# 2026-08-12 第1弾-2: SE/HR/UM/O1-O6 + SLOP/WOOD の常設デイリー取得。
# 方式: dataspec ごとに JVOpen(diff, from=checkpoint) → 全レコードを
#       data\jvlink\daily\<SPEC>_<ts>.dat に生保存 (UTF-8 text, 1行1レコード)。
#       checkpoint.json に lastTime を保存し冪等 (次回はそこから差分)。
# Run: C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -NoProfile -File tools\jv_daily_fetch.ps1
param(
    [string]$specs = "RACE,SLOP,WOOD,DIFF",
    [string]$seedFrom = "20260806"   # checkpoint 無し時の初期起点
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$base   = "C:\Users\takum\keiba-ai"
$outDir = Join-Path $base "data\jvlink\daily"
$ckPath = Join-Path $outDir "checkpoint.json"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$ck = @{}
if (Test-Path $ckPath) {
    try {
        $j = Get-Content $ckPath -Raw | ConvertFrom-Json
        $j.PSObject.Properties | ForEach-Object { $ck[$_.Name] = $_.Value }
    } catch {}
}

$jv = New-Object -ComObject "JVDTLab.JVLink"
$rcInit = $jv.JVInit("UNKNOWN/0.0")
Write-Host "[jv_daily] JVInit rc=$rcInit"
if ($rcInit -ne 0) { exit 1 }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$fail = 0

foreach ($sp in $specs.Split(",")) {
    $from = $ck[$sp]
    if (-not $from) { $from = $seedFrom + "000000" }
    $nData=[ref]([int]0); $nFiles=[ref]([int]0); $lastTime=[ref]("")
    $rcOpen = $jv.JVOpen($sp, $from, 1, $nData, $nFiles, $lastTime)
    Write-Host "[jv_daily] $sp from=$from JVOpen rc=$rcOpen nFiles=$($nFiles.Value) lastTime=$($lastTime.Value)"
    if ($rcOpen -eq -1) { $jv.JVClose() | Out-Null; continue }   # 差分なし = 正常
    if ($rcOpen -lt 0)  { $fail++; $jv.JVClose() | Out-Null; continue }

    $outFile = Join-Path $outDir ("{0}_{1}.dat" -f $sp, $ts)
    $writer = [System.IO.StreamWriter]::new($outFile, $false, [System.Text.Encoding]::UTF8)
    $buff=[ref](""); $size=[ref]([int]65536); $fname=[ref]("")
    $n = 0; $nSkip = 0
    while ($true) {
        $rcRead = $jv.JVRead($buff, $size, $fname)
        if ($rcRead -eq 0)  { break }                 # 全ファイル EOF
        if ($rcRead -eq -1) { continue }              # ファイル切替
        if ($rcRead -eq -3) {                         # DL待ち/バッファ
            $size = [ref]([int]131072); $nSkip++
            if ($nSkip -gt 20000) { Write-Host "  [WARN] $sp rc=-3 過多で打切"; break }
            Start-Sleep -Milliseconds 50; continue
        }
        if ($rcRead -lt 0)  { Write-Host "  [WARN] $sp JVRead rc=$rcRead"; break }
        $rec = $buff.Value.TrimEnd("`r", "`n")
        if ($rec.Length -gt 2) { $writer.WriteLine($rec); $n++ }
    }
    $writer.Close()
    $jv.JVClose() | Out-Null
    if ($n -eq 0) { Remove-Item $outFile -ErrorAction SilentlyContinue }
    Write-Host "[jv_daily] $sp records=$n"
    if ($lastTime.Value) { $ck[$sp] = $lastTime.Value }
}

# checkpoint 保存
($ck.GetEnumerator() | ForEach-Object { '"{0}":"{1}"' -f $_.Key, $_.Value }) -join "," |
    ForEach-Object { "{$_}" } | Set-Content -Path $ckPath -Encoding UTF8
Write-Host "[jv_daily] checkpoint saved. fail=$fail"
exit $fail
