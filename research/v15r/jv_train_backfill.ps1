# v15r 学習用 SLOP/WOOD 全量バックフィル (one-off、daily checkpoint には触れない)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = "C:\Users\takum\keiba-ai\research\v15r\jv_hist"
$jv = New-Object -ComObject "JVDTLab.JVLink"
$rc = $jv.JVInit("UNKNOWN/0.0"); Write-Host "JVInit $rc"
foreach ($sp in @("SLOP","WOOD")) {
  $nData=[ref]([int]0); $nFiles=[ref]([int]0); $lastTime=[ref]("")
  $rcO = $jv.JVOpen($sp, "20200101000000", 1, $nData, $nFiles, $lastTime)
  Write-Host "$sp JVOpen rc=$rcO nFiles=$($nFiles.Value)"
  if ($rcO -lt 0) { $jv.JVClose()|Out-Null; continue }
  $w = [System.IO.StreamWriter]::new((Join-Path $out "$sp.dat"), $false, [System.Text.Encoding]::UTF8)
  $buff=[ref](""); $size=[ref]([int]131072); $fname=[ref]("")
  $n=0; $skip=0
  while ($true) {
    $r = $jv.JVRead($buff, $size, $fname)
    if ($r -eq 0) { break }
    if ($r -eq -1) { continue }
    if ($r -eq -3) { $skip++; if ($skip -gt 200000) { break }; Start-Sleep -Milliseconds 20; continue }
    if ($r -lt 0) { Write-Host "  rc=$r"; break }
    $rec = $buff.Value.TrimEnd("`r","`n")
    if ($rec.Length -gt 2) { $w.WriteLine($rec); $n++ }
    if ($n % 200000 -eq 0 -and $n -gt 0) { Write-Host "  $sp $n..." }
  }
  $w.Close(); $jv.JVClose()|Out-Null
  Write-Host "$sp done records=$n"
}
Write-Host "ALL DONE"
