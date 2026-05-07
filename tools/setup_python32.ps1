# Setup 32-bit Python for JV-Link COM (Session #41 A)
#
# JV-Link DLL は 32-bit COM のみ提供 (C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll)。
# 既存 keiba-ai 用 64-bit Python (3.14) は維持し、 別環境として 32-bit Python を install する。
#
# 推奨 Python 版: 3.11 32-bit (Windows installer)
# 推奨 install path: C:\Python311-32bit\
# 推奨 venv path:   C:\Users\takum\jvlink-venv\
#
# 実行:
#   PowerShell を 管理者権限で起動
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\tools\setup_python32.ps1
#
# 本 script は admin 権限必須 (install 操作のため)。
# install 後の動作確認は tools/jvlink_test_python32.py を 32-bit Python で実行。

param(
    [string]$Python32InstallPath = "C:\Python311-32bit",
    [string]$VenvPath = "C:\Users\takum\jvlink-venv",
    [string]$Python32Url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe",
    [switch]$SkipInstall,
    [switch]$DryRun
)

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
}

function Test-Admin {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Step 0: 環境確認
Write-Step "Step 0: 環境確認"
if (-not (Test-Admin)) {
    Write-Host "[ERROR] Admin 権限で実行してください。" -ForegroundColor Red
    Write-Host "  PowerShell を 'Run as Administrator' で起動 → 再実行" -ForegroundColor Yellow
    exit 1
}
Write-Host "  OK: Admin 権限"

if (Test-Path "$Python32InstallPath\python.exe") {
    $existing = & "$Python32InstallPath\python.exe" -c "import platform; print(platform.architecture()[0])" 2>&1
    Write-Host "  既存 Python: $existing"
    if ($existing -like "*32bit*") {
        Write-Host "  [INFO] 32-bit Python 既に install 済 → install skip"
        $SkipInstall = $true
    }
}

# Step 1: 32-bit Python install
if (-not $SkipInstall) {
    Write-Step "Step 1: 32-bit Python install"
    $installer = "$env:TEMP\python-3.11.9-32bit.exe"

    if (-not (Test-Path $installer)) {
        Write-Host "  download: $Python32Url"
        if ($DryRun) { Write-Host "  [DRY] download skip" }
        else { Invoke-WebRequest -Uri $Python32Url -OutFile $installer -UseBasicParsing }
    }

    Write-Host "  install to: $Python32InstallPath"
    $args = @(
        "/quiet",
        "InstallAllUsers=1",
        "TargetDir=$Python32InstallPath",
        "Include_test=0",
        "Include_pip=1",
        "Include_launcher=0",
        "PrependPath=0"  # PATH 先頭に追加しない (既存 64-bit と衝突回避)
    )
    if ($DryRun) {
        Write-Host "  [DRY] $installer $args"
    } else {
        Start-Process -FilePath $installer -ArgumentList $args -Wait -NoNewWindow
    }

    if (-not (Test-Path "$Python32InstallPath\python.exe")) {
        Write-Host "[ERROR] install 失敗: $Python32InstallPath\python.exe 不在" -ForegroundColor Red
        exit 2
    }
    Write-Host "  OK: install 完了"
}

# Step 2: arch 確認
Write-Step "Step 2: 32-bit 確認"
$arch = & "$Python32InstallPath\python.exe" -c "import platform; print(platform.architecture()[0])" 2>&1
Write-Host "  arch: $arch"
if ($arch -notlike "*32bit*") {
    Write-Host "[ERROR] 32-bit ではない" -ForegroundColor Red
    exit 3
}

# Step 3: venv 作成
Write-Step "Step 3: venv 作成"
if (-not (Test-Path $VenvPath)) {
    if ($DryRun) {
        Write-Host "  [DRY] python -m venv $VenvPath"
    } else {
        & "$Python32InstallPath\python.exe" -m venv $VenvPath
    }
    Write-Host "  OK: venv 作成"
} else {
    Write-Host "  既に venv 存在: $VenvPath"
}

$venvPython = "$VenvPath\Scripts\python.exe"
$venvPip = "$VenvPath\Scripts\pip.exe"

# Step 4: 必須 package install (pywin32, pandas)
Write-Step "Step 4: package install"
$packages = @("pywin32", "pandas", "numpy")
foreach ($pkg in $packages) {
    Write-Host "  install: $pkg"
    if ($DryRun) {
        Write-Host "  [DRY] pip install $pkg"
    } else {
        & $venvPip install --quiet $pkg
    }
}

# Step 5: pywin32 post-install (COM 登録)
Write-Step "Step 5: pywin32 post-install"
$postInstall = "$VenvPath\Scripts\pywin32_postinstall.py"
if (Test-Path $postInstall) {
    if ($DryRun) {
        Write-Host "  [DRY] $venvPython $postInstall -install"
    } else {
        & $venvPython $postInstall -install
    }
    Write-Host "  OK: COM 登録完了"
} else {
    Write-Host "  [WARN] $postInstall 不在 (pywin32 未 install ?)"
}

# Step 6: 動作確認
Write-Step "Step 6: 動作確認"
$testScript = "$PSScriptRoot\jvlink_test_python32.py"
if (Test-Path $testScript) {
    Write-Host "  test 実行: $venvPython $testScript"
    if ($DryRun) {
        Write-Host "  [DRY] test skip"
    } else {
        & $venvPython $testScript --check-only
    }
} else {
    Write-Host "  [WARN] $testScript 不在 (本 Session A3 で作成予定)"
}

Write-Step "完了"
Write-Host ""
Write-Host "次の step:"
Write-Host "  1. activate venv: $VenvPath\Scripts\Activate.ps1"
Write-Host "  2. JV-Link 動作確認: python tools\jvlink_test_python32.py"
Write-Host "  3. 過去日付 fetch: python tools\jvlink_fetcher.py --date 20260503"
Write-Host ""
Write-Host "注意:"
Write-Host "  - 64-bit keiba-ai 環境とは別 (predict_core 等は 64-bit のまま)"
Write-Host "  - JV-Link 操作のみ 32-bit venv を使用"
