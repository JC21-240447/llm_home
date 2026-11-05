# Day 1 - one-shot environment setup for Phase 1 (Model A).
# ASCII only on purpose (Windows PowerShell 5.1 misreads UTF-8 without BOM).
#
# Run from the repo root in a SEPARATE Windows PowerShell window:
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Machine home -TorchIndexUrl <index url from the pytorch.org selector>
#
# -Machine        school | home            (label written into the run log)
#                 SCHOOL-PC SWAP / GAKKO-PC: use -Machine school there. No other change.
# -TorchIndexUrl  the PyTorch wheel index for your CUDA build. Take it from the
#                 selector on https://pytorch.org/get-started/locally/
#                 (Stable / Windows / Pip / Python / CUDA x.y). Do not guess.
# -PythonCmd      command that starts the Python you want (default: "py -3.12")

param(
    [Parameter(Mandatory = $true)][ValidateSet("school", "home")][string]$Machine,
    [Parameter(Mandatory = $true)][string]$TorchIndexUrl,
    [string]$PythonCmd = "py -3.12"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }

Step "1/7 GPU and driver (nvidia-smi)"
try { nvidia-smi } catch { Write-Host "nvidia-smi not found. Install/update the NVIDIA driver or ask the school admin." -ForegroundColor Yellow }

Step "2/7 Python version"
$pyParts = $PythonCmd.Split(" ")
$pyExe = $pyParts[0]
$pyArgs = @()
if ($pyParts.Length -gt 1) { $pyArgs = $pyParts[1..($pyParts.Length - 1)] }
& $pyExe @pyArgs --version

Step "3/7 Folder layout"
foreach ($d in @("data", "src", "configs", "scripts", "runs", "docs", "tests")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    if (-not (Test-Path "$d\.gitkeep")) { New-Item -ItemType File -Path "$d\.gitkeep" | Out-Null }
}

Step "4/7 Virtual environment (.venv)"
if (-not (Test-Path ".venv")) { & $pyExe @pyArgs -m venv .venv }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip

Step "5/7 PyTorch (CUDA build from $TorchIndexUrl)"
& $venvPy -m pip install torch --index-url $TorchIndexUrl

Step "6/7 Other packages"
& $venvPy -m pip install -r requirements.txt

Step "7/7 Freeze exact versions, then run the environment check"
& $venvPy -m pip freeze | Out-File -Encoding ascii "requirements_lock_$Machine.txt"
& $venvPy scripts\day1_env_check.py --machine $Machine

Write-Host ""
Write-Host "Done. Send me: run_log.md (new entry) and runs\env_check_${Machine}_*.json" -ForegroundColor Green
