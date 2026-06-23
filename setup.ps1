<#
.SYNOPSIS
    One-shot environment setup for humanoid_sim on a clean Windows machine.

.DESCRIPTION
    Creates a local virtual environment (.venv), upgrades pip, installs every
    Python dependency from requirements.txt (mujoco, numpy, scipy, pyyaml, Pillow,
    torch), and runs a quick import smoke-test. All model assets and policy weights
    are already in the repo, and the browser demos fetch three.js / mujoco-js from a
    CDN at runtime, so nothing else needs downloading.

.PARAMETER Python
    Path to a specific Python interpreter to build the venv from. By default the
    script auto-detects one (the `py` launcher, then `python`).

.PARAMETER Recreate
    Delete an existing .venv and build it fresh.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Recreate
    .\setup.ps1 -Python "C:\Python313\python.exe"
#>
[CmdletBinding()]
param(
    [string]$Python,
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'

function Info($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# --- 1. find a suitable base Python (>= 3.10) -----------------------------------
# Probe a candidate (command + optional args, e.g. @('py','-3')). Returns the resolved
# interpreter path + version, or $null if it's missing or older than 3.10. The -c snippet
# uses only single quotes inside so PowerShell forwards it to the exe without mangling.
function Resolve-Python([string[]]$parts) {
    $exe = $parts[0]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $null }
    $rest = if ($parts.Count -gt 1) { $parts[1..($parts.Count - 1)] } else { @() }
    try {
        $out = & $exe @rest -c "import sys; print(sys.executable); print('%d.%d' % sys.version_info[:2])" 2>$null
    } catch { return $null }
    if ($LASTEXITCODE -ne 0) { return $null }
    $lines = @($out)
    if ($lines.Count -lt 2) { return $null }
    $realExe = $lines[0].Trim(); $ver = $lines[1].Trim()
    if (-not (Test-Path $realExe)) { return $null }
    if ([version]$ver -lt [version]'3.10') { return $null }
    return [pscustomobject]@{ Exe = $realExe; Ver = $ver }
}

Info "Locating a Python interpreter (3.10+ required)"
$found = $null

if ($Python) {
    $found = Resolve-Python @($Python)
    if (-not $found) { Die "The interpreter '$Python' is missing or older than Python 3.10." }
} else {
    foreach ($cand in @(, @('py', '-3')), @('python'), @('python3')) {
        $found = Resolve-Python $cand
        if ($found) { break }
    }
}

if (-not $found) {
    Die "No Python 3.10+ found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run this script."
}
$basePy = $found.Exe; $baseVer = $found.Ver
Ok "Using Python $baseVer at $basePy"
if ([version]$baseVer -ge [version]'3.14') {
    Warn "Python $baseVer is very new; if torch has no wheel for it yet, install Python 3.12/3.13 and re-run with -Python <path>."
}

# --- 2. create the virtual environment ------------------------------------------
if ($Recreate -and (Test-Path $venv)) {
    Info "Removing existing .venv (-Recreate)"
    Remove-Item -Recurse -Force $venv
}

if (Test-Path $venvPy) {
    Info ".venv already exists - reusing it (use -Recreate to rebuild)"
} else {
    Info "Creating virtual environment in .venv"
    & $basePy -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { Die "Failed to create the virtual environment." }
    Ok "Created .venv"
}

# --- 3. install dependencies ----------------------------------------------------
Info "Upgrading pip"
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Die "pip upgrade failed." }

Info "Installing dependencies from requirements.txt (torch is large - this can take a few minutes)"
& $venvPy -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Die "Dependency installation failed. Scroll up for the failing package." }
Ok "All dependencies installed"

# --- 4. smoke test --------------------------------------------------------------
# Write the check to a temp file rather than passing it via `-c`: Windows PowerShell
# mangles embedded quotes when forwarding a multi-line string to a native exe.
Info "Verifying the install"
$check = @'
import importlib
for m in ["mujoco", "numpy", "scipy", "yaml", "PIL", "torch"]:
    mod = importlib.import_module(m)
    print("  %-8s %s" % (m, getattr(mod, "__version__", "?")))
print("OK")
'@
$tmp = Join-Path $env:TEMP ("hs_check_{0}.py" -f [guid]::NewGuid().ToString('N'))
Set-Content -Path $tmp -Value $check -Encoding utf8
try { & $venvPy $tmp } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
if ($LASTEXITCODE -ne 0) { Die "Smoke test failed - a dependency did not import cleanly." }

# --- 5. done --------------------------------------------------------------------
Write-Host ""
Info "Setup complete. Try one of:"
Write-Host "    .\.venv\Scripts\python scripts\boxing_demo.py        # 1v1 boxing game (browser)" -ForegroundColor White
Write-Host "    .\.venv\Scripts\python scripts\web_demo_wasm.py      # IMU sim, browser render" -ForegroundColor White
Write-Host "    .\.venv\Scripts\python scripts\demo_balance.py       # native viewer + IMU HUD" -ForegroundColor White
Write-Host "    .\.venv\Scripts\python -m pytest                     # run the test suite" -ForegroundColor White
Write-Host ""
