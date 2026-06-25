<#
.SYNOPSIS
    Launch the G1 boxing game, and/or register it to auto-start at Windows logon.

.DESCRIPTION
    With no switches this just launches the boxing server (what the autostart shortcut
    runs). With -Install it drops a shortcut into your Startup folder so the game starts
    every time you log in. With -Uninstall it removes that shortcut.

    Runs at *logon* (not at the boot screen) because the demo opens a browser. The repo
    location is captured at install time, so re-run -Install if you move the repo.

.PARAMETER Install
    Create the Startup-folder shortcut (turn autostart on).

.PARAMETER Uninstall
    Remove the Startup-folder shortcut (turn autostart off).

.PARAMETER Port
    Port for the boxing server (default 8002).

.PARAMETER NoBrowser
    Don't auto-open a browser when the server starts.

.EXAMPLE
    .\boxing_autostart.ps1 -Install
    .\boxing_autostart.ps1 -Install -Port 8080 -NoBrowser
    .\boxing_autostart.ps1 -Uninstall
    .\boxing_autostart.ps1            # run it now
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Uninstall,
    [int]$Port = 8002,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root   = $PSScriptRoot
$self   = $PSCommandPath
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$demo   = Join-Path $root 'scripts\boxing_demo.py'
$startup  = [Environment]::GetFolderPath('Startup')
$shortcut = Join-Path $startup 'G1 Boxing.lnk'

function Info($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    $m" -ForegroundColor Green }
function Die($m)  { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# --- install: create the Startup shortcut ---------------------------------------
if ($Install) {
    $psExe   = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $lnkArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$self`" -Port $Port"
    if ($NoBrowser) { $lnkArgs += ' -NoBrowser' }

    $ws  = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($shortcut)
    $lnk.TargetPath       = $psExe
    $lnk.Arguments        = $lnkArgs
    $lnk.WorkingDirectory = $root
    $lnk.WindowStyle      = 7            # minimized
    $lnk.Description       = 'Auto-start the G1 boxing game'
    $lnk.Save()

    Info "Autostart enabled."
    Ok   "Shortcut: $shortcut"
    Ok   "Launches: $venvPy $demo --port $Port$(if ($NoBrowser) { ' --no-browser' })"
    Ok   "It will start at your next logon. Re-run -Install if you move the repo."
    exit 0
}

# --- uninstall: remove the Startup shortcut -------------------------------------
if ($Uninstall) {
    if (Test-Path $shortcut) {
        Remove-Item $shortcut -Force
        Info "Autostart disabled (removed $shortcut)."
    } else {
        Info "Autostart was not enabled (no shortcut found)."
    }
    exit 0
}

# --- default: launch the game now (this is what the shortcut runs) --------------
if (-not (Test-Path $venvPy)) {
    Die "Virtual environment not found at $venvPy. Run .\setup.ps1 first."
}
if (-not (Test-Path $demo)) {
    Die "Could not find $demo - are you running this from the repo root?"
}

$pyArgs = @($demo, '--port', "$Port")
if ($NoBrowser) { $pyArgs += '--no-browser' }

Info "Starting G1 boxing on http://127.0.0.1:$Port/  (close this window to stop)"
& $venvPy @pyArgs
