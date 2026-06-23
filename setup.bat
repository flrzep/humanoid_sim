@echo off
REM Double-clickable launcher for setup.ps1 (bypasses PowerShell execution policy).
REM Forwards any arguments, e.g.  setup.bat -Recreate
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
