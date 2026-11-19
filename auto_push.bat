@echo off
chcp 65001 > nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0auto_push.ps1" %*
echo.
pause
