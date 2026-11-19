@echo off
chcp 65001 > nul
set PYTHONUTF8=1
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0auto_push.py" %*
) else (
    python "%~dp0auto_push.py" %*
)
echo.
pause
