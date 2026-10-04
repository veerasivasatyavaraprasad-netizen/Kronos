@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM Kill switch: the running bot stops opening new positions (stop-loss / take-profit exits still work).
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m automation.kronos_auto live-status --stop
pause
