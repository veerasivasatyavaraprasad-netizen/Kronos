@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m automation.kronos_auto live-status
pause
