@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m automation.kronos_auto live-status
pause
