@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Web UI: forecasts at http://localhost:7070 , live bot dashboard at http://localhost:7070/live
cd /d "%~dp0\.."
start "" http://localhost:7070/live
.venv\Scripts\python.exe -m automation.kronos_auto serve --host 127.0.0.1 --port 7070 %*
pause
