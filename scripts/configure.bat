@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Enter your API keys (typed hidden, saved only in .env on this PC) and test them.
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m automation.kronos_auto configure
pause
