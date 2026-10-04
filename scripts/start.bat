@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m automation.kronos_auto serve %*
