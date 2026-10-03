@echo off
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m automation.kronos_auto live-status
pause
