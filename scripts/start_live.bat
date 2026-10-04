@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Start the Kronos real-time trading bot. Restarts automatically if it crashes.
REM Settings: automation\live.yaml   Keys: .env   Log: outputs\live\live.log
REM Stop: close this window or press Ctrl+C.  Block new buys: scripts\stop_buying.bat
cd /d "%~dp0\.."
title Kronos live trader
if not exist .venv\Scripts\python.exe (
  echo Run scripts\setup.bat first.
  pause
  exit /b 1
)
if not exist .env (
  echo No .env file found. Copy .env.example to .env and add your API keys.
  pause
  exit /b 1
)
:loop
.venv\Scripts\python.exe -m automation.kronos_auto live %*
if %errorlevel%==2 (
  echo Configuration problem - see the message above. Not restarting.
  pause
  exit /b 2
)
if errorlevel 1 (
  echo Bot exited with an error. Restarting in 60 seconds... ^(close the window to stop^)
  timeout /t 60 /nobreak >nul
  goto loop
)
