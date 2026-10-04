@echo off
REM ============================================================
REM  Kronos - one click: install (first time), keys, check,
REM  dashboard in browser, and the real-time trading bot.
REM ============================================================
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title Kronos - one click start

REM ---- 1. Python 3.10+ (installs it with winget if missing) ----
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto :install_python
goto :have_python

:install_python
echo Python 3.10 or newer was not found. Installing Python 3.11 ...
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
echo.
echo Python installed. CLOSE this window and double-click RUN_ME.bat again.
echo If winget is not available, install Python 3.11 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation.
pause
exit /b 1

:have_python
REM ---- 2. First-time setup: packages + Kronos model (about 10 minutes) ----
if exist .venv\Scripts\python.exe goto :have_venv
echo First run: installing everything. This takes about 10 minutes ...
call scripts\setup.bat
if errorlevel 1 goto :failed
:have_venv

REM ---- 3. API keys (asked once, saved only in .env on this PC) ----
if exist .env goto :have_env
echo.
echo ===== Enter your API keys (typing is hidden; paste with right-click) =====
.venv\Scripts\python.exe -m automation.kronos_auto configure
:have_env

REM ---- 4. Connection check ----
echo.
echo ===== Checking connections =====
.venv\Scripts\python.exe -m automation.kronos_auto check
if errorlevel 1 (
  echo.
  echo Some checks FAILED. Fix them with scripts\configure.bat, then run RUN_ME.bat again.
  pause
  exit /b 1
)

REM ---- 5. Dashboard (separate minimized window) + open browser ----
start "Kronos dashboard" /min cmd /c ".venv\Scripts\python.exe -m automation.kronos_auto serve --host 127.0.0.1 --port 7070"
timeout /t 10 /nobreak >nul
start "" http://localhost:7070/live

REM ---- 6. Real-time trading bot (this window; close it to stop) ----
echo.
echo ===== Starting the trading bot. Close this window to stop it. =====
call scripts\start_live.bat
exit /b 0

:failed
echo Setup failed - scroll up for the error message.
pause
exit /b 1
