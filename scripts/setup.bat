@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM One-shot Windows setup: venv + dependencies + model download + smoke test.
cd /d "%~dp0\.."
if not defined PY call "%~dp0find_python.bat"
if not defined PY (
  echo Python 3.10+ not found. Run RUN_ME.bat, or install Python 3.11 from python.org.
  goto :error
)
if exist .venv\Scripts\python.exe goto :venv_ready
echo ^>^> Creating virtualenv .venv with %PY%
if exist .venv rmdir /s /q .venv
"%PY%" -m venv .venv
if not exist .venv\Scripts\python.exe goto :error
:venv_ready
set "VPY=.venv\Scripts\python.exe"

echo ^>^> Updating pip
"%VPY%" -m pip install --disable-pip-version-check -q --upgrade pip

"%VPY%" -c "import torch" >nul 2>nul
if not errorlevel 1 goto :torch_ready
echo ^>^> Installing PyTorch (about 200 MB)
"%VPY%" -m pip install --disable-pip-version-check torch
if errorlevel 1 goto :error
:torch_ready

echo ^>^> Installing app packages
"%VPY%" -m pip install --disable-pip-version-check -r requirements-app.txt
if errorlevel 1 goto :error

echo ^>^> Downloading Kronos models
"%VPY%" -m automation.kronos_auto setup-models --model kronos-small
if errorlevel 1 goto :error
"%VPY%" -m automation.kronos_auto setup-models --model kronos-mini
if errorlevel 1 goto :error

echo ^>^> Test forecast on the bundled sample data
"%VPY%" -m automation.kronos_auto run --no-backtest
if errorlevel 1 goto :error

echo.
echo Setup complete. Next steps:
echo   1. scripts\configure.bat   - enter your API keys and test them
echo   2. scripts\start_live.bat  - start the real-time trading bot (Binance starts in TEST mode)
echo   3. scripts\start.bat       - web UI at http://localhost:7070 (live bot dashboard: /live)
exit /b 0
:error
echo Setup failed.
exit /b 1
