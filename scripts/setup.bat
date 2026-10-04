@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM One-shot Windows setup: venv + dependencies + model download + smoke test.
cd /d "%~dp0\.."
if not defined PY call "%~dp0find_python.bat"
if not defined PY (
  echo Python 3.10+ not found. Run RUN_ME.bat, or install Python 3.11 from python.org.
  goto :error
)
if not exist .venv\Scripts\python.exe (
  echo ^>^> Creating virtualenv .venv with %PY%
  if exist .venv rmdir /s /q .venv
  "%PY%" -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -c "import torch" 2>nul || pip install -q torch || goto :error
pip install -q -r requirements-app.txt || goto :error
python -m automation.kronos_auto setup-models --model kronos-small || goto :error
python -m automation.kronos_auto setup-models --model kronos-mini || goto :error
python -m automation.kronos_auto run --no-backtest || goto :error
echo.
echo Setup complete. Next steps:
echo   1. scripts\configure.bat   - enter your API keys and test them
echo   2. scripts\start_live.bat  - start the real-time trading bot (Binance starts in TEST mode)
echo   3. scripts\start.bat       - web UI at http://localhost:7070 (live bot dashboard: /live)
exit /b 0
:error
echo Setup failed.
exit /b 1
