@echo off
REM One-shot Windows setup: venv + dependencies + model download + smoke test.
cd /d "%~dp0\.."
if not exist .venv (
  echo ^>^> Creating virtualenv .venv
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -c "import torch" 2>nul || pip install -q torch || goto :error
pip install -q -r requirements-app.txt || goto :error
python -m automation.kronos_auto setup-models --model kronos-small || goto :error
python -m automation.kronos_auto setup-models --model kronos-mini || goto :error
python -m automation.kronos_auto run --no-backtest || goto :error
echo.
echo Setup complete. Start the web UI with:  scripts\start.bat
echo Forecast all configured symbols with:  .venv\Scripts\python -m automation.kronos_auto run
exit /b 0
:error
echo Setup failed.
exit /b 1
