@echo off
REM Sets PY to a Python 3.10+ executable. Does not rely on PATH: Windows often installs Python
REM without adding it to PATH, and "python" may be the Microsoft Store placeholder.
set "PY="
for %%D in ("%LOCALAPPDATA%\Programs\Python\Python311" "%LOCALAPPDATA%\Programs\Python\Python312" "%LOCALAPPDATA%\Programs\Python\Python310" "%LOCALAPPDATA%\Programs\Python\Python313" "%ProgramFiles%\Python311" "%ProgramFiles%\Python312" "%ProgramFiles%\Python310" "%ProgramFiles%\Python313") do (
  if not defined PY if exist "%%~D\python.exe" set "PY=%%~D\python.exe"
)
if defined PY goto :verify

REM Python launcher (py.exe) or a python.exe that is on PATH
set "KRONOS_PYTMP=%TEMP%\kronos_python_path.txt"
if exist "%KRONOS_PYTMP%" del "%KRONOS_PYTMP%"
py -3 -c "import sys; print(sys.executable)" > "%KRONOS_PYTMP%" 2>nul
if errorlevel 1 python -c "import sys; print(sys.executable)" > "%KRONOS_PYTMP%" 2>nul
if exist "%KRONOS_PYTMP%" set /p PY=<"%KRONOS_PYTMP%"
if exist "%KRONOS_PYTMP%" del "%KRONOS_PYTMP%"
if not defined PY exit /b 0

:verify
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 set "PY="
exit /b 0
