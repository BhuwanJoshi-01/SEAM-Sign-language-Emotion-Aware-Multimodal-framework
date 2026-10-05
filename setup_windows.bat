@echo off
rem Set up SEAM in a virtual environment (venv). Windows.
rem
rem   setup_windows.bat          everything: website, analyse your own videos, tests
rem   setup_windows.bat --lite   only what the website needs (about 60 MB instead of 1 GB)
rem
rem Needs Python 3.11, 3.12 or 3.13 from python.org (3.12 is what it was built on). Nothing is installed outside the .venv folder.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set MODE=full
if /i "%~1"=="--lite" set MODE=lite

set "PY="
for %%C in ("py -3.12" "py -3.11" "py -3.13" "python") do (
  if not defined PY (
    %%~C -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=%%~C"
  )
)
if not defined PY (
  echo Python 3.11 to 3.13 was not found.
  echo Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  exit /b 1
)
for /f "delims=" %%V in ('%PY% --version') do echo Using %%V

%PY% -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip

if "%MODE%"=="lite" (
  echo Installing the website's packages ...
  python -m pip install -r requirements-lite.txt
) else (
  echo Installing everything ^(this downloads about 1 GB the first time^) ...
  python -m pip install -e ".[dev,avatar]"
)
if errorlevel 1 (
  echo.
  echo The install did not finish. Check the internet connection and run this file again.
  exit /b 1
)

set "PYTHONPATH=%~dp0src"
set "SEAM_MEDIAPIPE_MODELS=%~dp0models\mediapipe"
echo.
python scripts\check_install.py
endlocal
