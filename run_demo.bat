@echo off
rem Start the SEAM website on this computer and open it in the browser.
rem   run_demo.bat           http://127.0.0.1:8000/
rem   run_demo.bat 9000      another port
setlocal
cd /d "%~dp0"
set PORT=%~1
if "%PORT%"=="" set PORT=8000

if not exist .venv\Scripts\activate.bat (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
set "PYTHONPATH=%~dp0src"
set "SEAM_MEDIAPIPE_MODELS=%~dp0models\mediapipe"
rem Keep everything inside this folder instead of the paths of the machine it was built on.
if "%SEAM_DATA_ROOT%"=="" set "SEAM_DATA_ROOT=%~dp0data"

echo SEAM is starting at http://127.0.0.1:%PORT%/   (Ctrl+C here stops it)
rem Set SEAM_NO_BROWSER=1 to skip opening a browser.
if "%SEAM_NO_BROWSER%"=="" start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:%PORT%/"
python -m seam.cli serve --port %PORT%
endlocal
