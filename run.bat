@echo off
cd /d "%~dp0"

if not exist "best_water_segmentation_model.pth" (
    echo.
    echo ERROR: Model checkpoint is missing.
    echo.
    echo Download best_water_segmentation_model.pth
    echo from the repository Releases section.
    echo Place it next to app.py in this folder.
    echo.
    pause
    exit /b 1
)


echo.
echo ==================================================
echo           WATERSEG AI - GALLERY EDITION
echo ==================================================
echo.

python --version

if errorlevel 1 (
    echo.
    echo ERROR: Python was not found.
    echo Please install Python 3.10 or newer.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating Python environment...
    python -m venv .venv

    if errorlevel 1 goto failed
)

if not exist ".venv\.waterseg_installed" (
    echo.
    echo Installing required packages...
    echo First launch may take several minutes.
    echo.

    ".venv\Scripts\python.exe" -m pip install -r requirements.txt

    if errorlevel 1 goto failed

    echo Setup complete> ".venv\.waterseg_installed"
)

echo.
echo ==================================================
echo               STARTING WATERSEG AI
echo ==================================================
echo.
echo Open this address in your browser:
echo.
echo     http://127.0.0.1:5000
echo.
echo Keep this window open while using the website.
echo.

".venv\Scripts\python.exe" app.py

echo.
echo WaterSeg AI server stopped.
pause
exit /b 0

:failed
echo.
echo Setup failed. Please review the error above.
pause
exit /b 1
