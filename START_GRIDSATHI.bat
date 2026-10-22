@echo off
title GridSathi - Energy Forecasting and Smart Demand Scheduling
cd /d "%~dp0"

echo =====================================================================
echo    GridSathi: Energy Forecasting and Smart Demand Scheduling
echo    Starting FastAPI Backend and Launching Web Dashboard...
echo =====================================================================
echo.

rem Check if python environment exists
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

rem Start server in a background window if not already active
netstat -ano | findstr ":8000" >nul
if %errorlevel% neq 0 (
    echo Starting AI Server on http://127.0.0.1:8000 ...
    start "GridSathi Server" /min "%PYTHON_EXE%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
    timeout /t 3 /nobreak >nul
) else (
    echo GridSathi Server is already active on port 8000!
)

rem Open browser to the dashboard
echo Opening GridSathi Dashboard in your browser...
start http://127.0.0.1:8000/

echo.
echo =====================================================================
echo    GridSathi is LIVE at http://127.0.0.1:8000/
echo    Keep this window open or minimize it.
echo =====================================================================
pause
