@echo off
setlocal

cd /d "%~dp0"

set "PHASE=%~1"
if "%PHASE%"=="" set "PHASE=all"

echo Running project phase: %PHASE%
python -m src.main --phase %PHASE%

if errorlevel 1 (
    echo.
    echo Project run failed.
    pause
    exit /b 1
)

if /i "%PHASE%"=="all" goto open_results
if /i "%PHASE%"=="evaluate" goto open_results
goto done

:open_results
echo.
echo Opening main result files and folders...
if exist "outputs\metrics\evaluation_summary.csv" start "" "outputs\metrics\evaluation_summary.csv"
if exist "outputs\metrics\evaluation_summary.json" start "" "outputs\metrics\evaluation_summary.json"
if exist "outputs\figures" start "" "outputs\figures"
if exist "outputs\metrics" start "" "outputs\metrics"

:done
echo.
echo Finished.
pause
