@echo off
setlocal

set "PHASE=%~1"
if "%PHASE%"=="" set "PHASE=all"

if /I "%PHASE%"=="help" goto :help

echo Running TSLA company news exploration phase: %PHASE%
python -m src.main --phase %PHASE%
if errorlevel 1 (
  echo.
  echo Project phase failed: %PHASE%
  pause
  exit /b 1
)

if /I "%PHASE%"=="all" goto :open_results
if /I "%PHASE%"=="results" goto :open_results
goto :done

:open_results
if exist "outputs\\metrics" start "" "outputs\\metrics"
if exist "outputs\\figures" start "" "outputs\\figures"
goto :done

:help
echo Usage:
echo   run_project.bat collect_data
echo   run_project.bat build_dataset
echo   run_project.bat stage1_materiality
echo   run_project.bat stage2_direction
echo   run_project.bat stage3_sentiment
echo   run_project.bat stage3_amplitude
echo   run_project.bat results
echo   run_project.bat all
goto :done

:done
endlocal
