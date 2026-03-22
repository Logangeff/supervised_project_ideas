@echo off
setlocal

set "PHASE=%~1"
if "%PHASE%"=="" set "PHASE=all"

if /I "%PHASE%"=="help" goto :help

echo Running numerical claim detection project phase: %PHASE%
python -m src.main --phase %PHASE%
if errorlevel 1 (
  echo.
  echo Project phase failed: %PHASE%
  pause
  exit /b 1
)

if /I "%PHASE%"=="all" goto :open_results
if /I "%PHASE%"=="stage1_evaluate" goto :open_results
if /I "%PHASE%"=="stage2_evaluate" goto :open_results
if /I "%PHASE%"=="results" goto :open_results
goto :done

:open_results
echo.
echo Opening metrics and figures...
if exist "outputs\metrics\project_summary.json" start "" "outputs\metrics\project_summary.json"
if exist "outputs\metrics\stage1_evaluation_summary.csv" start "" "outputs\metrics\stage1_evaluation_summary.csv"
if exist "outputs\metrics\stage2_evaluation_summary.csv" start "" "outputs\metrics\stage2_evaluation_summary.csv"
if exist "outputs\figures" start "" "outputs\figures"
if exist "outputs\metrics" start "" "outputs\metrics"
goto :done

:help
echo Usage:
echo   run_project.bat stage1_data
echo   run_project.bat stage1_classical
echo   run_project.bat stage1_neural
echo   run_project.bat stage1_evaluate
echo   run_project.bat stage2_data
echo   run_project.bat stage2_models
echo   run_project.bat stage2_evaluate
echo   run_project.bat stage3_amplitude
echo   run_project.bat results
echo   run_project.bat smoke
echo   run_project.bat all
goto :done

:done
endlocal
