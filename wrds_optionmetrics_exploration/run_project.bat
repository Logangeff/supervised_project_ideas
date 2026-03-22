@echo off
setlocal

set "PHASE=%~1"
if "%PHASE%"=="" set "PHASE=all"

if /I "%PHASE%"=="help" goto :help

echo Running WRDS + OptionMetrics phase: %PHASE%
python -m src.main --phase %PHASE%
if errorlevel 1 (
  echo.
  echo Project phase failed: %PHASE%
  pause
  exit /b 1
)

if /I "%PHASE%"=="all" goto :open_results
if /I "%PHASE%"=="results" goto :open_results
if /I "%PHASE%"=="run_phase2_decision" goto :open_results
if /I "%PHASE%"=="run_phase2_bucket_analysis" goto :open_results
if /I "%PHASE%"=="train_text_news_extension" goto :open_results
goto :done

:open_results
if exist "outputs\\metrics" start "" "outputs\\metrics"
if exist "outputs\\figures" start "" "outputs\\figures"
goto :done

:help
echo Usage:
echo   run_project.bat extract_stock_data
echo   run_project.bat build_stock_panel
echo   run_project.bat train_part1
echo   run_project.bat test_wrds_connection
echo   run_project.bat extract_option_data
echo   run_project.bat build_option_features
echo   run_project.bat train_part2
echo   run_project.bat build_surface_factors
echo   run_project.bat train_surface_extension
echo   run_project.bat extract_calibrated_surface_inputs
echo   run_project.bat build_calibrated_surface
echo   run_project.bat train_calibrated_surface_extension
echo   run_project.bat run_phase2_decision
echo   run_project.bat run_phase2_bucket_analysis
echo   run_project.bat build_text_news_panel
echo   run_project.bat train_text_news_extension
echo   run_project.bat smoke
echo   run_project.bat results
echo   run_project.bat all
goto :done

:done
endlocal
