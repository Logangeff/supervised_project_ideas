@echo off
setlocal EnableDelayedExpansion

set "PHASE=%~1"
set "DASHBOARD_PORT=8501"
if "%PHASE%"=="" set "PHASE=all"

if /I "%PHASE%"=="help" goto :help
if /I "%PHASE%"=="dashboard" goto :dashboard

echo Running US Macro Regime Radar phase: %PHASE%
python -m src.main --phase %PHASE%
if errorlevel 1 (
  echo.
  echo Project phase failed: %PHASE%
  pause
  exit /b 1
)

if /I "%PHASE%"=="all" goto :open_results
if /I "%PHASE%"=="results" goto :open_results
if /I "%PHASE%"=="build_dashboard_payload" goto :open_results
goto :done

:open_results
if exist "outputs\\metrics" start "" "outputs\\metrics"
if exist "outputs\\dashboard" start "" "outputs\\dashboard"
goto :done

:dashboard
for /f %%P in ('powershell -NoProfile -Command "$l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0); $l.Start(); $p = $l.LocalEndpoint.Port; $l.Stop(); Write-Output $p"') do set "DASHBOARD_PORT=%%P"
echo Launching dashboard on port %DASHBOARD_PORT%...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:%DASHBOARD_PORT%'"
python -m streamlit run app/streamlit_app.py --server.port %DASHBOARD_PORT%
if errorlevel 1 (
  echo.
  echo Dashboard launch failed.
  pause
  exit /b 1
)
goto :done

:help
echo Usage:
echo   run_project.bat fetch_macro_data
echo   run_project.bat build_monthly_panel
echo   run_project.bat build_cycle_labels
echo   run_project.bat train_phase_forecasts
echo   run_project.bat train_recession_risk
echo   run_project.bat build_dashboard_payload
echo   run_project.bat dashboard
echo   run_project.bat smoke
echo   run_project.bat results
echo   run_project.bat all
goto :done

:done
endlocal
