@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo.
echo ============================================================
echo Launch App
echo ============================================================
echo 1. WRDS bucket dashboard
echo 2. US macro regime dashboard
echo.
set "APP_CHOICE=1"
set /p "APP_CHOICE=Enter app number to launch [1]: "
if not defined APP_CHOICE set "APP_CHOICE=1"

if "%APP_CHOICE%"=="1" (
    if exist "wrds_optionmetrics_exploration\launch_bucket_dashboard.bat" (
        call "wrds_optionmetrics_exploration\launch_bucket_dashboard.bat"
        exit /b %errorlevel%
    )
    echo Missing launcher: wrds_optionmetrics_exploration\launch_bucket_dashboard.bat
    pause
    exit /b 1
)

if "%APP_CHOICE%"=="2" (
    if exist "us_macro_regime_radar\launch_dashboard.bat" (
        call "us_macro_regime_radar\launch_dashboard.bat"
        exit /b %errorlevel%
    )
    echo Missing launcher: us_macro_regime_radar\launch_dashboard.bat
    pause
    exit /b 1
)

echo Invalid selection.
pause
exit /b 1
