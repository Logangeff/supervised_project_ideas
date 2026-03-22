@echo off
cd /d "%~dp0"
python -m src.main --phase all
if errorlevel 1 (
  echo Pipeline failed.
  pause
  exit /b 1
)
echo.
echo Outputs written under data\processed and outputs\.
pause
