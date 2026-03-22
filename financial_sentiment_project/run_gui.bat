@echo off
setlocal

cd /d "%~dp0"

python -m app.main

if errorlevel 1 (
    echo.
    echo GUI launch failed.
    pause
    exit /b 1
)
