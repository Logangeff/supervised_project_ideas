@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app/streamlit_app.py
if errorlevel 1 (
  echo.
  echo Dashboard launch failed.
  pause
)
