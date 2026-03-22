@echo off
setlocal
cd /d "%~dp0"
python -m src.main --phase all --start 2018-01-01 --universe-mode top_liquid --limit 300
