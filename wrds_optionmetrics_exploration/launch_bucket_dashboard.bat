@echo off
setlocal
cd /d %~dp0

for /f %%P in ('powershell -NoProfile -Command "$l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0); $l.Start(); $p = $l.LocalEndpoint.Port; $l.Stop(); Write-Output $p"') do set "DASHBOARD_PORT=%%P"

echo Launching bucket dashboard on port %DASHBOARD_PORT%...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:%DASHBOARD_PORT%'"
python -m streamlit run app/streamlit_bucket_dashboard.py --server.port %DASHBOARD_PORT%

endlocal
