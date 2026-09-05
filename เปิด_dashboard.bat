@echo off
cd /d "%~dp0"
echo Starting Footprint Signal Lab...
echo Waiting for dashboard, then opening browser...
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start "" http://localhost:8501/"
py -m streamlit run backtest_app.py --server.headless true --server.port 8501
pause
