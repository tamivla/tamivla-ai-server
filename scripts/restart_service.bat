REM \scripts\restart_service.bat
@echo off
echo ========================================
echo    Tamivla AI Server - Restart
echo ========================================
echo.

echo [1/3] Stopping service...
sc stop TamivlaAIServer
timeout /t 3 /nobreak >nul

echo [2/3] Starting service...
sc start TamivlaAIServer

if %ERRORLEVEL% EQU 0 (
    echo [3/3] SUCCESS: Service restarted
    echo Check status: sc query TamivlaAIServer
    echo Web interface: http://localhost:8000/docs
) else (
    echo [3/3] ERROR: Restart failed. Code: %ERRORLEVEL%
    echo Check logs: storage\logs\nssm_*.log
)

echo.
pause