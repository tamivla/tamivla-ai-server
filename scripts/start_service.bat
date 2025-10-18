@echo off
echo ========================================
echo    Tamivla AI Server - Start
echo ========================================
echo.

sc start TamivlaAIServer

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Service started
    echo Check status: sc query TamivlaAIServer
    echo Web interface: http://localhost:8000/docs
) else (
    echo ERROR: Service start failed. Code: %ERRORLEVEL%
    echo Check logs: storage\logs\nssm_*.log
)

echo.
pause