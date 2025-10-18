@echo off
echo ========================================
echo    Tamivla AI Server - Stop
echo ========================================
echo.

sc stop TamivlaAIServer

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Service stopped
    echo Check status: sc query TamivlaAIServer
) else (
    echo ERROR: Service stop failed. Code: %ERRORLEVEL%
    echo Check logs: storage\logs\nssm_*.log
)

echo.
pause