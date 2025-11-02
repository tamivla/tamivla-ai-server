REM \scripts\install_nssm.bat
@echo off
chcp 65001 >nul
echo ========================================
echo Установка Tamivla AI Server через NSSM
echo ========================================

cd /d C:\Tamivla_AI_Server

REM Удаляем старую службу, если существует
sc query TamivlaAIServer >nul && (
    echo Останавливаем старую службу...
    net stop TamivlaAIServer >nul 2>&1
    timeout /t 3 /nobreak >nul
    sc delete TamivlaAIServer >nul
    timeout /t 2 /nobreak >nul
)

REM Устанавливаем новую службу через NSSM
echo Устанавливаем службу через NSSM...
scripts\nssm.exe install TamivlaAIServer "C:\Tamivla_AI_Server\venv\Scripts\python.exe" "C:\Tamivla_AI_Server\src\main.py"

REM Настраиваем параметры службы
scripts\nssm.exe set TamivlaAIServer AppDirectory "C:\Tamivla_AI_Server"
scripts\nssm.exe set TamivlaAIServer DisplayName "Tamivla AI Server"
scripts\nssm.exe set TamivlaAIServer Description "Высокопроизводительный сервер для AI моделей от Tamivla Industrial Group"
scripts\nssm.exe set TamivlaAIServer AppStdout "C:\Tamivla_AI_Server\storage\logs\nssm_stdout.log"
scripts\nssm.exe set TamivlaAIServer AppStderr "C:\Tamivla_AI_Server\storage\logs\nssm_stderr.log"
scripts\nssm.exe set TamivlaAIServer AppRotateFiles 1
scripts\nssm.exe set TamivlaAIServer AppRotateOnline 1
scripts\nssm.exe set TamivlaAIServer AppRotateSeconds 86400

REM Запускаем службу
echo Запускаем службу...
net start TamivlaAIServer

echo ========================================
echo ✅ Установка завершена!
echo 🔍 Проверьте:
echo    - http://localhost:8000/docs
echo    - Логи: storage\logs\nssm_*.log
echo ========================================
pause