@echo off
title DataSheet - Armazenamento em Papel
echo ========================================================
echo   Iniciando o DataSheet...
echo   Abrindo a interface visual no seu navegador!
echo ========================================================
timeout /t 2 >nul
start http://127.0.0.1:5000
python app.py
pause
