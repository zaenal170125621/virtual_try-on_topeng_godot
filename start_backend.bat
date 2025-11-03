@echo off
echo ========================================
echo   Virtual Try-On Topeng - Backend
echo ========================================
echo.

cd /d "%~dp0"

echo Mengaktifkan virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Menjalankan backend server di http://localhost:5000
echo Tekan CTRL+C untuk stop server
echo.

python app.py

pause
