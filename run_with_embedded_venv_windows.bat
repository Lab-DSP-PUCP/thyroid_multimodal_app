@echo off
REM Ejecuta la app usando el entorno embebido en .\venv (sin instalar globalmente)
setlocal enabledelayedexpansion

REM Ubicación del script (carpeta del proyecto)
set "HERE=%~dp0"
cd /d "%HERE%"

REM Verifica Python
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python no esta en PATH. Instala Python 3.11 o 3.12 y vuelve a intentar.
  pause
  exit /b 1
)

REM Si no existe venv, lo creamos e instalamos dependencias SOLO dentro de la carpeta
if not exist venv (
  echo [INFO] Creando entorno virtual local...
  python -m venv venv || goto :fail
  echo [INFO] Instalando dependencias locales...
  venv\Scripts\python -m pip install --upgrade pip
  venv\Scripts\python -m pip install -r requirements.txt || goto :fail
)

echo [INFO] Levantando servidor Flask en http://127.0.0.1:5000
set "FLASK_APP=app.py"
set "FLASK_ENV=production"
venv\Scripts\python app.py
goto :eof

:fail
echo [ERROR] No se pudo preparar el entorno local. Revisa requirements.txt o tu conexion.
exit /b 1
