@echo off
setlocal
cd /d "%~dp0"
title HESHAM Fire and Smoke AI - CPU Lite v1.5
set "HESHAM_ACCELERATOR=cpu"
set "HESHAM_CPU_THREADS=4"
set "OMP_NUM_THREADS=4"
if not exist ".venv\Scripts\python.exe" (
 echo Python environment missing. Please run install_windows.bat first.
 pause
 exit /b 1
)
if not exist "models\fire\model.tflite" (
 echo ERROR: Bundled model.tflite missing. Extract complete ZIP again.
 pause
 exit /b 1
)
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" run.py
if errorlevel 1 pause
