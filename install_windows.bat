@echo off
setlocal
cd /d "%~dp0"
title HESHAM CPU Lite Installer
where py >nul 2>nul
if errorlevel 1 (echo Python launcher missing: install Python 3.11 or 3.12 x64. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
 py -3.12 -m venv .venv 2>nul
 if errorlevel 1 py -3.11 -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (echo Could not create .venv & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Installation failed; see error above. & pause & exit /b 1)
".venv\Scripts\python.exe" tools\self_test.py
if errorlevel 1 (echo Self-test failed. & pause & exit /b 1)
echo Installation done. Run run_windows.bat
pause
