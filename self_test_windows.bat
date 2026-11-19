@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python tools\self_test.py
python tools\runtime_model_test.py
pause
