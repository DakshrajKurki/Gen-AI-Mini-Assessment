@echo off
REM Starts the FastAPI backend at http://127.0.0.1:8000  (docs: /docs)
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python -m uvicorn backend.main:app --reload
pause
