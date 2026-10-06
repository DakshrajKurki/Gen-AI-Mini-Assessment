@echo off
REM Starts the Streamlit app at http://localhost:8501
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python -m streamlit run frontend/app.py
pause
