@echo off
REM Open the extracted project folder, then start Streamlit in UTF-8 mode.
cd /d "%~dp0"
python -X utf8 -m streamlit run app.py
pause
