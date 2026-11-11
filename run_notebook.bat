@echo off
REM Launch Jupyter from the project root with Unicode support.
cd /d "%~dp0"
python -X utf8 -m notebook
pause
