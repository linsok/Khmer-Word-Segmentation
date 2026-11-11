@echo off
REM Test the Maximum Matching algorithms and CSV processing.
cd /d "%~dp0"
python -X utf8 -m pytest -q
python -X utf8 tests\evaluate.py
pause
