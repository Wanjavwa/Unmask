@echo off
cd /d "%~dp0"
echo Starting Unmask backend at http://127.0.0.1:8000
echo API docs: http://127.0.0.1:8000/docs
echo.
uvicorn app:app --reload --host 127.0.0.1 --port 8000
