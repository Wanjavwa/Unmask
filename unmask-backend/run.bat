@echo off

cd /d "%~dp0"

echo Starting Unmask backend on port 8011 (use mobile/services/api.js BACKEND_PORT=8011)

echo Health: http://127.0.0.1:8011/health

echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-backend.ps1"

