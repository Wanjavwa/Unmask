# Stop stale Unmask backends, then start the current one.
$port = 8011
foreach ($p in 8000, 8010, 8011) {
  Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}
Start-Sleep -Seconds 2
Set-Location $PSScriptRoot
Write-Host "Starting Unmask backend on http://127.0.0.1:$port"
Write-Host "Health check: http://127.0.0.1:$port/health"
& "$PSScriptRoot\venv\Scripts\python.exe" -m uvicorn app:app --reload --host 0.0.0.0 --port $port
