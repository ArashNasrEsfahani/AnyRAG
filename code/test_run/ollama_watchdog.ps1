# Keeps Ollama alive during the true test: every 20 s, if the API does not answer, pause nothing, just restart
# Ollama and log it. The primary runner's infra-error rule (one rerun) covers any cell that hit the gap.
$log = "D:\KareOne\RAG Paper\work\datasets_v2\test_run\ollama_watchdog.log"
$exe = "C:\Users\arash\AppData\Local\Programs\Ollama\ollama.exe"
while ($true) {
  $ok = $false
  try { Invoke-RestMethod -Uri http://127.0.0.1:11434/api/tags -TimeoutSec 10 | Out-Null; $ok = $true } catch {}
  if (-not $ok) {
    Add-Content $log ("{0} ollama down -> restart" -f (Get-Date -Format s))
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Process $exe serve -WindowStyle Hidden
    Start-Sleep -Seconds 8
    try { Invoke-RestMethod -Uri http://127.0.0.1:11434/api/embed -Method Post -Body '{"model":"bge-m3","input":"warm"}' -ContentType 'application/json' -TimeoutSec 60 | Out-Null; Add-Content $log ("{0} restarted, embed ok" -f (Get-Date -Format s)) } catch { Add-Content $log ("{0} restart embed failed" -f (Get-Date -Format s)) }
  }
  if (-not (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_test.py' -and $_.CommandLine -notmatch 'Get-CimInstance|ollama_watchdog' })) { Add-Content $log ("{0} no test workers left -> watchdog exits" -f (Get-Date -Format s)); break }
  Start-Sleep -Seconds 20
}
