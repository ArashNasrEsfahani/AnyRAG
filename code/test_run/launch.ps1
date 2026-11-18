# Detached launch of the three resumable test workers (survives the agent session; each skips finished cells).
# Usage: powershell -File launch.ps1 [run|replay|judge|all]
param([string]$what = "all")
$py = "D:\KareOne\RAG Paper\.venv\Scripts\python.exe"
$fz = "D:\KareOne\RAG Paper\work\datasets_v2\test_freeze"
$tr = "D:\KareOne\RAG Paper\work\datasets_v2\test_run"
$env:PYTHONIOENCODING = "utf-8"
$jobs = @{ "run" = "run_test.py run --confirm-one-shot"; "replay" = "run_test.py replay --confirm-one-shot --follow"; "judge" = "run_test.py judge --confirm-one-shot --follow" }
foreach ($k in $jobs.Keys) {
  if ($what -ne "all" -and $what -ne $k) { continue }
  $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -like ("*" + $jobs[$k] + "*") }
  if ($running) { "$k already running"; continue }
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  Start-Process -FilePath $py -ArgumentList $jobs[$k] -WorkingDirectory $fz -WindowStyle Hidden -RedirectStandardOutput "$tr\${k}_stdout_$ts.log" -RedirectStandardError "$tr\${k}_stderr_$ts.log"
  "$k started"
}
