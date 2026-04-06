# setup_winutils.ps1
# Run once from PowerShell (as your normal user - no admin needed).
# Downloads winutils.exe for Hadoop 3.3.6 (compatible with PySpark 3.5.x / Hadoop 3.3.4)
# and sets HADOOP_HOME permanently in your user environment.
#
# Usage:
#   Open PowerShell, navigate to the project folder, then run:
#   .\setup_winutils.ps1

$hadoopHome = "$env:USERPROFILE\hadoop"
$binDir     = "$hadoopHome\bin"
$winutilsUrl = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe"
$hadoopDllUrl = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/hadoop.dll"

Write-Host "Creating $binDir ..."
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

Write-Host "Downloading winutils.exe ..."
Invoke-WebRequest -Uri $winutilsUrl -OutFile "$binDir\winutils.exe"

Write-Host "Downloading hadoop.dll (required alongside winutils.exe) ..."
Invoke-WebRequest -Uri $hadoopDllUrl -OutFile "$binDir\hadoop.dll"

Write-Host "Setting HADOOP_HOME in user environment ..."
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", $hadoopHome, "User")

Write-Host ""
Write-Host "Done. HADOOP_HOME = $hadoopHome"
Write-Host "Restart your terminal (or PowerShell session) for the env var to take effect."
Write-Host "Then re-run any PySpark script - model serialization should work."
