# Install the Tracewright plugin into KiCad 10 (per-user).
#   tools\install_plugin.ps1 [-Exe path\to\tracewright.exe] [-KicadVersion 10.0]
# Copies the plugin executable, manifest and icons into
#   <Documents>\KiCad\<version>\plugins\com.tracewright.router\
# Restart KiCad's PCB editor afterwards; the "Tracewright: Autoroute" button appears in its
# toolbar (Preferences > Plugins must have "Enable KiCad API" turned on).
param(
    [string]$Exe = "",
    [string]$KicadVersion = "10.0"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Exe) { $Exe = Join-Path $root "dist\tracewright.exe" }
if (-not (Test-Path $Exe)) { throw "plugin executable not found: $Exe (build with: cargo build --release)" }
$docs = [Environment]::GetFolderPath("MyDocuments")
$dest = Join-Path $docs "KiCad\$KicadVersion\plugins\com.tracewright.router"
New-Item -ItemType Directory -Force $dest | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dest "icons") | Out-Null
Copy-Item $Exe (Join-Path $dest "tracewright.exe") -Force
Copy-Item (Join-Path $root "plugin\icons\*.png") (Join-Path $dest "icons") -Force
# KiCad resolves the entry point itself; an absolute path works regardless of version details.
$manifest = Get-Content (Join-Path $root "plugin\plugin.json") -Raw | ConvertFrom-Json
$manifest.actions[0].entrypoint = (Join-Path $dest "tracewright.exe")
$json = $manifest | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText((Join-Path $dest "plugin.json"), $json, (New-Object Text.UTF8Encoding($false)))
Write-Host "Installed to $dest"
