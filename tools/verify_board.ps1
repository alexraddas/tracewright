# Route a KiCad board headlessly and check the result with KiCad's own DRC.
#   tools\verify_board.ps1 -Pcb path\to\board.kicad_pcb -Name demo [-Backend gpu] [-Cli path\to\tracewright-cli.exe] [-Extra "--pitch 0.1"]
# The board is copied (with its .kicad_pro) into testdata\<Name>\, stripped of tracks, routed,
# written back, and checked with `kicad-cli pcb drc`.
param(
    [Parameter(Mandatory = $true)][string]$Pcb,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Backend = "gpu",
    [string]$Cli = "",
    [string]$Extra = ""
)
$ErrorActionPreference = "Continue"
$kicad = "C:\Program Files\KiCad\10.0\bin"
$py = "$kicad\python.exe"
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root "testdata\$Name"
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item $Pcb "$dir\board.kicad_pcb" -Force
$pro = [IO.Path]::ChangeExtension($Pcb, ".kicad_pro")
if (-not (Test-Path $pro)) { $pro = Get-ChildItem (Split-Path $Pcb) -Filter *.kicad_pro | Select-Object -First 1 -ExpandProperty FullName }
if ($pro) { Copy-Item $pro "$dir\board.kicad_pro" -Force; Copy-Item $pro "$dir\routed.kicad_pro" -Force }
& $py "$root\tools\kicad_dump.py" "$dir\board.kicad_pcb" "$dir\board.json" --strip
$args = @("route", "$dir\board.json", "--out", "$dir\result.json", "--backend", $Backend, "--dump-settings", "$dir\settings.json") + ($Extra -split " " | Where-Object { $_ })
& $Cli @args 2>&1 | Where-Object { $_ -notmatch "outside the board outline" } | ForEach-Object { "$_" }
& $py "$root\tools\kicad_apply.py" "$dir\board.kicad_pcb" "$dir\board.json" "$dir\result.json" "$dir\routed.kicad_pcb" --strip 2>&1 | Where-Object { $_ -notmatch "memory leak" }
& "$kicad\kicad-cli.exe" pcb drc --refill-zones --format json --severity-error --output "$dir\drc.json" "$dir\routed.kicad_pcb" | Out-Null
& $py "$root\tools\drc_summary.py" "$dir\drc.json" "$dir\board.json"
& $py "$root\tools\render_png.py" "$dir\board.json" "$dir\result.json" "$dir\routed.png" --scale 8 | Out-Null
