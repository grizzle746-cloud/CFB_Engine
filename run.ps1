
# ======= Edit these for your matchup =======
$InputPath = "$PSScriptRoot\sportsline_cfb_single_export.csv"
$Home = "ND"
$Away = "ARK"
$HomeTotal = 36
$AwayTotal = 23
$OutDir = "$PSScriptRoot\out"
# ===========================================

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Installing dependencies (first run only)..."
python -m pip install -r "$PSScriptRoot\requirements.txt" | Out-Null

Write-Host "Running CFB pipeline..."
python "$PSScriptRoot\cfb_cli.py" run --input "$InputPath" --home $Home --away $Away --home-total $HomeTotal --away-total $AwayTotal --out-dir "$OutDir"

Write-Host "`nDone. Outputs in: $OutDir"
