param(
  [string]$league = "nfl",
  [string]$projections = ".\data\nfl.csv",
  [string]$menu = ".\menus\pick6_nfl.json",
  [int]$top = 6,                     # <-- 6-leg default
  [float]$voidProb = 0.0,
  [string]$provider = ".\cfg\provider.pick6.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ensure venv is active if you use one; else comment next two lines
if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) { Write-Host "Note: no venv found. Using system Python." }
else { .\.venv\Scripts\Activate.ps1 | Out-Null }

# 1) Build board
python .\engine.py evaluate --league $league --menu $menu --projections $projections --export .\out\board_$league.csv

# 2) Generate ticket (gentle logistic fallback if no model)
python .\tools\generate_ticket.py --board .\out\board_$league.csv --top $top --void-prob $voidProb --out .\data\ticket.from_board.csv

# 3) EV with provider minimums (stake set in provider config)
python .\engine.py ticket-ev --provider-config $provider --ticket .\data\ticket.from_board.csv --export .\out\ticket_ev.csv --dist-export .\out\ticket_dist.csv

# 4) Show EV result
Get-Content .\out\ticket_ev.csv
