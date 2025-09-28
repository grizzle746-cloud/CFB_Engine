
#!/usr/bin/env bash
set -e

# ======= Edit these for your matchup =======
INPUT="$(dirname "$0")/sportsline_cfb_single_export.csv"
HOME_T="ND"
AWAY_T="ARK"
HOME_TOTAL=36
AWAY_TOTAL=23
OUTDIR="$(dirname "$0")/out"
# ===========================================

mkdir -p "$OUTDIR"

echo "Installing dependencies (first run only)..."
python3 -m pip install -r "$(dirname "$0")/requirements.txt" >/dev/null 2>&1 || true

echo "Running CFB pipeline..."
python3 "$(dirname "$0")/cfb_cli.py" run --input "$INPUT" --home "$HOME_T" --away "$AWAY_T" --home-total "$HOME_TOTAL" --away-total "$AWAY_TOTAL" --out-dir "$OUTDIR"

echo
echo "Done. Outputs in: $OUTDIR"
