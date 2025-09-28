
@echo off
REM ======= Edit these three lines for your matchup =======
set "INPUT=%~dp0sportsline_cfb_single_export.csv"
set "HOME=ND"
set "AWAY=ARK"
set "HOME_TOTAL=36"
set "AWAY_TOTAL=23"
set "OUTDIR=%~dp0out"
REM =======================================================

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Installing dependencies (first run only)...
python -m pip install -r "%~dp0requirements.txt" >nul 2>&1

echo Running CFB pipeline...
python "%~dp0cfb_cli.py" run --input "%INPUT%" --home %HOME% --away %AWAY% --home-total %HOME_TOTAL% --away-total %AWAY_TOTAL% --out-dir "%OUTDIR%"

echo.
echo Done. Outputs in: %OUTDIR%
pause
