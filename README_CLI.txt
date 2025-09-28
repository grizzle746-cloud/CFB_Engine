
CFB One-Click CLI – Quick Start
===============================

Files
-----
- cfb_cli.py               (pipeline: single-export -> normalized -> engine-menu board)
- requirements.txt         (pandas only)
- run.bat                  (Windows double-click)
- run.ps1                  (PowerShell)
- run.sh                   (macOS/Linux)
- sportsline_cfb_single_export.csv  (your single-export input)

Windows (double-click)
----------------------
1) Put these files in your project folder.
2) Open sportsline_cfb_single_export.csv and paste SportsLine numbers.
3) Right-click run.bat -> Edit the team codes and totals at the top if needed.
4) Double-click run.bat. Outputs go to .\out

PowerShell (if needed)
----------------------
- If you get an execution policy message: open PowerShell as Admin and run:
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

macOS/Linux
-----------
chmod +x ./run.sh
./run.sh

Outputs
-------
- out/sportsline_cfb_single_export.csv
- out/normalized_cfb_projections.csv
- out/cfb_engine_menu_board.csv

Notes
-----
- Team codes in the CSV must match your rows (e.g., ND vs ARK).
- You can re-run with different totals/spreads by editing the run script header.
- If Python isn't found, install Python 3.9+ and re-run.
