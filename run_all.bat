@echo off
cd /d C:\CFB_Engine

echo ================================
echo Importing SportsLine projections
echo ================================
python sportsline_importer.py

echo ================================
echo Running Sports Engine
echo ================================
python sports_engine.py --csv sportsline_projections.csv --bankroll 1000 --stake 20 --pool 12 --slate CFB_Test

echo ================================
echo Finished. Press any key to close.
echo ================================
pause >nul