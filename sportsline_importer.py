# sportsline_importer.py
# Cleans raw SportsLine projection CSV into engine-ready format

import pandas as pd
import sys
import os

# Required engine format
HEADERS = ["Player", "PropType", "Line", "Projection", "Odds", "MyProb", "League", "Team", "Opponent"]

def clean_sportsline(raw_csv: str, output_csv: str = "C:\\CFB_Engine\\sportsline_projections.csv"):
    if not os.path.exists(raw_csv):
        print(f"❌ File not found: {raw_csv}")
        return
    
    # Try to read raw SportsLine CSV
    df = pd.read_csv(raw_csv)

    # Normalize headers
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common SportsLine names → engine names
    rename_map = {
        "player": "Player",
        "line": "Line",
        "proj": "Projection",
        "projection": "Projection",
        "stat": "PropType",
        "prop": "PropType"
    }
    df = df.rename(columns=rename_map)

    # If PropType missing, fill with "Unknown"
    if "PropType" not in df.columns:
        df["PropType"] = "Unknown"

    # Add missing engine columns
    for col in HEADERS:
        if col not in df.columns:
            df[col] = ""

    # Default Odds = -110
    df["Odds"] = df["Odds"].replace("", -110).fillna(-110)

    # Reorder
    df = df[HEADERS]

    # Save cleaned file
    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned file saved: {output_csv}")
    print(f"Rows processed: {len(df)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sportsline_importer.py raw_file.csv")
    else:
        clean_sportsline(sys.argv[1])
