
import argparse
import pandas as pd
from pathlib import Path

STAT_MAP = {
    "pass_att": "Passing Attempts",
    "pass_comp": "Completions",
    "pass_yd": "Passing Yards",
    "pass_td": "Passing TDs",
    "pass_int": "Interceptions",
    "rush_att": "Rushing Attempts",
    "rush_yd": "Rushing Yards",
    "rush_td": "Rushing TDs",
    "rec_rec": "Receptions",
    "rec_yd": "Receiving Yards",
    "rec_td": "Receiving TDs",
    "fumbles": "Fumbles",
}

ENGINE_MENU = [
    ("60+ Receiving Yards", "Receiving Yards", 60.0),
    ("80+ Rushing Yards", "Rushing Yards", 80.0),
    ("250+ Passing Yards", "Passing Yards", 250.0),
    ("4+ Receptions", "Receptions", 4.0),
    ("20+ Completions", "Completions", 20.0),
    ("32+ Passing Attempts", "Passing Attempts", 32.0),
    ("14+ Rush Attempts", "Rushing Attempts", 14.0),
    ("2+ Passing Touchdowns", "Passing TDs", 2.0),
    ("Passing Touchdowns", "Passing TDs", 1.0),
    ("Rush + Rec TDs", None, 1.0),  # special: rush_td + rec_td
]

def fill_totals_spread(df, home, away, home_total, away_total):
    if home and away and (home_total is not None) and (away_total is not None):
        for idx, row in df.iterrows():
            if row.get("team") == home and row.get("opponent") == away:
                df.at[idx, "team_total"] = home_total
                df.at[idx, "opp_total"] = away_total
                df.at[idx, "spread"] = -(home_total - away_total)
            elif row.get("team") == away and row.get("opponent") == home:
                df.at[idx, "team_total"] = away_total
                df.at[idx, "opp_total"] = home_total
                df.at[idx, "spread"] = (home_total - away_total)
    return df

def normalize_long(df):
    id_cols = ["player_name","pos","team","opponent","game_date","kickoff_et","source","team_total","opp_total","spread","notes"]
    stat_cols = [c for c in df.columns if c in STAT_MAP]
    long = df[id_cols + stat_cols].melt(
        id_vars=id_cols,
        value_vars=stat_cols,
        var_name="stat_key",
        value_name="projection"
    )
    long["stat_name"] = long["stat_key"].map(STAT_MAP)
    long = long.drop(columns=["stat_key"])
    long["projection"] = pd.to_numeric(long["projection"], errors="coerce")
    long = long.dropna(subset=["projection"])
    long = long[[
        "player_name","pos","team","opponent","game_date","kickoff_et",
        "stat_name","projection","source","team_total","opp_total","spread","notes"
    ]]
    return long

def build_engine_menu_board(long_df):
    wide = long_df.pivot_table(
        index=["player_name","pos","team","opponent","game_date","kickoff_et","source","team_total","spread"],
        columns="stat_name",
        values="projection",
        aggfunc="mean"
    ).reset_index()

    rows = []
    for _, r in wide.iterrows():
        for prop_name, stat_name, thr in ENGINE_MENU:
            if prop_name == "Rush + Rec TDs":
                rr = (r.get("Rushing TDs", 0) if pd.notna(r.get("Rushing TDs", float("nan"))) else 0) + \
                     (r.get("Receiving TDs", 0) if pd.notna(r.get("Receiving TDs", float("nan"))) else 0)
                proj = rr
            else:
                proj = r.get(stat_name, float("nan"))
            if pd.notna(proj):
                edge = proj - thr
                rows.append({
                    "Prop": prop_name,
                    "Player": r["player_name"],
                    "Pos": r["pos"],
                    "Team": r["team"],
                    "Opp": r["opponent"],
                    "Projection": round(proj, 3),
                    "Threshold": thr,
                    "Edge": round(edge, 3),
                    "Team Total": r.get("team_total", None),
                    "Spread": r.get("spread", None),
                    "Source": r["source"],
                })
    out = pd.DataFrame(rows)
    out["Team Total"] = pd.to_numeric(out["Team Total"], errors="coerce")
    out = out.sort_values(by=["Prop","Edge","Team Total"], ascending=[True, False, False])
    return out

def main():
    ap = argparse.ArgumentParser(description="CFB CLI: single-export -> normalized -> engine-menu board")
    ap.add_argument("run", nargs="?", help="Run the pipeline", default="run")
    ap.add_argument("--input", required=True, help="Path to sportsline_cfb_single_export.csv")
    ap.add_argument("--out-dir", default="./out", help="Output directory")
    ap.add_argument("--home", help="Home/favorite team code (e.g., ND)")
    ap.add_argument("--away", help="Away team code (e.g., ARK)")
    ap.add_argument("--home-total", type=float, help="Projected team total for home team")
    ap.add_argument("--away-total", type=float, help="Projected team total for away team")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Fill totals/spread if provided
    if args.home and args.away and args.home_total is not None and args.away_total is not None:
        df = fill_totals_spread(df, args.home, args.away, args.home_total, args.away_total)

    # Save possibly-updated wide
    wide_out = out_dir / "sportsline_cfb_single_export.csv"
    df.to_csv(wide_out, index=False)

    # Normalize
    long = normalize_long(df)
    long_out = out_dir / "normalized_cfb_projections.csv"
    long.to_csv(long_out, index=False)

    # Engine-menu board
    board = build_engine_menu_board(long)
    board_out = out_dir / "cfb_engine_menu_board.csv"
    board.to_csv(board_out, index=False)

    print(f"Wrote: {wide_out}")
    print(f"Wrote: {long_out}")
    print(f"Wrote: {board_out}")

if __name__ == "__main__":
    main()
