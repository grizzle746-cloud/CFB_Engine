# path: engine.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# Default menu, mirrors original behavior.
DEFAULT_ENGINE_MENU: List[Tuple[str, Optional[str], float]] = [
    ("60+ Receiving Yards", "Receiving Yards", 60.0),
    ("80+ Rushing Yards", "Rushing Yards", 80.0),
    ("250+ Passing Yards", "Passing Yards", 250.0),
    ("4+ Receptions", "Receptions", 4.0),
    ("20+ Completions", "Completions", 20.0),
    ("32+ Passing Attempts", "Passing Attempts", 32.0),
    ("14+ Rush Attempts", "Rushing Attempts", 14.0),
    ("2+ Passing Touchdowns", "Passing TDs", 2.0),
    ("Passing Touchdowns", "Passing TDs", 1.0),
    ("Rush + Rec TDs", None, 1.0),  # special: rush+rec
]


REQUIRED_COLS = {
    "player_name",
    "pos",
    "team",
    "opponent",
    "game_date",
    "kickoff_et",
    "stat_name",
    "projection",
    "source",
}


@dataclass(frozen=True)
class MenuItem:
    prop_name: str
    stat_name: Optional[str]  # None for composite
    threshold: float


def _to_menu(items: Sequence[Tuple[str, Optional[str], float]]) -> List[MenuItem]:
    return [MenuItem(*t) for t in items]


def _load_menu(path: Optional[str]) -> List[MenuItem]:
    """Load menu from JSON or CSV. Falls back to DEFAULT_ENGINE_MENU."""
    if not path:
        return _to_menu(DEFAULT_ENGINE_MENU)

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Menu file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        # Expect list of {prop, stat, threshold} or [prop, stat, threshold]
        out: List[MenuItem] = []
        for item in data:
            if isinstance(item, dict):
                out.append(
                    MenuItem(
                        prop_name=item["prop"],
                        stat_name=item.get("stat"),
                        threshold=float(item["threshold"]),
                    )
                )
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                prop, stat, thr = item
                out.append(MenuItem(str(prop), (str(stat) if stat is not None else None), float(thr)))
            else:
                raise SystemExit(f"Invalid menu JSON item: {item!r}")
        return out

    if p.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        need = {"prop", "threshold"}
        if not need.issubset(cols):
            raise SystemExit(f"Menu CSV must include columns: {sorted(need)}")
        prop_col = cols["prop"]
        thr_col = cols["threshold"]
        stat_col = cols.get("stat")
        out = [
            MenuItem(
                prop_name=str(r[prop_col]),
                stat_name=(str(r[stat_col]) if (stat_col and pd.notna(r[stat_col])) else None),
                threshold=float(r[thr_col]),
            )
            for _, r in df.iterrows()
        ]
        return out

    raise SystemExit(f"Unsupported menu file type: {p.suffix} (use .json or .csv)")


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in projections: {sorted(missing)}")

    # Trim common string columns for stability.
    for col in ["player_name", "pos", "team", "opponent", "game_date", "kickoff_et", "stat_name", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Coerce numerics.
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    if "team_total" in df.columns:
        df["team_total"] = pd.to_numeric(df["team_total"], errors="coerce")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    return df


def build_board(long_df: pd.DataFrame, menu: Optional[Sequence[MenuItem | Tuple[str, Optional[str], float]]] = None) -> pd.DataFrame:
    """
    Turn normalized (long) projections into a menu-aligned board.

    Parameters
    ----------
    long_df : DataFrame
        Long/normalized projections with REQUIRED_COLS + optional 'team_total', 'spread'.
    menu : Sequence[MenuItem or tuple], optional
        Custom menu. Defaults to DEFAULT_ENGINE_MENU.

    Returns
    -------
    DataFrame
        Columns: Prop, Player, Pos, Team, Opp, Projection, Threshold, Edge, Team Total, Spread, Source
    """
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "Prop",
                "Player",
                "Pos",
                "Team",
                "Opp",
                "Projection",
                "Threshold",
                "Edge",
                "Team Total",
                "Spread",
                "Source",
            ]
        )

    menu_items: List[MenuItem]
    if menu is None:
        menu_items = _to_menu(DEFAULT_ENGINE_MENU)
    else:
        # Normalize tuples -> MenuItem
        menu_items = [mi if isinstance(mi, MenuItem) else MenuItem(*mi) for mi in menu]

    df = _validate_df(long_df.copy())

    # Pivot to wide; mean aggregation handles multiple rows per player/stat/source.
    index_cols = [
        "player_name",
        "pos",
        "team",
        "opponent",
        "game_date",
        "kickoff_et",
        "source",
    ]
    if "team_total" in df.columns:
        index_cols.append("team_total")
    else:
        df["team_total"] = pd.NA
        index_cols.append("team_total")
    if "spread" in df.columns:
        index_cols.append("spread")
    else:
        df["spread"] = pd.NA
        index_cols.append("spread")

    wide = (
        df.pivot_table(
            index=index_cols,
            columns="stat_name",
            values="projection",
            aggfunc="mean",
            observed=False,
        )
        .reset_index()
    )

    rows = []
    # Iterate wide rows and compute edges for each menu item.
    for _, r in wide.iterrows():
        rushing_tds = r.get("Rushing TDs", pd.NA)
        receiving_tds = r.get("Receiving TDs", pd.NA)

        for mi in menu_items:
            if mi.prop_name == "Rush + Rec TDs":
                # Why: composite prop not present as a single stat.
                comp = 0.0
                if pd.notna(rushing_tds):
                    comp += float(rushing_tds)
                if pd.notna(receiving_tds):
                    comp += float(receiving_tds)
                proj = comp
            else:
                proj = r.get(mi.stat_name, float("nan")) if mi.stat_name else float("nan")

            if pd.notna(proj):
                edge = float(proj) - float(mi.threshold)
                rows.append(
                    {
                        "Prop": mi.prop_name,
                        "Player": r["player_name"],
                        "Pos": r["pos"],
                        "Team": r["team"],
                        "Opp": r["opponent"],
                        "Projection": round(float(proj), 3),
                        "Threshold": float(mi.threshold),
                        "Edge": round(edge, 3),
                        "Team Total": pd.to_numeric(r.get("team_total"), errors="coerce"),
                        "Spread": pd.to_numeric(r.get("spread"), errors="coerce"),
                        "Source": r["source"],
                    }
                )

    out = pd.DataFrame.from_records(rows)

    if out.empty:
        return out

    # Deterministic sort; ties broken by Player for stability.
    out = out.sort_values(
        by=["Prop", "Edge", "Team Total", "Player"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _read_projections(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Projections file not found: {p}")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise SystemExit(f"Failed to read projections CSV: {p} ({e})")
    return df


def _apply_filters(df: pd.DataFrame, game_date: Optional[str], sources: Optional[List[str]]) -> pd.DataFrame:
    out = df
    if game_date:
        out = out[out["game_date"].astype(str).str.strip() == str(game_date).strip()]
    if sources:
        srcs = {s.strip() for s in sources if s and str(s).strip()}
        if srcs:
            out = out[out["source"].isin(srcs)]
    return out


def cli_evaluate(args: argparse.Namespace) -> None:
    df = _read_projections(args.projections)
    df = _validate_df(df)

    # Basic sanity after validation.
    if df.empty:
        raise SystemExit("Projections CSV is empty after validation.")

    # Optional filtering for convenience.
    df = _apply_filters(df, args.date, args.source)

    if df.empty:
        raise SystemExit("No rows left after applying filters (date/source).")

    menu = _load_menu(args.menu)

    board = build_board(df, menu)

    out_path = Path(args.export)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_path, index=False)
    print(f"Wrote evaluation board -> {out_path} (rows={len(board)})")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Minimal CFB engine for test evaluation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_eval = sub.add_parser("evaluate", help="Evaluate projections to engine-menu board")
    ap_eval.add_argument("--league", required=True, choices=["cfb"], help="League (cfb)")
    ap_eval.add_argument("--projections", required=True, help="Path to normalized projections CSV")
    ap_eval.add_argument("--export", required=True, help="Where to write the evaluation CSV")
    ap_eval.add_argument(
        "--menu",
        required=False,
        default=None,
        help="Optional custom menu file (.json or .csv) with columns/keys: prop, stat (optional), threshold",
    )
    ap_eval.add_argument("--date", required=False, default=None, help="Optional exact game_date filter (e.g., 2025-09-28)")
    ap_eval.add_argument(
        "--source",
        required=False,
        action="append",
        help="Optional projection source filter; can be repeated (e.g., --source MyModel --source Market)",
    )
    ap_eval.set_defaults(func=cli_evaluate)
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
