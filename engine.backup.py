# path: engine.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# --------------------------
# Default menu (unchanged)
# --------------------------
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
    ("Rush + Rec TDs", None, 1.0),  # special: rush+rec (legacy composite)
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


# --------------------------
# Data structures
# --------------------------
@dataclass(frozen=True)
class MenuItem:
    prop_name: str
    stat_name: Optional[str]  # None for composite
    threshold: float


def _to_menu(items: Sequence[Tuple[str, Optional[str], float]]) -> List[MenuItem]:
    return [MenuItem(*t) for t in items]


# --------------------------
# Menu loader
# --------------------------
def _load_menu(path: Optional[str]) -> List[MenuItem]:
    """
    Load menu from JSON or CSV. Falls back to DEFAULT_ENGINE_MENU.

    JSON accepts either:
      - [{"prop": "...", "stat": "... or null", "threshold": number}, ...]
      - [["prop", "stat or null", threshold], ...]
    CSV requires columns: prop, threshold [, stat]
    """
    if not path:
        return _to_menu(DEFAULT_ENGINE_MENU)

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Menu file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        out: List[MenuItem] = []
        for item in data:
            if isinstance(item, dict):
                out.append(
                    MenuItem(
                        prop_name=str(item["prop"]),
                        stat_name=(str(item["stat"]) if item.get("stat") is not None else None),
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


# --------------------------
# Input validation
# --------------------------
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


# --------------------------
# Board builder
# --------------------------
def build_board(
    long_df: pd.DataFrame,
    menu: Optional[Sequence[MenuItem | Tuple[str, Optional[str], float]]] = None,
) -> pd.DataFrame:
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
        # Normalize tuples -> MenuItem for safety across environments.
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

    # --- FIX 1: normalize pivot column labels to plain trimmed strings
    wide.columns = [str(c).strip() for c in wide.columns]

    rows: List[Dict[str, object]] = []

    # Iterate wide rows and compute edges for each menu item.
    for _, r in wide.iterrows():
        rushing_tds = r.get("Rushing TDs", pd.NA)
        receiving_tds = r.get("Receiving TDs", pd.NA)

        for mi in menu_items:
            # Legacy composite supported by default menu
            if mi.prop_name == "Rush + Rec TDs":
                comp = 0.0
                if pd.notna(rushing_tds):
                    comp += float(rushing_tds)
                if pd.notna(receiving_tds):
                    comp += float(receiving_tds)
                proj = comp
            else:
                # --- FIX 2: safe Series lookup
                if mi.stat_name and (mi.stat_name in r.index):
                    proj = r[mi.stat_name]
                else:
                    proj = float("nan")

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


# --------------------------
# I/O helpers
# --------------------------
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


# --------------------------
# CLI: evaluate
# --------------------------
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


# --------------------------
# EV engine (ticket-ev)
# --------------------------
def _ticket_ev(
    win_prob: List[float],
    void_prob: List[float],
    min_multipliers: Dict[int, Dict[int, float]],
    stake: float,
    entry_sizes: Iterable[int],
) -> Tuple[float, Dict[int, Dict[int, float]]]:
    """
    Dynamic-program the distribution over (non-void m, correct k).
    Each leg has: hit = (1-void)*win, miss = (1-void)*(1-win), void = void.
    Returns:
      ev: expected payout (in currency units)
      dist: nested dict dist[m][k] = probability
    """
    # dp[m][k] = probability after processing i legs
    dp: Dict[int, Dict[int, float]] = {0: {0: 1.0}}

    for w, v in zip(win_prob, void_prob):
        hit = (1.0 - v) * max(0.0, min(1.0, w))
        voi = max(0.0, min(1.0, v))
        mis = max(0.0, 1.0 - voi - hit)

        next_dp: Dict[int, Dict[int, float]] = {}
        for m, row in dp.items():
            for k, p in row.items():
                # void -> (m, k)
                next_dp.setdefault(m, {}).setdefault(k, 0.0)
                next_dp[m][k] += p * voi
                # hit -> (m+1, k+1)
                next_dp.setdefault(m + 1, {}).setdefault(k + 1, 0.0)
                next_dp[m + 1][k + 1] += p * hit
                # miss -> (m+1, k)
                next_dp.setdefault(m + 1, {}).setdefault(k, 0.0)
                next_dp[m + 1][k] += p * mis
        dp = next_dp

    # Compute EV using provider minimum multipliers per entry size.
    allowed = set(int(x) for x in entry_sizes)
    ev = 0.0
    for m, row in dp.items():
        if m not in allowed:
            continue
        mult_m = {int(k): float(v) for k, v in (min_multipliers.get(m, {}) or {}).items()}
        for k, p in row.items():
            mult = mult_m.get(k, 0.0)
            if mult > 0 and p > 0:
                ev += p * (mult * stake)

    return ev, dp


def cli_ticket_ev(args: argparse.Namespace) -> None:
    cfg_path = Path(args.provider_config)
    if not cfg_path.exists():
        raise SystemExit(f"Provider config not found: {cfg_path}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed reading provider config: {e}")

    entry_sizes = cfg.get("entry_sizes", [])
    if not entry_sizes:
        raise SystemExit("Provider config must include 'entry_sizes'.")

    min_multipliers_raw = cfg.get("min_multipliers", {})
    # Normalize keys to int->int->float
    min_multipliers: Dict[int, Dict[int, float]] = {}
    for m_str, obj in min_multipliers_raw.items():
        m = int(m_str)
        min_multipliers[m] = {int(k): float(v) for k, v in obj.items()}

    stake = float(cfg.get("stake", 0.0))

    ticket_path = Path(args.ticket)
    if not ticket_path.exists():
        raise SystemExit(f"Ticket file not found: {ticket_path}")
    tdf = pd.read_csv(ticket_path)

    need_cols = {"Prop", "Player", "win_prob", "void_prob"}
    if not need_cols.issubset(tdf.columns):
        raise SystemExit(f"Ticket CSV must include columns: {sorted(need_cols)}")

    wins = [float(x) for x in tdf["win_prob"].tolist()]
    voids = [float(x) for x in tdf["void_prob"].tolist()]

    ev, dist = _ticket_ev(wins, voids, min_multipliers, stake=stake, entry_sizes=entry_sizes)

    # Write summary
    out = Path(args.export)
    out.parent.mkdir(parents=True, exist_ok=True)
    legs = len(wins)
    pd.DataFrame([{"legs": legs, "stake": stake, "ev": ev}]).to_csv(out, index=False)
    print(f"Wrote ticket EV -> {out} (legs={legs}, ev={ev:.6f})")

    # Optional distribution export
    if args.dist_export:
        rows = []
        for m, row in sorted(dist.items()):
            for k, p in sorted(row.items()):
                rows.append({"non_void": m, "correct": k, "prob": p})
        pd.DataFrame(rows).to_csv(Path(args.dist_export), index=False)
        print(f"Wrote distribution -> {args.dist_export}")


# --------------------------
# CLI wiring
# --------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Minimal CFB/NFL engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # evaluate
    ap_eval = sub.add_parser("evaluate", help="Evaluate projections to engine-menu board")
    ap_eval.add_argument("--league", required=True, choices=["cfb", "nfl"], help="League")
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

    # ticket-ev
    ap_tev = sub.add_parser("ticket-ev", help="Compute EV for a ticket using provider config")
    ap_tev.add_argument("--provider-config", required=True, help="Path to provider JSON")
    ap_tev.add_argument("--ticket", required=True, help="Path to ticket CSV (Prop,Player,win_prob,void_prob)")
    ap_tev.add_argument("--export", required=True, help="Where to write EV CSV")
    ap_tev.add_argument("--dist-export", required=False, default=None, help="Optional distribution CSV")
    ap_tev.set_defaults(func=cli_ticket_ev)

    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
