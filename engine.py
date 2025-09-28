from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ---- Default menu ----
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
    ("Rush + Rec TDs", None, 1.0),
]

REQUIRED_COLS = {
    "player_name","pos","team","opponent","game_date","kickoff_et",
    "stat_name","projection","source",
}

@dataclass(frozen=True)
class MenuItem:
    prop_name: str
    stat_name: Optional[str]  # None for composite
    threshold: float

def _to_menu(items: Sequence[Tuple[str, Optional[str], float]]) -> List[MenuItem]:
    return [MenuItem(*t) for t in items]

def _load_menu(path: Optional[str]) -> List[MenuItem]:
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
                out.append(MenuItem(
                    prop_name=str(item["prop"]),
                    stat_name=(str(item["stat"]) if item.get("stat") is not None else None),
                    threshold=float(item["threshold"]),
                ))
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
        prop_col = cols["prop"]; thr_col = cols["threshold"]; stat_col = cols.get("stat")
        return [
            MenuItem(
                prop_name=str(r[prop_col]),
                stat_name=(str(r[stat_col]) if (stat_col and pd.notna(r[stat_col])) else None),
                threshold=float(r[thr_col]),
            )
            for _, r in df.iterrows()
        ]

    raise SystemExit(f"Unsupported menu file type: {p.suffix} (use .json or .csv)")

def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in projections: {sorted(missing)}")
    for col in ["player_name","pos","team","opponent","game_date","kickoff_et","stat_name","source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    if "team_total" in df.columns:
        df["team_total"] = pd.to_numeric(df["team_total"], errors="coerce")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    return df

def build_board(
    long_df: pd.DataFrame,
    menu: Optional[Sequence[MenuItem | Tuple[str, Optional[str], float]]] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=[
            "Prop","Player","Pos","Team","Opp","Projection","Threshold","Edge",
            "Team Total","Spread","Source",
        ])

    menu_items: List[MenuItem] = (
        _to_menu(DEFAULT_ENGINE_MENU)
        if menu is None else [mi if isinstance(mi, MenuItem) else MenuItem(*mi) for mi in menu]
    )
    df = _validate_df(long_df.copy())

    index_cols = ["player_name","pos","team","opponent","game_date","kickoff_et","source"]
    if "team_total" in df.columns: index_cols.append("team_total")
    if "spread" in df.columns:     index_cols.append("spread")

    aggfunc = {"mean": "mean", "median": "median", "max": "max"}[agg]
    wide = (
        df.pivot_table(index=index_cols, columns="stat_name", values="projection",
                       aggfunc=aggfunc, observed=False)
          .reset_index()
    )
    wide.columns = [str(c).strip() for c in wide.columns]

    rows: List[Dict[str, object]] = []
    for _, r in wide.iterrows():
        rushing_tds = r.get("Rushing TDs", pd.NA)
        receiving_tds = r.get("Receiving TDs", pd.NA)
        for mi in menu_items:
            if mi.prop_name == "Rush + Rec TDs":
                comp = 0.0
                if pd.notna(rushing_tds): comp += float(rushing_tds)
                if pd.notna(receiving_tds): comp += float(receiving_tds)
                proj = comp
            else:
                proj = r[mi.stat_name] if (mi.stat_name and mi.stat_name in r.index) else float("nan")
            if pd.notna(proj):
                edge = float(proj) - float(mi.threshold)
                rows.append({
                    "Prop": mi.prop_name,
                    "Player": r["player_name"], "Pos": r["pos"], "Team": r["team"], "Opp": r["opponent"],
                    "Projection": round(float(proj), 3), "Threshold": float(mi.threshold),
                    "Edge": round(edge, 3),
                    "Team Total": pd.to_numeric(r.get("team_total", pd.NA), errors="coerce"),
                    "Spread": pd.to_numeric(r.get("spread", pd.NA), errors="coerce"),
                    "Source": r["source"],
                })
    out = pd.DataFrame.from_records(rows)
    if out.empty: return out
    out = out.sort_values(
        by=["Prop","Edge","Team Total","Player"],
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

def _apply_filters(
    df: pd.DataFrame,
    game_date: Optional[str],
    sources: Optional[List[str]],
    pos: Optional[List[str]] = None,
    team: Optional[List[str]] = None,
    opp: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df
    if game_date:
        out = out[out["game_date"].astype(str).str.strip() == str(game_date).strip()]
    if sources:
        srcs = {s.strip() for s in sources if s and str(s).strip()}
        if srcs: out = out[out["source"].isin(srcs)]
    if pos:
        poss = {p.strip() for p in pos if p and str(p).strip()}
        if poss: out = out[out["pos"].astype(str).str.strip().isin(poss)]
    if team:
        teams = {t.strip() for t in team if t and str(t).strip()}
        if teams: out = out[out["team"].astype(str).str.strip().isin(teams)]
    if opp:
        opps = {o.strip() for o in opp if o and str(o).strip()}
        if opps: out = out[out["opponent"].astype(str).str.strip().isin(opps)]
    return out

def cli_evaluate(args: argparse.Namespace) -> None:
    df = _read_projections(args.projections)
    df = _validate_df(df)
    if args.verbose: print(f"[evaluate] loaded rows={len(df)}")
    if df.empty: raise SystemExit("Projections CSV is empty after validation.")

    df = _apply_filters(df, args.date, args.source, args.pos, args.team, args.opp)
    if args.verbose: print(f"[evaluate] after filters rows={len(df)}")
    if df.empty: raise SystemExit("No rows left after applying filters.")

    menu = _load_menu(args.menu)
    board = build_board(df, menu, agg=args.agg)
    if args.preview:
        print(board.head(args.preview).to_string(index=False))

    if args.dry_run:
        print("[evaluate] dry-run: not writing CSV")
        return

    out_path = Path(args.export); out_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_path, index=False)
    print(f"Wrote evaluation board -> {out_path} (rows={len(board)})")

def _ticket_ev(
    win_prob: List[float],
    void_prob: List[float],
    min_multipliers: Dict[int, Dict[int, float]],
    stake: float,
    entry_sizes: Iterable[int],
):
    dp: Dict[int, Dict[int, float]] = {0:{0:1.0}}
    for w, v in zip(win_prob, void_prob):
        hit = (1.0 - v) * max(0.0, min(1.0, w))
        voi = max(0.0, min(1.0, v))
        mis = max(0.0, 1.0 - voi - hit)
        nxt: Dict[int, Dict[int, float]] = {}
        for m,row in dp.items():
            for k,p in row.items():
                nxt.setdefault(m,   {}).setdefault(k,   0.0); nxt[m][k]   += p*voi
                nxt.setdefault(m+1, {}).setdefault(k+1, 0.0); nxt[m+1][k+1] += p*hit
                nxt.setdefault(m+1, {}).setdefault(k,   0.0); nxt[m+1][k]   += p*mis
        dp = nxt

    allowed = set(int(x) for x in entry_sizes)
    ev = 0.0
    for m,row in dp.items():
        if m not in allowed: continue
        mults = {int(k): float(v) for k,v in (min_multipliers.get(m, {}) or {}).items()}
        for k,p in row.items():
            mult = mults.get(k, 0.0)
            if mult>0 and p>0:
                ev += p * (mult * stake)
    return ev, dp

def cli_ticket_ev(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.provider_config).read_text(encoding="utf-8"))
    entry_sizes = cfg.get("entry_sizes", [])
    min_multipliers: Dict[int, Dict[int, float]] = {
        int(m): {int(k): float(v) for k, v in d.items()}
        for m, d in cfg.get("min_multipliers", {}).items()
    }
    stake = float(cfg.get("stake", 0.0))

    tdf = pd.read_csv(args.ticket)
    need = {"Prop","Player","win_prob","void_prob"}
    if not need.issubset(tdf.columns):
        raise SystemExit(f"Ticket CSV must include columns: {sorted(need)}")

    wins = [float(x) for x in tdf["win_prob"].tolist()]
    voids = [float(x) for x in tdf["void_prob"].tolist()]
    ev, dist = _ticket_ev(wins, voids, min_multipliers, stake=stake, entry_sizes=entry_sizes)

    if args.preview:
        print(f"[ticket-ev] legs={len(wins)} stake={stake} EV={ev:.6f}")
        top = sorted([(m,k,p) for m,row in dist.items() for k,p in row.items()], key=lambda x: -x[2])[:args.preview]
        print("[ticket-ev] top outcomes:")
        for m,k,p in top:
            print(f"  non_void={m}, correct={k}, prob={p:.6f}")

    if args.dry_run:
        print("[ticket-ev] dry-run: not writing CSVs")
        return

    out = Path(args.export); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"legs": len(wins), "stake": stake, "ev": ev}]).to_csv(out, index=False)
    if args.dist_export:
        rows = [{"non_void": m, "correct": k, "prob": p} for m, row in sorted(dist.items()) for k, p in sorted(row.items())]
        pd.DataFrame(rows).to_csv(args.dist_export, index=False)

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Minimal CFB/NFL engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_eval = sub.add_parser("evaluate", help="Evaluate projections to engine-menu board")
    ap_eval.add_argument("--league", required=True, choices=["cfb","nfl"])
    ap_eval.add_argument("--projections", required=True)
    ap_eval.add_argument("--export", required=True)
    ap_eval.add_argument("--menu", required=False, default=None)
    ap_eval.add_argument("--date", required=False, default=None)
    ap_eval.add_argument("--source", required=False, action="append")
    ap_eval.add_argument("--pos", required=False, action="append", help="Filter: include positions (repeatable)")
    ap_eval.add_argument("--team", required=False, action="append", help="Filter: include teams (repeatable)")
    ap_eval.add_argument("--opp", required=False, action="append", help="Filter: include opponents (repeatable)")
    ap_eval.add_argument("--agg", choices=["mean","median","max"], default="mean", help="Aggregation for duplicate projections")
    ap_eval.add_argument("--preview", type=int, default=0, help="Show first N rows of the board")
    ap_eval.add_argument("--dry-run", action="store_true", help="Preview only; do not write --export CSV")
    ap_eval.add_argument("--verbose", action="store_true", help="Print row counts/stats")
    ap_eval.set_defaults(func=cli_evaluate)

    ap_tev = sub.add_parser("ticket-ev", help="Compute EV for a ticket")
    ap_tev.add_argument("--provider-config", required=True)
    ap_tev.add_argument("--ticket", required=True)
    ap_tev.add_argument("--export", required=True)
    ap_tev.add_argument("--dist-export", required=False, default=None)
    ap_tev.add_argument("--preview", type=int, default=0, help="Show top-N probability outcomes")
    ap_tev.add_argument("--dry-run", action="store_true", help="Preview only; do not write CSVs")
    ap_tev.set_defaults(func=cli_ticket_ev)

    return ap

def main() -> None:
    args = _build_arg_parser().parse_args()
    args.func(args)

__version__ = open('VERSION').read().strip() if Path('VERSION').exists() else '0.0.0'

if __name__ == "__main__":
    main()
