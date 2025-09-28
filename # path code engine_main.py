# path: engine_main.py
from __future__ import annotations

import argparse
import sys
import pathlib
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from sports_engine import run_engine, build_tickets


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Pick-6 full engine")
    ap.add_argument("--input", required=True, help="Path to input CSV (e.g., exports/props.csv)")
    ap.add_argument("--out_dir", default="outputs", help="Output directory")
    ap.add_argument("--league", default="CFB", choices=["CFB", "NFL"])
    ap.add_argument("--source", default="sportsline", choices=["sportsline", "auto"])
    ap.add_argument("--mode", default="max", choices=["max", "balanced"])
    ap.add_argument("--ticket_size", type=int, default=6)
    ap.add_argument("--num_tickets", type=int, default=3)
    args = ap.parse_args(argv)

    # Why: prevent silent bad configs producing junk output
    if args.ticket_size < 1:
        ap.error("--ticket_size must be >= 1")
    if args.num_tickets < 1:
        ap.error("--num_tickets must be >= 1")
    return args


def _utc_stamp() -> str:
    # Why: deterministic, sortable filenames across environments
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def _read_csv_strict(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERR] Input file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[ERR] Not a file: {path}")

    try:
        # Low-memory off for correctness; encoding fixed per upstream contract.
        df = pd.read_csv(path, encoding="utf-8", dtype=None, keep_default_na=True)
    except Exception as exc:
        raise SystemExit(f"[ERR] Failed to read CSV: {path} ({exc})")

    if df.empty:
        raise SystemExit(f"[ERR] Input CSV has no rows: {path}")

    # Gentle normalization only; avoid forcing schema unknown to us.
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    return df


def _ensure_df(obj, name: str) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise SystemExit(f"[ERR] {name} returned non-DataFrame: {type(obj)!r}")
    return obj


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    inp = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise SystemExit(f"[ERR] Failed to create output directory '{out_dir}': {exc}")

    ts = _utc_stamp()

    df = _read_csv_strict(inp)

    # Engine pass 1: picks
    try:
        picks = _ensure_df(
            run_engine(df, league=args.league, source=args.source, mode=args.mode),
            "run_engine",
        )
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"[ERR] run_engine failed: {exc}")

    if picks.empty:
        # Why: let the pipeline proceed but signal emptiness explicitly
        print("[WARN] Engine produced 0 picks; continuing to write empty CSV.", file=sys.stderr)

    picks_path = out_dir / f"picks_{ts}.csv"
    try:
        picks.to_csv(picks_path, index=False, encoding="utf-8")
    except Exception as exc:
        raise SystemExit(f"[ERR] Failed to write picks CSV '{picks_path}': {exc}")

    # Engine pass 2: tickets
    try:
        tickets = _ensure_df(
            build_tickets(
                picks,
                ticket_size=args.ticket_size,
                num_tickets=args.num_tickets,
                mode=args.mode,
            ),
            "build_tickets",
        )
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"[ERR] build_tickets failed: {exc}")

    if tickets.empty:
        print("[WARN] No tickets were built; writing empty CSV.", file=sys.stderr)

    tickets_path = out_dir / f"tickets_{ts}.csv"
    try:
        tickets.to_csv(tickets_path, index=False, encoding="utf-8")
    except Exception as exc:
        raise SystemExit(f"[ERR] Failed to write tickets CSV '{tickets_path}': {exc}")

    print(f"Wrote {picks_path} and {tickets_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
