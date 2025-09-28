# path: sports_engine.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple, Set, Dict

import numpy as np
import pandas as pd


# -------------------- config & types --------------------
DEFAULT_BASE_WIN_PROB: float = 0.55  # fallback when Prob missing
REQUIRED_DISPLAY_COLS: Tuple[str, ...] = (
    "Player",
    "Team",
    "Market",
    "Pick",
    "Line",
    "DKLine",
    "Multiplier",
    "WinProb",
    "Edge",
    "EV",
    "RankScore",
    "KellyFrac",
    "Reason",
    "League",
    "Source",
    "Mode",
)


@dataclass(frozen=True)
class TicketSettings:
    ticket_size: int = 6
    num_tickets: int = 3
    diversity_penalty: float = 0.97  # cross-ticket mild penalty when player reused
    per_team_limit: Optional[int] = None  # None = unlimited; typical: 2
    seed: Optional[int] = None  # reproducibility


# -------------------- helpers --------------------
def _norm_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _safe_prod(arr: Iterable[float]) -> float:
    # Why: numerical stability for empty or all-NaN sequences
    vals = [float(v) for v in arr if pd.notna(v)]
    return float(np.prod(vals)) if vals else 0.0


def _safe_mean(arr: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in arr if pd.notna(v)]
    return float(np.mean(vals)) if vals else default


# -------------------- standardization & metrics --------------------
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize columns and add missing expected ones
    out.columns = [_norm_text(pd.Series([c]))[0] for c in out.columns]
    for col in ("Player", "Market", "Line"):
        if col not in out.columns:
            out[col] = np.nan

    # Normalize text columns if present
    for col in ("Player", "Team", "Market"):
        if col in out.columns:
            out[col] = _norm_text(out[col])

    # Normalize numerics
    for c in ("Line", "DKLine", "Multiplier", "Prob"):
        if c in out.columns:
            out[c] = _norm_num(out[c])

    if "DKLine" not in out.columns:
        out["DKLine"] = out["Line"]
    if "Multiplier" not in out.columns:
        out["Multiplier"] = 1.0

    return out


def _calc_metrics(df: pd.DataFrame, base_win_prob: float = DEFAULT_BASE_WIN_PROB) -> pd.DataFrame:
    out = df.copy()

    if "Prob" in out.columns:
        out["WinProb"] = out["Prob"].clip(0.0, 1.0).fillna(base_win_prob)
    else:
        out["WinProb"] = base_win_prob

    # EV under parlay-style multiplier semantics
    out["EV"] = out["WinProb"] * out["Multiplier"]

    # Edge & Pick relative to the book line
    diff = _norm_num(out["Line"]) - _norm_num(out["DKLine"])
    out["Edge"] = diff.fillna(0.0)
    out["Pick"] = np.where(diff >= 0, "MORE", "LESS")

    return out


def _dedupe_players(df: pd.DataFrame) -> pd.DataFrame:
    if "Player" not in df.columns:
        return df
    # Deterministic ordering for stability
    ordered = df.sort_values(
        ["Player", "EV", "WinProb", "Market", "DKLine"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    return ordered.groupby("Player", as_index=False, sort=False).head(1)


def _rank(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    mult = out["Multiplier"].replace(0, np.nan)

    if mode == "balanced":
        # Reward hit rate slightly more than payout, normalized by multiplier
        out["RankScore"] = 0.6 * out["WinProb"] + 0.4 * (out["EV"] / mult)
    else:  # "max"
        out["RankScore"] = 0.4 * out["WinProb"] + 0.6 * out["EV"]

    out["RankScore"] = out["RankScore"].fillna(0.0)

    # Deterministic tie-breakers
    out = out.sort_values(
        ["RankScore", "EV", "WinProb", "Player", "Market"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return out


def _market_group(market: str) -> str:
    m = (market or "").lower()
    if "rushing" in m:
        return "rush"
    if "receiving" in m or "receptions" in m:
        return "recv"
    if "pass" in m or "attempt" in m or "complet" in m:
        return "pass"
    if "td" in m or "touchdown" in m:
        return "td"
    return "misc"


def _pairwise_corr_penalty(p1: pd.Series, p2: pd.Series) -> float:
    # Same player: strong penalty
    if str(p1.get("Player", "")) == str(p2.get("Player", "")):
        return 0.25

    team1 = str(p1.get("Team", "") or "").strip()
    team2 = str(p2.get("Team", "") or "").strip()
    g1 = _market_group(str(p1.get("Market", "")))
    g2 = _market_group(str(p2.get("Market", "")))

    pen = 0.0
    if team1 and (team1 == team2):
        pen += 0.10 if g1 == g2 else 0.05
    # QB pass ↔ WR receptions mildly positive → no penalty
    return float(np.clip(pen, 0.0, 0.9))


def _kelly_fraction(win_prob: float, payout_mult: float) -> float:
    b = float(max(payout_mult - 1.0, 0.0))
    p = float(np.clip(win_prob, 0.0, 1.0))
    q = 1.0 - p
    if b <= 0.0:
        return 0.0
    f = (b * p - q) / b
    return float(np.clip(f, 0.0, 1.0))


# -------------------- public API --------------------
def run_engine(df: pd.DataFrame, league: str = "CFB", source: str = "sportsline", mode: str = "max") -> pd.DataFrame:
    """
    Build a ranked pick table from a normalized props CSV-like DataFrame.
    Expected columns (best-effort): Player, Team, Market, Line, DKLine, Multiplier, Prob.
    """
    data = _standardize(df)
    data = _calc_metrics(data)
    data = _dedupe_players(data)

    # Drop low-quality rows
    data = data.dropna(subset=["Player", "Market", "DKLine"])
    data = data[data["Player"].astype(str).str.len() > 0]
    data = data[data["Market"].astype(str).str.len() > 0]

    ranked = _rank(data, mode)

    # Metadata
    ranked["League"] = league
    ranked["Source"] = source
    ranked["Mode"] = mode
    ranked["Reason"] = np.where(
        ranked["Pick"].eq("MORE"),
        "Proj ≥ DKLine (lean OVER/MORE)",
        "Proj < DKLine (lean UNDER/LESS)",
    )
    ranked["KellyFrac"] = [
        _kelly_fraction(p, m) for p, m in zip(ranked["WinProb"].astype(float), ranked["Multiplier"].astype(float))
    ]

    # Column pruning
    existing = [c for c in REQUIRED_DISPLAY_COLS if c in ranked.columns]
    return ranked[existing].reset_index(drop=True)


def _ticket_score(rows: pd.DataFrame) -> float:
    """Ticket quality: product(WinProb) × correlation penalties × (1 + 0.15 * mean(EV))."""
    probs = rows["WinProb"].clip(0.0, 1.0).tolist() if "WinProb" in rows.columns else []
    base = _safe_prod(probs)
    # Correlation penalties
    pen = 1.0
    n = len(rows)
    for i, j in combinations(range(n), 2):
        pen *= (1.0 - _pairwise_corr_penalty(rows.iloc[i], rows.iloc[j]))
    ev_boost = _safe_mean(rows["EV"].tolist(), default=0.0) if "EV" in rows.columns and n else 0.0
    return base * pen * (1.0 + 0.15 * ev_boost)


def build_tickets(
    picks: pd.DataFrame,
    ticket_size: int = 6,
    num_tickets: int = 3,
    mode: str = "max",
    *,
    diversity_penalty: float = 0.97,
    per_team_limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Greedy ticket assembly with mild cross-ticket diversity and optional per-team cap.

    Parameters
    ----------
    picks : DataFrame
        Output of `run_engine`.
    ticket_size : int
        Legs per ticket (>=1).
    num_tickets : int
        Number of tickets to build (>=1).
    mode : str
        Ranking mode; forwarded for compatibility.
    diversity_penalty : float
        Multiplier (<1) applied when a player already used in previous tickets.
    per_team_limit : Optional[int]
        Max legs from the same team per ticket. None for unlimited.
    seed : Optional[int]
        RNG seed for deterministic tie-breaking.

    Returns
    -------
    DataFrame
        Ticket rows with TicketID, Leg, core pick fields, and TicketScore.
    """
    if ticket_size < 1 or num_tickets < 1:
        return pd.DataFrame(columns=["TicketID", "Leg", "Player", "Market", "Pick", "WinProb", "EV", "TicketScore"])

    df = picks.copy().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["TicketID", "Leg", "Player", "Market", "Pick", "WinProb", "EV", "TicketScore"])

    rng = np.random.default_rng(seed)  # Why: stable, reproducible selection among ties
    used_players: Set[str] = set()
    tickets: List[pd.DataFrame] = []

    # Static candidate pool with deterministic shuffle to break hard ties
    pool_records: List[Dict] = df.to_dict("records")
    # Shuffle once with RNG to avoid deterministic bias toward early rows
    if seed is not None:
        rng.shuffle(pool_records)

    for t in range(num_tickets):
        chosen: List[Dict] = []
        local_used: Set[str] = set()
        team_counts: Dict[str, int] = {}

        for _ in range(ticket_size):
            best_idx: Optional[int] = None
            best_score: float = -1e18

            for idx, r in enumerate(pool_records):
                player = str(r.get("Player", "") or "")
                team = str(r.get("Team", "") or "")
                if not player:
                    continue
                if player in local_used:
                    continue  # no repeats within a ticket
                if per_team_limit is not None and team:
                    if team_counts.get(team, 0) >= per_team_limit:
                        continue

                tmp_rows = pd.DataFrame(chosen + [r])
                score = _ticket_score(tmp_rows)
                if player in used_players:
                    score *= float(diversity_penalty)

                # Deterministic tie-break: prefer higher WinProb, then EV, then Player asc
                if score > best_score:
                    best_score = score
                    best_idx = idx
                elif np.isclose(score, best_score, rtol=0, atol=1e-12) and best_idx is not None:
                    rb = pool_records[best_idx]
                    wp_a, wp_b = r.get("WinProb", 0.0), rb.get("WinProb", 0.0)
                    if wp_a > wp_b:
                        best_idx = idx
                    elif np.isclose(wp_a, wp_b) and r.get("EV", 0.0) > rb.get("EV", 0.0):
                        best_idx = idx
                    elif (
                        np.isclose(wp_a, wp_b)
                        and np.isclose(r.get("EV", 0.0), rb.get("EV", 0.0))
                        and str(player) < str(rb.get("Player", ""))
                    ):
                        best_idx = idx

            if best_idx is None:
                break

            chosen.append(pool_records[best_idx])
            pl = str(pool_records[best_idx].get("Player", ""))
            tm = str(pool_records[best_idx].get("Team", "") or "")
            local_used.add(pl)
            if tm:
                team_counts[tm] = team_counts.get(tm, 0) + 1

        if len(chosen) == ticket_size:
            used_players.update(local_used)
            rows = pd.DataFrame(chosen)
            rows["TicketID"] = f"T{t+1}"
            rows["Leg"] = list(range(1, ticket_size + 1))
            rows["TicketScore"] = _ticket_score(rows)
            tickets.append(rows)

    if not tickets:
        return pd.DataFrame(columns=["TicketID", "Leg", "Player", "Market", "Pick", "WinProb", "EV", "TicketScore"])

    out = pd.concat(tickets, ignore_index=True)
    cols = [
        "TicketID",
        "Leg",
        "Player",
        "Team",
        "Market",
        "Pick",
        "Line",
        "DKLine",
        "Multiplier",
        "WinProb",
        "Edge",
        "EV",
        "TicketScore",
    ]
    existing = [c for c in cols if c in out.columns]
    return out[existing].sort_values(["TicketID", "Leg"]).reset_index(drop=True)
