#!/usr/bin/env python3
import argparse, json, os
import pandas as pd
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def main():
    ap = argparse.ArgumentParser(description="Generate ticket.csv with win_prob from board.csv")
    ap.add_argument("--board", required=True)
    ap.add_argument("--model", required=False, help="Optional logistic model JSON (a,b). If missing or invalid, fall back")
    ap.add_argument("--void-prob", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=6, help="take top-N by Edge (desc)")
    ap.add_argument("--fallback-slope", type=float, default=0.35,
                    help="Slope for gentle logistic fallback on Edge when no model is used")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    board = pd.read_csv(args.board).copy()
    board = board.sort_values("Edge", ascending=False).head(args.top)

    # Try model; if missing/invalid → gentle fallback
    use_model = False
    if args.model and os.path.exists(args.model):
        try:
            with open(args.model) as f:
                m = json.load(f)
            a = float(m["a"]); b = float(m["b"])
            z = a + b * board["Edge"]
            if np.isfinite(z).all():
                board["win_prob"] = sigmoid(z)
                use_model = True
        except Exception:
            use_model = False

    if not use_model:
        board["win_prob"] = sigmoid(args.fallback_slope * board["Edge"])

    board["void_prob"] = float(args.void_prob)
    board[["Prop","Player","win_prob","void_prob"]].to_csv(args.out, index=False)
    print(f"Wrote ticket -> {args.out} (rows={len(board)}) | model_used={use_model} | slope={args.fallback_slope}")

if __name__ == "__main__":
    main()
