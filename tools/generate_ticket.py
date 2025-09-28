import argparse, json
import numpy as np
import pandas as pd

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def main():
    ap = argparse.ArgumentParser(description="Generate ticket.csv with win_prob from board.csv")
    ap.add_argument("--board", required=True, help="Path to board CSV")
    ap.add_argument("--model", required=False, help="Optional logistic model JSON (a,b)")
    ap.add_argument("--void-prob", required=False, type=float, default=0.0, help="Per-leg void probability")
    ap.add_argument("--top", required=True, type=int, help="take top-N by Edge (desc)")
    ap.add_argument("--fallback-slope", required=False, type=float, default=0.35, help="Slope for gentle logistic fallback")
    ap.add_argument("--out", required=True, help="Output ticket CSV")
    args = ap.parse_args()

    board = pd.read_csv(args.board).copy()
    board = board.sort_values("Edge", ascending=False).head(args.top)

    # guardrail for provider rules (needs 5 or 6 legs)
    if len(board) < 5:
        raise SystemExit(f"Need at least 5 legs, got {len(board)}. Check menu/stat_name mapping or use more projections.")

    # try model
    use_model = False
    if args.model:
        try:
            m = json.loads(open(args.model).read())
            a, b = float(m["a"]), float(m["b"])
            z = a + b * board["Edge"].astype(float)
            if np.isfinite(z).all() and len(board) > 0:
                board["win_prob"] = sigmoid(z)
                use_model = True
        except Exception:
            use_model = False

    # fallback
    if not use_model:
        board["win_prob"] = sigmoid(float(args.fallback_slope) * board["Edge"].astype(float))

    board["void_prob"] = float(args.void_prob)
    board[["Prop","Player","win_prob","void_prob"]].to_csv(args.out, index=False)
    print(f"Wrote ticket -> {args.out} (rows={len(board)}) | model_used={use_model} | slope={args.fallback_slope}")

if __name__ == "__main__":
    main()
