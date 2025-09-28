#!/usr/bin/env python3
import argparse, json
import pandas as pd
import numpy as np

# simple logistic fit via Newton-Raphson
def fit_logit(edge: np.ndarray, y: np.ndarray, iters: int = 50):
    X = np.c_[np.ones_like(edge), edge]  # intercept + edge
    w = np.zeros(2)
    for _ in range(iters):
        z = X @ w
        p = 1/(1+np.exp(-z))
        g = X.T @ (y - p)
        W = p*(1-p)
        H = -(X.T * W) @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.linalg.norm(step) < 1e-6:
            break
    return float(w[0]), float(w[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', required=True, help='CSV with columns: edge,hit')
    ap.add_argument('--out', required=True, help='Where to save model JSON')
    args = ap.parse_args()
    df = pd.read_csv(args.history)[['edge','hit']].dropna()
    a,b = fit_logit(df['edge'].to_numpy(float), df['hit'].to_numpy(float))
    with open(args.out, 'w') as f:
        json.dump({'a':a,'b':b,'created_by':'calibrate_win_prob.py'}, f)
    print(f'Saved model -> {args.out} (a={a:.4f}, b={b:.4f}, n={len(df)})')

if __name__ == '__main__':
    main()
