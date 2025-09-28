#!/usr/bin/env python3
import json, argparse
import pandas as pd
import numpy as np

def read_cfg(p):
    with open(p, "r") as f:
        cfg = json.load(f)
    cfg["min_multipliers"] = {
        int(m): {int(k): float(v) for k, v in d.items()}
        for m, d in cfg["min_multipliers"].items()
    }
    cfg["entry_sizes"] = [int(x) for x in cfg["entry_sizes"]]
    cfg.setdefault("stake", 1.0)
    cfg.setdefault("void_policy", "reduce_size")
    return cfg

def read_ticket(p):
    df = pd.read_csv(p)
    if "win_prob" not in df.columns:
        raise SystemExit("ticket CSV must include win_prob")
    if "void_prob" not in df.columns:
        df["void_prob"] = 0.0
    for c in ["win_prob","void_prob"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(0,1)
    return df

def ev_conv(win_probs, void_probs, multipliers, stake=1.0, entry_sizes=None):
    n = len(win_probs)
    layers = [np.array([1.0])] + [np.zeros(1) for _ in range(n)]  # layers[m] is poly over k
    for i in range(n):
        v = float(void_probs[i])
        p_cond = float(win_probs[i])
        p = (1-v)*p_cond
        w = (1-v)*(1-p_cond)
        new_layers = [np.zeros_like(l) for l in layers]
        def grow(poly, size):
            if len(poly) >= size: return poly
            out = np.zeros(size); out[:len(poly)] = poly; return out
        for m, poly in enumerate(layers):
            if poly.sum() == 0: continue
            # void: stay m
            new_layers[m] = grow(new_layers[m], len(poly))
            new_layers[m][:len(poly)] += poly * v
            # correct: m+1, shift k +1
            if m+1 < len(layers):
                tgt = grow(new_layers[m+1], len(poly)+1)
                tgt[1:len(poly)+1] += poly * p
                new_layers[m+1] = tgt
            # wrong: m+1, no shift
            if m+1 < len(layers):
                tgt = grow(new_layers[m+1], len(poly))
                tgt[:len(poly)] += poly * w
                new_layers[m+1] = tgt
        layers = new_layers

    valid_m = set(entry_sizes or multipliers.keys())
    ev = 0.0
    for m, poly in enumerate(layers):
        if m not in valid_m: 
            continue
        pay = multipliers.get(m, {})
        for k, prob in enumerate(poly):
            if prob > 0:
                ev += stake * prob * float(pay.get(k, 0.0))
    return ev, layers

def main():
    ap = argparse.ArgumentParser(description="Ticket EV (sidecar) for Pick6-like payouts")
    ap.add_argument("--provider-config", required=True)
    ap.add_argument("--ticket", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--dist-export", required=False)
    args = ap.parse_args()

    cfg = read_cfg(args.provider_config)
    df = read_ticket(args.ticket)

    ev, layers = ev_conv(
        win_probs=df["win_prob"].tolist(),
        void_probs=df["void_prob"].tolist(),
        multipliers=cfg["min_multipliers"],
        stake=float(cfg["stake"]),
        entry_sizes=cfg["entry_sizes"],
    )

    pd.DataFrame([{"legs": len(df), "stake": cfg["stake"], "ev": ev}]).to_csv(args.export, index=False)
    if args.dist_export:
        rows = []
        for m, poly in enumerate(layers):
            for k, pr in enumerate(poly):
                if pr > 0:
                    rows.append({"non_void": m, "correct": k, "prob": pr})
        pd.DataFrame(rows).to_csv(args.dist_export, index=False)
    print(f"EV = {ev:.4f} (legs={len(df)}) -> {args.export}")

if __name__ == "__main__":
    main()
