# Slate Builder — Pro v2
# + Constraints (max/team, min distinct props)
# + Per-PropType Sigma defaults (writes Sigma column)
# + Live table preview, delete, save/load
# + Settings persistence

import os, re, json, subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import pyperclip
from io import StringIO
from datetime import datetime

BASE_DIR = r"C:\CFB_Engine"
SLATE_DIR = os.path.join(BASE_DIR, "slates")
CFB_PATH = os.path.join(SLATE_DIR, "cfb", "sportsline_projections_CFB.csv")
NFL_PATH = os.path.join(SLATE_DIR, "nfl", "sportsline_projections_NFL.csv")
ENGINE_PATH = os.path.join(BASE_DIR, "sports_engine.py")
CONFIG_PATH = os.path.join(BASE_DIR, "slate_builder_settings.json")

HEADERS = ["Player","PropType","Line","Projection","Odds","MyProb","League","Team","Opponent","Sigma"]

CANON = {
    "player":"Player","name":"Player",
    "prop":"PropType","stat":"PropType","market":"PropType","proptype":"PropType",
    "line":"Line","projection":"Projection","proj":"Projection",
    "odds":"Odds","myprob":"MyProb","league":"League","team":"Team","opponent":"Opponent",
    "sigma":"Sigma",
}

PROP_TYPES = [
    "Passing Yards","Rushing Yards","Receiving Yards",
    "Receptions","Completions","Pass Attempts","Rushing Attempts",
    "Passing Touchdowns","Rush + Rec TDs","Interceptions"
]

DEFAULTS = {
    "league": "CFB",
    "bankroll": 1000.0,
    "stake": 20.0,
    "pool_by_league": {"CFB": 40, "NFL": 30},
    "default_odds": -110.0,
    "dedupe": True,
    "max_per_team": 2,
    "min_prop_kinds": 3,
    "sigma_by_prop": {
        "Passing Yards": 14.0,
        "Rushing Yards": 12.0,
        "Receiving Yards": 13.0,
        "Receptions": 2.2,
        "Completions": 2.5,
        "Pass Attempts": 3.5,
        "Rushing Attempts": 3.0,
        "Passing Touchdowns": 0.7,
        "Rush + Rec TDs": 0.6,
        "Interceptions": 0.5
    }
}

def ensure_dirs():
    os.makedirs(os.path.join(SLATE_DIR, "cfb"), exist_ok=True)
    os.makedirs(os.path.join(SLATE_DIR, "nfl"), exist_ok=True)

def save_settings(data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def load_settings():
    d = DEFAULTS.copy()
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                x = json.load(f); 
                # deep merge for sigma map/pools
                for k in ("sigma_by_prop","pool_by_league"):
                    if k in x: d[k].update(x[k])
                x.pop("sigma_by_prop", None); x.pop("pool_by_league", None)
                d.update(x)
    except Exception:
        pass
    return d

def _clean_headers(cols):
    out=[]
    for c in cols:
        s=str(c)
        s=re.sub(r"[^\x00-\x7F]+","",s).strip()
        key=re.sub(r"[\s\-_]+","",s).lower()
        out.append(CANON.get(key, s))
    return out

def read_clipboard_df():
    text = pyperclip.paste()
    if not text: raise ValueError("Clipboard is empty. Copy a SportsLine table first.")
    try:
        return pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(StringIO(text))

class SlateModel:
    def __init__(self, dedupe=True, sigma_map=None, default_odds=-110):
        self.df = pd.DataFrame(columns=HEADERS)
        self.dedupe = dedupe
        self.sigma_map = sigma_map or {}

        self.default_odds = default_odds

    def append_clip(self, league, proptype, team=None, opponent=None):
        raw = read_clipboard_df()
        raw.columns = _clean_headers(raw.columns)

        # required minimal fields
        if "Player" not in raw.columns: raise ValueError("Player column not found in clipboard.")
        if "Line" not in raw.columns and "Projection" not in raw.columns:
            raise ValueError("Need Line/Projection columns in clipboard.")

        # normalize/augment
        out = pd.DataFrame()
        out["Player"] = raw["Player"]
        out["PropType"] = proptype
        out["Line"] = raw["Line"] if "Line" in raw.columns else None
        out["Projection"] = raw["Projection"] if "Projection" in raw.columns else None
        out["Odds"] = self.default_odds
        out["MyProb"] = ""
        out["League"] = league
        out["Team"] = team or ""
        out["Opponent"] = opponent or ""
        # per-prop sigma default
        out["Sigma"] = self.sigma_map.get(proptype, "")

        merged = pd.concat([self.df, out], ignore_index=True)
        if self.dedupe:
            merged = merged.drop_duplicates(subset=["Player","PropType","Line"], keep="last")
        self.df = merged

    def clear(self): self.df = pd.DataFrame(columns=HEADERS)
    def delete_indices(self, idxs): 
        if idxs: self.df = self.df.drop(idxs).reset_index(drop=True)
    def count(self): return len(self.df)
    def counts_by_prop(self):
        return {} if self.df.empty else self.df["PropType"].value_counts().to_dict()
    def save_to_path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_csv(path, index=False, encoding="utf-8")
    def load_from_path(self, path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = _clean_headers(df.columns)
        # ensure all headers exist
        for h in HEADERS:
            if h not in df.columns: df[h] = ""
        self.df = df[HEADERS]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Slate Builder — Pro v2")
        self.geometry("950x600")
        ensure_dirs()
        self.settings = load_settings()
        self.model = SlateModel(
            dedupe=self.settings.get("dedupe", True),
            sigma_map=self.settings.get("sigma_by_prop", {}),
            default_odds=self.settings.get("default_odds", -110.0),
        )

        top = ttk.Frame(self, padding=10); top.pack(fill="x")

        self.league = tk.StringVar(value=self.settings.get("league","CFB"))
        ttk.Label(top, text="League").grid(row=0, column=0, sticky="w")
        ttk.Combobox(top, textvariable=self.league, values=["CFB","NFL"], width=6, state="readonly").grid(row=0, column=1, padx=(6,18))

        self.proptype = tk.StringVar(value="Passing Yards")
        ttk.Label(top, text="PropType").grid(row=0, column=2, sticky="w")
        ttk.Combobox(top, textvariable=self.proptype, values=PROP_TYPES, width=22, state="readonly").grid(row=0, column=3, padx=(6,18))

        self.team = tk.StringVar(); self.opp  = tk.StringVar()
        ttk.Label(top, text="Team").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.team, width=14).grid(row=0, column=5, padx=(6,12))
        ttk.Label(top, text="Opponent").grid(row=0, column=6, sticky="w")
        ttk.Entry(top, textvariable=self.opp, width=14).grid(row=0, column=7, padx=(6,0))

        # row 2
        self.bankroll = tk.DoubleVar(value=float(self.settings.get("bankroll",1000.0)))
        self.stake    = tk.DoubleVar(value=float(self.settings.get("stake",20.0)))
        pool_map = self.settings.get("pool_by_league", DEFAULTS["pool_by_league"])
        self.pool     = tk.IntVar(value=int(pool_map.get(self.league.get(), 40)))
        self.default_odds = tk.DoubleVar(value=float(self.settings.get("default_odds",-110.0)))
        self.dedupe   = tk.BooleanVar(value=bool(self.settings.get("dedupe",True)))
        self.max_per_team = tk.IntVar(value=int(self.settings.get("max_per_team",2)))
        self.min_prop_kinds = tk.IntVar(value=int(self.settings.get("min_prop_kinds",3)))

        ttk.Label(top, text="Bankroll").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Entry(top, textvariable=self.bankroll, width=10).grid(row=1, column=1, padx=(6,18), pady=(8,0))
        ttk.Label(top, text="Stake").grid(row=1, column=2, sticky="w", pady=(8,0))
        ttk.Entry(top, textvariable=self.stake, width=10).grid(row=1, column=3, padx=(6,18), pady=(8,0))
        ttk.Label(top, text="Pool").grid(row=1, column=4, sticky="w", pady=(8,0))
        ttk.Entry(top, textvariable=self.pool, width=10).grid(row=1, column=5, padx=(6,12), pady=(8,0))
        ttk.Label(top, text="Default Odds").grid(row=1, column=6, sticky="w", pady=(8,0))
        ttk.Entry(top, textvariable=self.default_odds, width=12).grid(row=1, column=7, padx=(6,0), pady=(8,0))
        ttk.Checkbutton(top, text="De-dupe", variable=self.dedupe).grid(row=1, column=8, padx=(12,0), pady=(8,0))

        # constraints row
        cons = ttk.Frame(self, padding=(10,0)); cons.pack(fill="x")
        ttk.Label(cons, text="Max per Team").pack(side="left")
        ttk.Entry(cons, textvariable=self.max_per_team, width=6).pack(side="left", padx=8)
        ttk.Label(cons, text="Min distinct PropTypes").pack(side="left", padx=(12,0))
        ttk.Entry(cons, textvariable=self.min_prop_kinds, width=6).pack(side="left", padx=8)

        # sigma editing (simple: click to edit in a popup)
        def edit_sigma_map():
            popup = tk.Toplevel(self); popup.title("Sigma per PropType")
            rows = []
            for i, p in enumerate(PROP_TYPES):
                ttk.Label(popup, text=p, width=22).grid(row=i, column=0, sticky="w")
                v = tk.DoubleVar(value=float(self.settings.get("sigma_by_prop", DEFAULTS["sigma_by_prop"]).get(p, 12.0)))
                e = ttk.Entry(popup, textvariable=v, width=8); e.grid(row=i, column=1, padx=6, pady=2)
                rows.append((p, v))
            def save_sigma():
                smap = self.settings.get("sigma_by_prop", {})
                for p, var in rows:
                    try:
                        smap[p] = float(var.get())
                    except Exception:
                        pass
                self.settings["sigma_by_prop"] = smap
                self.model.sigma_map = smap
                save_settings(self.settings)
                popup.destroy()
            ttk.Button(popup, text="Save", command=save_sigma).grid(row=len(rows), column=0, columnspan=2, pady=8)

        ttk.Button(cons, text="Edit Sigma per PropType", command=edit_sigma_map).pack(side="left", padx=12)

        # buttons
        btns = ttk.Frame(self, padding=(10,8)); btns.pack(fill="x")
        ttk.Button(btns, text="Add from Clipboard", command=self.add_clip).pack(side="left")
        ttk.Button(btns, text="Delete Selected", command=self.delete_selected).pack(side="left", padx=8)
        ttk.Button(btns, text="Clear Slate", command=self.clear_slate).pack(side="left")
        ttk.Button(btns, text="Save Slate", command=self.save_slate).pack(side="left", padx=8)
        ttk.Button(btns, text="Load Slate", command=self.load_slate).pack(side="left")
        ttk.Button(btns, text="Run Engine", command=self.run_engine).pack(side="right")

        # table
        table_frame = ttk.Frame(self, padding=10); table_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table_frame, columns=HEADERS, show="headings", height=16)
        for c in HEADERS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120 if c not in ("Player","PropType") else 160, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set); sb.pack(side="right", fill="y")

        # status
        status = ttk.Frame(self, padding=10); status.pack(fill="x")
        self.status = tk.StringVar(value="Rows: 0"); self.breakdown = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(status, textvariable=self.breakdown, foreground="#555").pack(side="right")

        # react to league change → pool default
        def on_league_change(*_):
            pmap = self.settings.get("pool_by_league", DEFAULTS["pool_by_league"])
            self.pool.set(int(pmap.get(self.league.get(), 40)))
        self.league.trace_add("write", on_league_change)

    def _refresh_table(self):
        # clear
        for i in self.tree.get_children(): self.tree.delete(i)
        # populate
        for idx, row in self.model.df.iterrows():
            self.tree.insert("", "end", iid=str(idx), values=[row.get(h,"") for h in HEADERS])
        self.status.set(f"Rows: {self.model.count()}")
        counts = self.model.counts_by_prop()
        self.breakdown.set(" | ".join(f"{k}:{v}" for k,v in sorted(counts.items())) if counts else "")

    def add_clip(self):
        try:
            self.model.dedupe = self.dedupe.get()
            self.model.default_odds = float(self.default_odds.get())
            self.model.sigma_map = self.settings.get("sigma_by_prop", DEFAULTS["sigma_by_prop"])
            self.model.append_clip(
                self.league.get(), self.proptype.get(),
                team=self.team.get().strip() or None,
                opponent=self.opp.get().strip() or None,
            )
            self._refresh_table()
        except Exception as e:
            messagebox.showerror("Add failed", str(e))

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel: return
        idxs = [int(s) for s in sel]
        self.model.delete_indices(idxs)
        self._refresh_table()

    def clear_slate(self):
        self.model.clear(); self._refresh_table()

    def save_slate(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            initialdir=os.path.join(SLATE_DIR, self.league.get().lower()),
            initialfile=f"sportsline_projections_{self.league.get()}.csv"
        )
        if not path: return
        try:
            self.model.save_to_path(path); messagebox.showinfo("Saved", f"Slate saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def load_slate(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV","*.csv")],
            initialdir=os.path.join(SLATE_DIR, self.league.get().lower())
        )
        if not path: return
        try:
            self.model.load_from_path(path); self._refresh_table()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def run_engine(self):
        if self.model.count() < 6:
            messagebox.showwarning("Not enough props", "Add at least 6 rows before running.")
            return

        league = self.league.get()
        path = CFB_PATH if league == "CFB" else NFL_PATH
        try:
            self.model.save_to_path(path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e)); return

        # persist settings
        self.settings["league"] = league
        self.settings["bankroll"] = float(self.bankroll.get())
        self.settings["stake"] = float(self.stake.get())
        self.settings["default_odds"] = float(self.default_odds.get())
        pmap = self.settings.get("pool_by_league", DEFAULTS["pool_by_league"])
        pmap[league] = int(self.pool.get()); self.settings["pool_by_league"] = pmap
        self.settings["dedupe"] = bool(self.dedupe.get())
        self.settings["max_per_team"] = int(self.max_per_team.get())
        self.settings["min_prop_kinds"] = int(self.min_prop_kinds.get())
        save_settings(self.settings)

        slate_name = f"{league}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cmd = [
            "python", ENGINE_PATH,
            "--csv", path,
            "--bankroll", str(self.settings["bankroll"]),
            "--stake", str(self.settings["stake"]),
            "--pool", str(pmap[league]),
            "--slate", slate_name,
            "--max_per_team", str(self.settings["max_per_team"]),
            "--min_prop_kinds", str(self.settings["min_prop_kinds"]),
        ]
        try:
            subprocess.Popen(cmd, creationflags=0)
            messagebox.showinfo(
                "Engine started",
                f"Running on {league} slate.\nRows: {self.model.count()}\nPool: {pmap[league]}"
                "\nChosen card CSV will be written to logs/."
            )
        except Exception as e:
            messagebox.showerror("Run failed", str(e))

if __name__ == "__main__":
    ensure_dirs()
    App().mainloop()
