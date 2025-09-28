# Slate Builder — SportsLine Edition v2.0 (Tier 1 + Tier 2 Upgrades)
# Features:
# - Batch Loader (folder)
# - Header Alias Editor (JSON at data/header_aliases.json)
# - Expanded mapper: Receptions, Rush Attempts, Pass Attempts, Completions
# - Zero/Blank filters; Pick-6 only filter
# - In-app Log Viewer
# - Row EV highlight (Projection > Line baseline)
# - Saved Settings (settings.json) + Save/Load Session (CSV)
# - Right-click context menu (Lock/Exclude/Team/Opp/Copy)
# - Better dedupe (position files > game files)
# - Player Panel
# - Pick-6 Timeline dropdown
# - Headless engine run + log capture; auto Pick-6 column fill

import os, re, math, glob, subprocess, json, sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from datetime import datetime
import pandas as pd
try:
    import pyperclip
except Exception:
    pyperclip = None

BASE   = r"C:\CFB_Engine"
SLATES = os.path.join(BASE, "slates")
ENGINE = os.path.join(BASE, "sports_engine.py")
LOGS   = os.path.join(BASE, "logs")
DATA   = os.path.join(BASE, "data")
CFB_OUT = os.path.join(SLATES, "cfb", "sportsline_projections_CFB.csv")
NFL_OUT = os.path.join(SLATES, "nfl", "sportsline_projections_NFL.csv")
SETTINGS_FILE = os.path.join(DATA, "settings.json")
HEADER_ALIASES = os.path.join(DATA, "header_aliases.json")

HEADERS = [
    "Lock","Exclude","Chosen","Pick6Line","Pick6Odds","LineΔ",
    "Player","PropType","Line","Projection","Odds","MyProb","League","Team","Opponent","Sigma"
]

DEFAULT_ALIASES = {
    "player": ["player","name","player name","full name","PLAYER"],
    "pos":    ["pos","position"],
    "team":   ["team"],
    "opp":    ["opponent","opp"],
    "PASSYD": "Passing Yards",
    "RUSHYD": "Rushing Yards",
    "RECYD":  "Receiving Yards",
    "RUSHTD": "Rush + Rec TDs",
    "RECTD":  "Rush + Rec TDs",
    "TD":     "Passing Touchdowns",
    "INTS":   "Interceptions",
    "RECEPTIONS": "Receptions",
    "REC":       "Receptions",
    "RUSHATT":   "Rushing Attempts",
    "ATT":       "Rushing Attempts",
    "PASSATT":   "Pass Attempts",
    "COMP":      "Completions"
}

def ensure_dirs():
    os.makedirs(os.path.join(SLATES, "cfb"), exist_ok=True)
    os.makedirs(os.path.join(SLATES, "nfl"), exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(HEADER_ALIASES):
        with open(HEADER_ALIASES, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_ALIASES, f, indent=2)

def norm(s:str)->str: return re.sub(r"[^a-z0-9]+","", str(s).lower())
def num(x):
    m = re.search(r"[-+]?\d+(\.\d+)?", str(x))
    return float(m.group()) if m else None

def load_aliases():
    try:
        with open(HEADER_ALIASES, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_ALIASES.copy()

def find_col(raw_cols, candidates):
    # raw_cols is usually a pandas Index; never test it directly for truthiness.
    if raw_cols is None or len(raw_cols) == 0:
        return None
    # Normalize to strings
    normed = [str(c) for c in list(raw_cols)]
    # Map normalized -> original
    m = {norm(c): c for c in normed}

    # Try exact normalized match against the candidate list
    for cand in candidates:
        key = norm(cand)
        if key in m:
            return m[key]

    # No exact match; try a loose contains check on normalized names
    for cand in candidates:
        key = norm(cand)
        for k, v in m.items():
            if key == k:
                return v
    return None


def parse_teams_from_filename(path):
    name = os.path.basename(path)
    for tag in ("NCAAF","NFL"):
        m = re.search(rf"{tag}_([^@]+)@([^_]+)_", name, re.I)
        if m: return m.group(1).upper(), m.group(2).upper()
    return "", ""

def guess_league_from_filename(path):
    return "NFL" if "nfl" in path.lower() else "CFB"

def is_position_file(filename):
    return bool(re.search(r"_(QB|RB|WR|TE|DEF|K)_", os.path.basename(filename), re.I))

def pos_filter(series_pos, ptype):
    if series_pos is None: return None
    pos = series_pos.astype(str).str.upper()
    if ptype in ("Passing Yards","Passing Touchdowns","Completions","Pass Attempts","Interceptions"):
        return pos.str.contains("QB", na=False)
    if ptype in ("Receiving Yards","Receptions"):
        return pos.isin(["WR","TE","RB"])
    if ptype in ("Rushing Yards","Rushing Attempts","Rush + Rec TDs"):
        return pos.isin(["RB","WR","QB","TE"])
    return None

def import_sportsline_csv(path, league_pref, default_odds=-110.0):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if raw.empty:
        return pd.DataFrame(columns=HEADERS), False

    aliases = load_aliases()
    player_col = find_col(raw.columns, aliases.get("player", ["player"]))
    if not player_col:
        raise ValueError(f"Could not find Player column in: {os.path.basename(path)}")

    pos_col  = find_col(raw.columns, aliases.get("pos", ["pos","position"]))
    team_col = find_col(raw.columns, aliases.get("team", ["team"]))
    opp_col  = find_col(raw.columns, aliases.get("opp", ["opponent","opp"]))

    fn_team, fn_opp = parse_teams_from_filename(path)
    league = league_pref or guess_league_from_filename(path) or "CFB"

    # Build stat header → PropType from aliases (all UPPER keys not reserved)
    stat_pairs = []
    for k, v in aliases.items():
        if isinstance(v, str) and k.upper() == k and k not in ("PLAYER","POS","TEAM","OPPONENT","OPP"):
            stat_pairs.append((k, v))

    rows = []
    for header_key, ptype in stat_pairs:
        col = find_col(raw.columns, [header_key])
        if not col: 
            continue
        tmp = pd.DataFrame()
        tmp["Player"]     = raw[player_col].astype(str)
        tmp["PropType"]   = ptype
        tmp["Line"]       = ""  # projections only by default
        tmp["Projection"] = raw[col].apply(num)
        tmp["Odds"]       = default_odds
        tmp["MyProb"]     = ""
        tmp["League"]     = league
        tmp["Team"]       = (raw[team_col].astype(str) if team_col else fn_team)
        tmp["Opponent"]   = (raw[opp_col].astype(str)  if opp_col  else fn_opp)
        tmp["Sigma"]      = ""
        tmp = tmp[pd.to_numeric(tmp["Projection"], errors="coerce").notna()].copy()
        mask = pos_filter(raw[pos_col] if pos_col else None, ptype)
        if mask is not None:
            tmp = tmp[mask.reindex(tmp.index, fill_value=False)]
        if not tmp.empty:
            tmp["_priority"] = 1 if is_position_file(path) else 0
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=HEADERS), is_position_file(path)

    out = pd.concat(rows, ignore_index=True)
    # De-dupe: Player+PropType; prefer position files
    out.sort_values(by=["_priority"], ascending=False, inplace=True)
    out = out.drop_duplicates(subset=["Player","PropType"], keep="first")

    out["Lock"]=False; out["Exclude"]=False; out["Chosen"]=False
    out["Pick6Line"]=""; out["Pick6Odds"]=""; out["LineΔ"]=""
    for h in HEADERS:
        if h not in out.columns:
            out[h] = "" if h not in ("Lock","Exclude","Chosen") else False
    out.drop(columns=[c for c in out.columns if c not in HEADERS], inplace=True, errors="ignore")
    return out[HEADERS].reset_index(drop=True), is_position_file(path)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.title("Slate Builder — SportsLine Edition v2.0")
        self.geometry("1260x760")

        # state
        self.df = pd.DataFrame(columns=HEADERS)
        self.league = tk.StringVar(value="CFB")
        self.default_odds = tk.DoubleVar(value=-110.0)
        self.bankroll = tk.DoubleVar(value=1000.0)
        self.stake = tk.DoubleVar(value=20.0)
        self.pool = tk.IntVar(value=40)
        self.max_per_team = tk.IntVar(value=2)
        self.min_prop_kinds = tk.IntVar(value=3)
        self.block_run = tk.BooleanVar(value=True)

        # filters
        self.hide_zero_proj = tk.BooleanVar(value=False)
        self.hide_blank_line = tk.BooleanVar(value=False)
        self.pick6_only = tk.BooleanVar(value=False)

        # settings + UI
        self.load_settings()
        self.build_top_bar()
        self.build_buttons()
        self.build_filters()
        self.build_main_panes()
        self.build_status()
        self.build_context_menu()
        self.refresh_table()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # UI builders
    def build_top_bar(self):
        top = ttk.Frame(self, padding=10); top.pack(fill="x")
        ttk.Label(top, text="League").grid(row=0,column=0,sticky="w")
        ttk.Combobox(top, textvariable=self.league, values=["CFB","NFL"], width=6, state="readonly").grid(row=0,column=1,padx=(6,18))
        ttk.Label(top, text="Default Odds").grid(row=0,column=2,sticky="w")
        ttk.Entry(top, textvariable=self.default_odds, width=10).grid(row=0,column=3,padx=(6,18))
        ttk.Label(top, text="Bankroll").grid(row=0,column=4,sticky="w")
        ttk.Entry(top, textvariable=self.bankroll, width=10).grid(row=0,column=5,padx=(6,18))
        ttk.Label(top, text="Stake").grid(row=0,column=6,sticky="w")
        ttk.Entry(top, textvariable=self.stake, width=10).grid(row=0,column=7,padx=(6,18))
        ttk.Label(top, text="Pool").grid(row=0,column=8,sticky="w")
        ttk.Entry(top, textvariable=self.pool, width=10).grid(row=0,column=9,padx=(6,18))
        ttk.Checkbutton(top, text="Block Run if Team missing", variable=self.block_run).grid(row=0,column=10)

    def build_buttons(self):
        btns = ttk.Frame(self, padding=(10,8)); btns.pack(fill="x")
        ttk.Button(btns, text="Add CSV file(s)", command=self.add_csvs).pack(side="left")
        ttk.Button(btns, text="Load SportsLine Folder…", command=self.batch_load_folder).pack(side="left", padx=8)
        ttk.Button(btns, text="Header Aliases…", command=self.edit_aliases).pack(side="left", padx=8)
        ttk.Button(btns, text="Clear Slate", command=self.clear_slate).pack(side="left", padx=8)
        ttk.Button(btns, text="Review Unresolved Teams", command=self.review_unresolved).pack(side="left", padx=8)
        ttk.Button(btns, text="Export Engine CSV", command=self.export_engine_csv).pack(side="left", padx=8)
        ttk.Button(btns, text="View Engine Log", command=self.view_engine_log).pack(side="left", padx=8)
        ttk.Button(btns, text="Save Session CSV", command=self.save_session).pack(side="left", padx=8)
        ttk.Button(btns, text="Load Session CSV", command=self.load_session).pack(side="left", padx=8)
        ttk.Button(btns, text="Refresh Pick-6", command=self.manual_pick6_refresh).pack(side="right", padx=(0,8))
        ttk.Button(btns, text="Run Engine", command=self.run_engine).pack(side="right")

    def build_filters(self):
        bar = ttk.Frame(self, padding=(10,0)); bar.pack(fill="x")
        ttk.Label(bar, text="Search/Filter").pack(side="left")
        self.filter_text = tk.StringVar()
        ttk.Entry(bar, textvariable=self.filter_text, width=28).pack(side="left", padx=6)
        ttk.Button(bar, text="Apply", command=self.refresh_table).pack(side="left")
        ttk.Checkbutton(bar, text="Hide Projection = 0", variable=self.hide_zero_proj, command=self.refresh_table).pack(side="left", padx=12)
        ttk.Checkbutton(bar, text="Hide Line blank", variable=self.hide_blank_line, command=self.refresh_table).pack(side="left")
        ttk.Checkbutton(bar, text="Show Pick-6 only", variable=self.pick6_only, command=self.refresh_table).pack(side="left", padx=12)

        ttk.Label(bar, text=" |  Pick-6 Timeline:").pack(side="left", padx=(16,4))
        self.pick6_choice = tk.StringVar(value="(latest)")
        self.pick6_combo = ttk.Combobox(bar, textvariable=self.pick6_choice, width=40, state="readonly")
        self.refresh_pick6_list()
        self.pick6_combo.pack(side="left")
        ttk.Button(bar, text="Compare", command=self.compare_pick6_selected).pack(side="left", padx=(6,0))

    def build_main_panes(self):
        container = ttk.PanedWindow(self, orient="horizontal"); container.pack(fill="both", expand=True, padx=10, pady=8)
        left = ttk.Frame(container)
        self.tree = ttk.Treeview(left, columns=HEADERS, show="headings", height=22)
        for c in HEADERS:
            self.tree.heading(c, text=c)
            w = 70 if c in ("Lock","Exclude","Chosen") else 90 if c in ("Pick6Line","Pick6Odds","LineΔ","Odds","Sigma") else 140 if c in ("Player","PropType") else 100
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set); sb.pack(side="right", fill="y")
        self.tree.tag_configure("chosen_tag", background="#FFF9CC")
        self.tree.tag_configure("ev_plus", background="#E8FFE8")
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Button-3>", self.on_right_click)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_player)
        container.add(left, weight=3)

        right = ttk.Frame(container, padding=8)
        ttk.Label(right, text="Player Panel", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.player_info = tk.Text(right, height=16, width=46, wrap="word")
        self.player_info.pack(fill="both", expand=True, pady=(6,0))
        container.add(right, weight=1)

    def build_status(self):
        status = ttk.Frame(self, padding=10); status.pack(fill="x")
        self.status = tk.StringVar(value="Rows: 0")
        self.breakdown = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(status, textvariable=self.breakdown, foreground="#555").pack(side="right")

    def build_context_menu(self):
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Toggle Lock", command=lambda: self.toggle_flag("Lock"))
        self.menu.add_command(label="Toggle Exclude", command=lambda: self.toggle_flag("Exclude"))
        self.menu.add_separator()
        self.menu.add_command(label="Set Team…", command=self.bulk_set_team)
        self.menu.add_command(label="Set Opp…", command=self.bulk_set_opp)
        self.menu.add_separator()
        self.menu.add_command(label="Copy Player", command=self.copy_player)
        self.menu.add_command(label="Copy Row (CSV)", command=self.copy_row_csv)

    # Data ops
    def add_csvs(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV","*.csv")])
        if not paths: return
        self._import_paths(paths)

    def batch_load_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not paths:
            messagebox.showinfo("Folder empty", "No CSV files found."); return
        self._import_paths(paths)

    def _import_paths(self, paths):
        added = 0
        pos_priority = 0
        for p in paths:
            try:
                part, posfile = import_sportsline_csv(p, self.league.get(), self.default_odds.get())
                if part is None or part.empty: continue
                self.df = pd.concat([self.df, part], ignore_index=True)
                # De-dupe Player+PropType; latest import wins
                self.df = self.df.drop_duplicates(subset=["Player","PropType"], keep="last").reset_index(drop=True)
                added += len(part); pos_priority += int(posfile)
            except Exception as e:
                messagebox.showerror("Import failed", f"{os.path.basename(p)}\n\n{e}")
        self.refresh_table()
        if added:
            messagebox.showinfo("Imported", f"Added {added} rows from {len(paths)} file(s).\n(Position files prioritized: {pos_priority})")

    def clear_slate(self):
        if messagebox.askyesno("Confirm", "Clear all rows?"):
            self.df = pd.DataFrame(columns=HEADERS)
            self.refresh_table()

    def bulk_set_team(self):
        sel = [int(i) for i in self.tree.selection()]
        if not sel: return
        team = simpledialog.askstring("Bulk Set Team", "Team (abbr preferred):", parent=self)
        if not team: return
        for i in sel: self.df.at[i, "Team"] = team.strip().upper()
        self.refresh_table()

    def bulk_set_opp(self):
        sel = [int(i) for i in self.tree.selection()]
        if not sel: return
        opp = simpledialog.askstring("Bulk Set Opp", "Opponent (abbr preferred):", parent=self)
        if not opp: return
        for i in sel: self.df.at[i, "Opponent"] = opp.strip().upper()
        self.refresh_table()

    # Table & Filters
    def filtered_indices(self):
        if self.df.empty: return []
        q = self.filter_text.get().strip().lower()
        idx = []
        for i, r in self.df.iterrows():
            if self.pick6_only.get() and not bool(r.get("Chosen", False)): 
                continue
            if self.hide_zero_proj.get():
                try:
                    if float(r.get("Projection", 0)) == 0.0: 
                        continue
                except Exception: pass
            if self.hide_blank_line.get():
                if str(r.get("Line","")).strip() == "": 
                    continue
            if not q:
                idx.append(i); continue
            if any(q in str(r[c]).lower() for c in HEADERS):
                idx.append(i)
        return idx

    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        show = self.filtered_indices()
        for i in show:
            row = self.df.loc[i]
            vals = [row.get(h,"") for h in HEADERS]
            tags = []
            if bool(row.get("Chosen", False)): tags.append("chosen_tag")
            try:
                proj = float(row.get("Projection", "nan"))
                line = float(row.get("Line", "nan"))
                if not math.isnan(proj) and not math.isnan(line) and proj > line:
                    tags.append("ev_plus")
            except Exception:
                pass
            self.tree.insert("", "end", iid=str(i), values=vals, tags=tuple(tags))
        self.status.set(f"Rows: {len(self.df)}")
        if not self.df.empty:
            counts = self.df["PropType"].value_counts().to_dict()
            unresolved = int((self.df["Team"].astype(str).str.strip()=="").sum())
            parts = []
            if counts: parts.append(" | ".join(f"{k}:{v}" for k,v in sorted(counts.items())))
            parts.append(f"Unresolved teams: {unresolved}")
            self.breakdown.set("   ".join(parts))
        else:
            self.breakdown.set("")

    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        col  = self.tree.identify_column(event.x)
        if not item or not col: return
        i = int(item); j = int(col.replace("#","")) - 1; name = HEADERS[j]
        if name in ("Lock","Exclude","Chosen"):
            self.df.at[i, name] = bool(self.df.at[i, name]) ^ True
            self.refresh_table(); return
        old = str(self.df.at[i, name])
        new = simpledialog.askstring("Edit", f"{name}:", initialvalue=old, parent=self)
        if new is None: return
        self.df.at[i, name] = new; self.refresh_table()

    def on_right_click(self, event):
        try:
            iid = self.tree.identify_row(event.y)
            if iid: self.tree.selection_set(iid)
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def toggle_flag(self, flag):
        for iid in self.tree.selection():
            i = int(iid)
            self.df.at[i, flag] = bool(self.df.at[i, flag]) ^ True
        self.refresh_table()

    def copy_player(self):
        if not pyperclip: return
        sel = self.tree.selection()
        if not sel: return
        name = self.df.loc[int(sel[0]), "Player"]
        try: pyperclip.copy(str(name))
        except Exception: pass

    def copy_row_csv(self):
        if not pyperclip: return
        sel = self.tree.selection()
        if not sel: return
        row = self.df.loc[int(sel[0])][HEADERS]
        csv_line = ",".join(str(row[h]) for h in HEADERS)
        try: pyperclip.copy(csv_line)
        except Exception: pass

    def on_select_player(self, _evt):
        sel = self.tree.selection()
        if not sel or self.df.empty: 
            self.player_info.delete("1.0","end"); return
        i = int(sel[0])
        player = self.df.loc[i, "Player"]
        dfp = self.df[self.df["Player"]==player][["PropType","Line","Projection","Odds","Team","Opponent"]]
        lines = [f"{player} — {len(dfp)} props\n"]
        for _, r in dfp.iterrows():
            lines.append(f"• {r['PropType']}: Line {r['Line']}  Proj {r['Projection']}  Odds {r['Odds']}  [{r['Team']} vs {r['Opponent']}]")
        self.player_info.delete("1.0","end")
        self.player_info.insert("1.0", "\n".join(lines))

    # Validation / Export / Run
    def review_unresolved(self):
        if self.df.empty:
            messagebox.showinfo("Empty", "Add rows first."); return
        idxs = list(self.df[self.df["Team"].astype(str).str.strip()==""].index)
        if not idxs:
            messagebox.showinfo("Teams OK", "All rows have Team set."); return
        win = tk.Toplevel(self); win.title("Resolve Teams"); win.geometry("560x420")
        lb = tk.Listbox(win, selectmode="extended")
        for i in idxs:
            r = self.df.loc[i]
            lb.insert("end", f"{i}: {r['League']} | {r['Player']} | {r['PropType']}")
        lb.pack(fill="both", expand=True, padx=10, pady=10)
        def set_team():
            team = simpledialog.askstring("Set Team", "Team (abbr preferred):", parent=win)
            if not team: return
            sel = list(lb.curselection())
            chosen = [idxs[s] for s in sel] if sel else idxs
            for k in chosen: self.df.at[k, "Team"] = team.strip().upper()
            self.refresh_table()
        ttk.Button(win, text="Set Team for Selected (or All)", command=set_team).pack(pady=6)

    def pre_run_clean(self):
        if self.df.empty:
            raise ValueError("No rows. Add CSVs first.")
        df = self.df.copy()
        if "Exclude" in df.columns:
            excl_bool = df["Exclude"].astype(str).str.strip().str.lower().isin(["true","1","t","yes"])
            df = df.loc[~excl_bool].copy()
        if df.empty:
            raise ValueError("All rows are excluded. Uncheck Exclude on some rows or add more data.")
        engine_cols = ["Player","PropType","Line","Projection","Odds","MyProb","League","Team","Opponent","Sigma"]
        for c in engine_cols:
            if c not in df.columns: df[c] = ""
        df = df[engine_cols].copy()
        for c in ("Line","Projection","Odds","MyProb","Sigma"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[(df["Projection"].notna()) | (df["Line"].notna())].copy()
        issues = []
        if self.block_run.get():
            missing_team = df["Team"].astype(str).str.strip().eq("")
            if missing_team.any():
                issues.append(f"Teams missing: {int(missing_team.sum())}")
        if df["Projection"].isna().all() and df["Line"].isna().all():
            issues.append("No numeric Line or Projection found.")
        if len(df) < 6:
            issues.append("Need at least 6 rows.")
        if issues:
            raise ValueError("Cannot run:\n- " + "\n- ".join(issues))
        snap = os.path.join(LOGS, f"slate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(snap, index=False, encoding="utf-8")
        return df

    def export_engine_csv(self):
        try:
            cleaned = self.pre_run_clean()
        except Exception as e:
            messagebox.showerror("Validation failed", str(e)); return
        path = CFB_OUT if self.league.get()=="CFB" else NFL_OUT
        try:
            cleaned.to_csv(path, index=False, encoding="utf-8")
            messagebox.showinfo("Exported", f"Engine CSV written to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def run_engine(self):
        try:
            cleaned = self.pre_run_clean()
        except Exception as e:
            messagebox.showerror("Validation failed", str(e)); return
        path = CFB_OUT if self.league.get()=="CFB" else NFL_OUT
        try:
            cleaned.to_csv(path, index=False, encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save failed", str(e)); return
        slate_name = f"{self.league.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cmd = [
            sys.executable, ENGINE,
            "--csv", path,
            "--bankroll", str(self.bankroll.get()),
            "--stake", str(self.stake.get()),
            "--pool", str(self.pool.get()),
            "--slate", slate_name,
            "--max_per_team", str(self.max_per_team.get()),
            "--min_prop_kinds", str(self.min_prop_kinds.get()),
        ]
        os.makedirs(LOGS, exist_ok=True)
        log_path = os.path.join(LOGS, f"engine_run_{slate_name}.log")
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=0)
            messagebox.showinfo("Engine started",
                                f"Running on {self.league.get()} slate.\n"
                                f"Rows: {len(cleaned)}  Pool: {self.pool.get()}\n\n"
                                f"Live log: {log_path}")
        except Exception as e:
            messagebox.showerror("Run failed", str(e)); return
        self.after(2000, self.post_run_pick6)

    # Pick-6
    def refresh_pick6_list(self):
        files = sorted(glob.glob(os.path.join(LOGS, "chosen_card_*.csv")), key=os.path.getmtime, reverse=True)
        names = ["(latest)"] + [os.path.basename(f) for f in files]
        self.pick6_combo["values"] = names

    def compare_pick6_selected(self):
        sel = self.pick6_choice.get()
        if sel == "(latest)":
            self.manual_pick6_refresh(); return
        path = os.path.join(LOGS, sel)
        if not os.path.exists(path):
            messagebox.showinfo("Pick-6", "File not found."); return
        try:
            cdf = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            cdf = None
        self.update_pick6_columns(cdf)

    def load_latest_pick6(self):
        files = sorted(glob.glob(os.path.join(LOGS, "chosen_card_*.csv")), key=os.path.getmtime, reverse=True)
        if not files: return None, None
        path = files[0]
        try:
            cdf = pd.read_csv(path, encoding="utf-8-sig"); return path, cdf
        except Exception:
            return path, None

    def update_pick6_columns(self, cdf):
        if cdf is None or cdf.empty or self.df.empty: return
        def key(s): return str(s).strip().lower()
        idx = {}
        for _, r in cdf.iterrows():
            idx.setdefault((key(r.get("Player","")), key(r.get("PropType",""))), []).append(r)
        df = self.df.copy()
        df["Chosen"]=False; df["Pick6Line"]=""; df["Pick6Odds"]=""; df["LineΔ"]=""
        for i, r in df.iterrows():
            k = (key(r.get("Player","")), key(r.get("PropType","")))
            matches = idx.get(k, [])
            if not matches: continue
            pt = str(r.get("PropType","")).lower()
            tol = 0.5 if any(w in pt for w in ["yard","reception","attempt","completion"]) else 0.0
            try: sline = float(r.get("Line", float("nan")))
            except Exception: sline = float("nan")
            chosen_row = None; delta = ""
            for cr in matches:
                try: pline = float(cr.get("Line", float("nan")))
                except Exception: pline = float("nan")
                if math.isnan(sline) or sline == "":
                    chosen_row = cr; break
                if not math.isnan(pline) and abs(pline - sline) <= tol:
                    chosen_row = cr; delta = round(pline - sline, 2); break
            if chosen_row is None:
                cr = matches[0]
                try: pline = float(cr.get("Line", float("nan")))
                except Exception: pline = float("nan")
                delta = "" if math.isnan(sline) or math.isnan(pline) else round(pline - sline, 2)
                chosen_row = cr
            df.at[i,"Chosen"]    = True
            df.at[i,"Pick6Line"] = chosen_row.get("Line","")
            df.at[i,"Pick6Odds"] = chosen_row.get("Odds","")
            df.at[i,"LineΔ"]     = delta
        self.df = df; self.refresh_table(); self.refresh_pick6_list()

    def post_run_pick6(self):
        path, cdf = self.load_latest_pick6()
        if path:
            self.update_pick6_columns(cdf)
            try:
                if pyperclip:
                    txt = open(path, "r", encoding="utf-8").read()
                    pyperclip.copy(txt)
            except Exception:
                pass

    def manual_pick6_refresh(self):
        path, cdf = self.load_latest_pick6()
        if not path:
            messagebox.showinfo("Pick-6", "No chosen_card file found in logs yet."); return
        self.update_pick6_columns(cdf)
        messagebox.showinfo("Pick-6", f"Loaded: {os.path.basename(path)}")

    # Logs / Sessions / Aliases / Settings
    def view_engine_log(self):
        files = sorted(glob.glob(os.path.join(LOGS, "engine_run_*.log")), key=os.path.getmtime, reverse=True)
        if not files:
            messagebox.showinfo("Logs", "No engine logs yet."); return
        path = files[0]
        try:
            txt = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            messagebox.showerror("Logs", str(e)); return
        win = tk.Toplevel(self); win.title(os.path.basename(path)); win.geometry("720x520")
        t = tk.Text(win, wrap="word"); t.pack(fill="both", expand=True); t.insert("1.0", txt)

    def save_session(self):
        if self.df.empty:
            messagebox.showinfo("Save", "Nothing to save."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            self.df.to_csv(path, index=False, encoding="utf-8")
            messagebox.showinfo("Saved", f"Session saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def load_session(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            for h in HEADERS:
                if h not in df.columns:
                    df[h] = "" if h not in ("Lock","Exclude","Chosen") else False
            self.df = df[HEADERS].copy(); self.refresh_table()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def edit_aliases(self):
        try:
            current = json.dumps(load_aliases(), indent=2)
        except Exception:
            current = json.dumps(DEFAULT_ALIASES, indent=2)
        win = tk.Toplevel(self); win.title("Header Aliases (JSON)"); win.geometry("720x560")
        t = tk.Text(win, wrap="none"); t.pack(fill="both", expand=True); t.insert("1.0", current)
        def save_aliases():
            try:
                data = json.loads(t.get("1.0","end"))
                with open(HEADER_ALIASES, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Saved", "Aliases updated. New imports will use them.")
            except Exception as e:
                messagebox.showerror("JSON error", str(e))
        ttk.Button(win, text="Save", command=save_aliases).pack(pady=6)

    def load_settings(self):
        ensure_dirs()
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.league.set(s.get("league","CFB"))
            self.default_odds.set(s.get("default_odds",-110.0))
            self.bankroll.set(s.get("bankroll",1000.0))
            self.stake.set(s.get("stake",20.0))
            self.pool.set(s.get("pool",40))
            self.hide_zero_proj.set(s.get("hide_zero_proj",False))
            self.hide_blank_line.set(s.get("hide_blank_line",False))
            self.pick6_only.set(s.get("pick6_only",False))
        except Exception:
            pass

    def on_close(self):
        s = dict(
            league=self.league.get(),
            default_odds=self.default_odds.get(),
            bankroll=self.bankroll.get(),
            stake=self.stake.get(),
            pool=self.pool.get(),
            hide_zero_proj=self.hide_zero_proj.get(),
            hide_blank_line=self.hide_blank_line.get(),
            pick6_only=self.pick6_only.get(),
        )
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
