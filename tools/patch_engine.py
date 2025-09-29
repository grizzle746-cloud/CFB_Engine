import re, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
engine_path = ROOT / "engine.py"
composites_path = ROOT / "composites" / "pick6_composites.json"

# --- backup ---
engine_bak = engine_path.with_suffix(".py.bak")
if engine_path.exists():
    engine_bak.write_text(engine_path.read_text(encoding="utf-8"), encoding="utf-8")

src = engine_path.read_text(encoding="utf-8")

# --- ensure numpy import ---
if "import numpy as np" not in src:
    src = re.sub(r"(import pandas as pd\s*\n)", r"\1import numpy as np\n", src, count=1)

# --- canonical composites function ---
func_text = '''
def _evaluate_composites_inplace(wide: pd.DataFrame, composites: List[dict], verbose: bool = False) -> None:
    """
    For each composite:
      - Build local variables as Series from wide[stat_name]
      - Apply missing policy
      - Safely eval arithmetic expression using a tiny whitelist:
        +, -, *, /, parentheses
        min(a,b), max(a,b)   # two-arg versions
        clamp(x, lo, hi)     # clip Series to [lo, hi]
      - Write result into wide[name]
    """
    if not composites or wide.empty:
        return

    def s_min(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return pd.concat([a, b], axis=1).min(axis=1)

    def s_max(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return pd.concat([a, b], axis=1).max(axis=1)

    def clamp(x: pd.Series, lo: float, hi: float) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        return x.clip(lower=float(lo), upper=float(hi))

    safe_funcs = {"min": s_min, "max": s_max, "clamp": clamp}

    for comp in composites:
        name = comp["name"]
        expr = comp["expr"]
        vars_map: Dict[str, str] = comp["vars"]
        policy = comp["missing_policy"]

        # Build locals dict {var: Series}
        local_vars: Dict[str, pd.Series] = {}
        for var, stat_col in vars_map.items():
            col = str(stat_col).strip()
            if col in wide.columns:
                s = wide[col]
            else:
                # column missing -> all NaN (float-safe)
                s = pd.Series(np.nan, index=wide.index, dtype="float64")

            # normalize to numeric and apply missing policy
            s = pd.to_numeric(s, errors="coerce")
            if policy == "zero":
                s = s.fillna(0.0)
            # else: keep NaNs to propagate

            local_vars[var] = s

        # Safe eval: only allow our functions and locals; no builtins.
        try:
            result = eval(expr, {"__builtins__": {}, **safe_funcs}, local_vars)  # type: ignore[eval-used]
        except Exception as e:
            raise SystemExit(f"Composite '{name}' failed to evaluate: {e}")

        result = pd.to_numeric(result, errors="coerce")
        wide[name] = result
        if verbose:
            print(f"[composite] computed '{name}' from expr='{expr}' (policy={policy})")
'''.lstrip("\n")

# replace existing function block (from its def to next top-level def or EOF)
pat = re.compile(r'(?ms)^def\s+_evaluate_composites_inplace\s*\(.*?\):\n.*?(?=^\s*def\s|\Z)')
if pat.search(src):
    src = pat.sub(func_text, src, count=1)
else:
    # if not found, just append (shouldn't happen, but safe)
    src += "\n\n" + func_text

# also replace any old placeholder pd.NA->np.nan if present elsewhere
src = src.replace('pd.Series([pd.NA] * len(wide), index=wide.index, dtype="float64")',
                  'pd.Series(np.nan, index=wide.index, dtype="float64")')

engine_path.write_text(src, encoding="utf-8")

# --- fix composites JSON: avoid 'pass' keyword var ---
if composites_path.exists():
    try:
        data = json.loads(composites_path.read_text(encoding="utf-8-sig").lstrip("\ufeff"))
        changed = False
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            expr = item.get("expr", "")
            vars_map = item.get("vars", {})
            if not isinstance(vars_map, dict):
                continue
            # heuristic: if expr uses 'pass + rush' or vars has key 'pass'
            if name == "Pass + Rush Yards" or "pass + rush" in expr or "pass" in vars_map:
                # rename var
                if "pass" in vars_map:
                    vars_map["pass_"] = vars_map.pop("pass")
                # ensure expr uses pass_
                item["expr"] = expr.replace("pass +", "pass_ +")
                item["vars"] = vars_map
                changed = True
        if changed:
            composites_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warn] composites JSON not updated: {e}")

print("OK: engine.py patched and composites checked. Backup at:", engine_bak)
