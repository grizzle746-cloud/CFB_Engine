import sys, pandas as pd
if len(sys.argv) < 2:
    print("Usage: python tools/statnames.py <projections.csv>")
    raise SystemExit(2)
df = pd.read_csv(sys.argv[1])
names = sorted(df["stat_name"].astype(str).str.strip().unique())
print("Distinct stat_name values ({}):".format(len(names)))
for s in names:
    print("-", s)
