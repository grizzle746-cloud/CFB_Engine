import pandas as pd
from engine import build_board

def test_receiving_yards_edge():
    df = pd.DataFrame([dict(
        player_name="WR B", pos="WR", team="X", opponent="Y",
        game_date="2025-09-28", kickoff_et="1:00 PM", source="Model",
        stat_name="Receiving Yards", projection=62.0
    )])
    menu = [("60+ Receiving Yards", "Receiving Yards", 60.0)]
    out = build_board(df, menu)
    assert len(out) >= 1, "board is empty"
    assert abs(out.iloc[0]["Edge"] - 2.0) < 1e-9
