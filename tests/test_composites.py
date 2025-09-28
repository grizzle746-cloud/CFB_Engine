import pandas as pd
from engine import build_board

def test_legacy_composite_rush_plus_rec_tds():
    df = pd.DataFrame([
        dict(player_name="Back A", pos="RB", team="X", opponent="Y",
             game_date="2025-09-28", kickoff_et="1:00 PM", source="Model",
             stat_name="Rushing TDs", projection=0.6),
        dict(player_name="Back A", pos="RB", team="X", opponent="Y",
             game_date="2025-09-28", kickoff_et="1:00 PM", source="Model",
             stat_name="Receiving TDs", projection=0.5),
    ])

    menu = [("Rush + Rec TDs", None, 1.0)]  # threshold 1.0

    board = build_board(df, menu)
    assert len(board) >= 1, "board is empty"
    r = board.iloc[0]
    # 0.6 + 0.5 = 1.1
    assert abs(r["Projection"] - 1.1) < 1e-9
    assert abs(r["Edge"] - 0.1) < 1e-9
