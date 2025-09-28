import pandas as pd, engine
def test_smoke_build_board_not_empty():
    df = pd.DataFrame([dict(player_name="T1",pos="WR",team="X",opponent="Y",
        game_date="2025-09-28",kickoff_et="1:00 PM",source="Model",
        stat_name="Receiving Yards",projection=61)])
    out = engine.build_board(df, [engine.MenuItem("60+ Receiving Yards","Receiving Yards",60.0)])
    assert not out.empty
