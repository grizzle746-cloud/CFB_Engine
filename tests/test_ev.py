from engine import _ticket_ev
def test_six_leg_all_hit_pays_25x():
    ev,_ = _ticket_ev([1]*6, [0]*6, {6:{6:25.0,5:1.25}}, stake=5.0, entry_sizes=[6])
    assert abs(ev - 25.0*5.0) < 1e-9
def test_six_leg_one_miss_pays_1p25x():
    ev,_ = _ticket_ev([1,1,1,1,1,0], [0]*6, {6:{6:25.0,5:1.25}}, stake=5.0, entry_sizes=[6])
    assert abs(ev - 1.25*5.0) < 1e-9
