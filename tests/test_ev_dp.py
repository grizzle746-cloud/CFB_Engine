from engine import _ticket_ev

def test_six_leg_all_hit_pays_25x():
    win, void = [1]*6, [0]*6
    mult = {6: {6: 25.0, 5: 1.25}}
    ev, dist = _ticket_ev(win, void, mult, stake=5.0, entry_sizes=[6])
    assert abs(ev - 25.0*5.0) < 1e-9
    assert abs(dist[6][6] - 1.0) < 1e-12

def test_six_leg_exactly_one_miss_pays_1p25x():
    win, void = [1,1,1,1,1,0], [0]*6
    mult = {6: {6: 25.0, 5: 1.25}}
    ev, _ = _ticket_ev(win, void, mult, stake=5.0, entry_sizes=[6])
    assert abs(ev - 1.25*5.0) < 1e-9
