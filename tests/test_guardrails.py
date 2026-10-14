"""Unit tests for the metric definitions and the no-hallucination rules."""
from datetime import date

import pandas as pd
import pytest

import aggregate as agg


# ---------- helpers ----------

def make_ships(routes):
    """routes: {"A-B": ("Short", [cost_per_tonne_km per week, ...])}.
    One 10 t x 100 km shipment per week, on that week's Monday."""
    rows = []
    for route, (route_type, cpts) in routes.items():
        origin, destination = route.split("-")
        for i, cpt in enumerate(cpts):
            rows.append({
                "origin": origin, "destination": destination, "route_type": route_type,
                "quantity_tonnes": 10.0, "distance_km": 100.0,
                "freight_cost_inr": cpt * 1000.0,
                "shipment_date": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i),
            })
    return pd.DataFrame(rows)


def note(note_id, applies_to, start, end, effect="increase", magnitude=None):
    return {"note_id": note_id, "applies_to": applies_to, "date": start, "start": start,
            "end": end, "cost_effect": effect, "magnitude": magnitude,
            "text": f"Text of {note_id}. Second sentence."}


# ---------- metric definitions ----------

def test_cost_is_weighted_not_average_of_ratios():
    ships = pd.DataFrame([
        {"origin": "A", "destination": "B", "route_type": "Short", "quantity_tonnes": 10.0,
         "distance_km": 100.0, "freight_cost_inr": 1000.0, "shipment_date": "2024-01-02"},
        {"origin": "A", "destination": "B", "route_type": "Short", "quantity_tonnes": 30.0,
         "distance_km": 100.0, "freight_cost_inr": 6000.0, "shipment_date": "2024-01-03"},
    ])
    weekly = agg.weekly_costs(ships)
    assert weekly.loc[0, "cost_per_tonne_km"] == pytest.approx(7000 / 4000)  # not 1.5


def test_weeks_run_monday_to_sunday():
    ships = pd.DataFrame([
        {"origin": "A", "destination": "B", "route_type": "Short", "quantity_tonnes": 10.0,
         "distance_km": 100.0, "freight_cost_inr": 1000.0, "shipment_date": d}
        for d in ["2024-01-07", "2024-01-08"]  # Sunday, then next Monday
    ])
    assert list(agg.weekly_costs(ships)["week_of"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-01-08"]


def test_own_baseline_is_strictly_prior_8_weeks():
    weekly = agg.add_baselines(agg.weekly_costs(make_ships({"A-B": ("Short", list(range(1, 11)))})))
    own = weekly["own_baseline"].tolist()
    assert pd.isna(own[0])                      # no prior weeks: no baseline
    assert own[3] == pytest.approx(2.0)         # fewer than 8 prior: mean(1,2,3)
    assert own[9] == pytest.approx(5.5)         # weeks 2..9 only: mean(2..9)


def test_own_baseline_has_no_lookahead():
    base = list(range(1, 11))
    changed = base[:-1] + [100]
    a = agg.add_baselines(agg.weekly_costs(make_ships({"A-B": ("Short", base)})))
    b = agg.add_baselines(agg.weekly_costs(make_ships({"A-B": ("Short", changed)})))
    # changing the LAST week must not move any baseline, including its own
    pd.testing.assert_series_equal(a["own_baseline"], b["own_baseline"])


def test_peer_baseline_excludes_self_and_other_route_types():
    ships = make_ships({
        "A-B": ("Short", [2.0]), "C-D": ("Short", [4.0]), "E-F": ("Short", [6.0]),
        "G-H": ("Long", [100.0]),
    })
    weekly = agg.add_baselines(agg.weekly_costs(ships)).set_index("route")
    assert weekly.loc["A-B", "peer_baseline"] == pytest.approx(5.0)  # mean(4, 6)
    assert weekly.loc["C-D", "peer_baseline"] == pytest.approx(4.0)  # mean(2, 6)


# ---------- hallucination guardrails ----------

NOTES = [
    note("N1", "Chennai-Bangalore", date(2025, 2, 24), date(2025, 3, 8)),
    note("N2", "Delhi-Jaipur", date(2024, 5, 20), date(2024, 5, 26), effect="none"),
    note("N3", "All Routes", date(2025, 5, 5), date(2025, 5, 11), magnitude=7.0),
]


def test_justified_when_route_effect_and_window_all_match():
    note_id, verdict, reason = agg.explain("Chennai-Bangalore", date(2025, 3, 3), NOTES)
    assert (note_id, verdict) == ("N1", "No (justified)")
    assert "N1" in reason


def test_wrong_route_is_not_justified():
    assert agg.explain("Mumbai-Pune", date(2025, 3, 3), NOTES)[:2] == ("", "Yes")


def test_wrong_window_is_not_justified():
    # week of Mar 10 starts after the window ends on Mar 8
    assert agg.explain("Chennai-Bangalore", date(2025, 3, 10), NOTES)[:2] == ("", "Yes")
    # week of Feb 17 ends Feb 23, the day before the window starts
    assert agg.explain("Chennai-Bangalore", date(2025, 2, 17), NOTES)[:2] == ("", "Yes")


def test_note_without_cost_increase_is_not_justified():
    note_id, verdict, reason = agg.explain("Delhi-Jaipur", date(2024, 5, 20), NOTES)
    assert (note_id, verdict) == ("", "Yes")
    assert "Matches note" not in reason


def test_all_routes_note_applies_to_any_route():
    assert agg.explain("Mumbai-Pune", date(2025, 5, 5), NOTES)[:2] == ("N3", "No (justified)")


def test_route_specific_note_beats_all_routes_note():
    both = NOTES + [note("N0", "Mumbai-Pune", date(2025, 5, 5), date(2025, 5, 11))]
    assert agg.explain("Mumbai-Pune", date(2025, 5, 5), both)[0] == "N0"


def test_no_notes_at_all_is_unexplained():
    note_id, verdict, reason = agg.explain("Mumbai-Pune", date(2025, 9, 15), [])
    assert (note_id, verdict) == ("", "Yes")
    assert "No matching note found" in reason


# ---------- ground truth from the brief's sample output ----------

def test_brief_sample_rows_reproduce_exactly():
    weekly = agg.add_baselines(agg.weekly_costs(pd.read_csv(agg.SHIPMENTS_PATH)))
    report = agg.build_report(weekly, agg.load_notes()).set_index(["route", "week_of"])
    expected = {
        ("Delhi-Jaipur", "2024-11-11"): (4.17, "+35.5%", "+21.0%"),
        ("Ahmedabad-Mumbai", "2025-01-20"): (3.29, "+29.5%", "+22.5%"),
        ("Mumbai-Pune", "2025-09-15"): (3.98, "+9.2%", "+23.6%"),
    }
    for key, (cost, own, peer) in expected.items():
        row = report.loc[key]
        assert row["cost_per_tonne_km"] == cost
        assert row["vs_own_history"].startswith(own)
        assert row["vs_similar_routes"].startswith(peer)
