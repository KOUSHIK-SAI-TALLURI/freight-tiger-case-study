"""Weekly cost-per-tonne-km monitor.

Pipeline:
1. Group shipments by route and Monday-based week; compute weighted cost
   per tonne-km = sum(cost) / sum(tonnes * km).
2. Baselines, exactly as the brief defines them:
   - own history : mean of the route's previous 8 weeks (strictly prior,
                   all prior weeks if fewer than 8, nothing if none).
   - similar     : same-week mean over the OTHER routes of the same
                   route_type (the route itself is excluded).
3. Candidate rows = cost is above BOTH baselines by the thresholds below.
4. Each candidate is checked against the cached note facts. It is
   "justified" only if a note applies to the route (or "All Routes"),
   states a cost increase, and its date window overlaps the week.
   Otherwise it is flagged. No LLM is called here, so output is
   deterministic.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SHIPMENTS_PATH = "data/shipment_records.csv"
NOTES_PATH = "data/context_notes.csv"
FACTS_PATH = Path("cache/note_facts.json")
OUTPUT_PATH = Path("output/flagged_routes.csv")

OWN_WINDOW_WEEKS = 8
OWN_RISE_PCT = 5.0    # must be at least this far above own history ...
PEER_RISE_PCT = 15.0  # ... AND at least this far above similar routes
NEAR_NOTE_DAYS = 14   # a non-qualifying note this close is mentioned
ALL_ROUTES = "All Routes"

OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason",
]


# ---------- step 1 and 2: weekly cost and baselines ----------

def weekly_costs(ships):
    ships = ships.copy()
    ships["route"] = ships["origin"] + "-" + ships["destination"]
    ships["shipment_date"] = pd.to_datetime(ships["shipment_date"])
    ships["week_of"] = ships["shipment_date"] - pd.to_timedelta(
        ships["shipment_date"].dt.weekday, unit="D"
    )
    ships["tonne_km"] = ships["quantity_tonnes"] * ships["distance_km"]
    weekly = ships.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        total_cost=("freight_cost_inr", "sum"),
        tonne_km=("tonne_km", "sum"),
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["tonne_km"]
    return weekly.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(weekly):
    weekly = weekly.copy()
    # shift(1) first, so the current week never enters its own baseline
    weekly["own_baseline"] = weekly.groupby("route")["cost_per_tonne_km"].transform(
        lambda s: s.shift(1).rolling(OWN_WINDOW_WEEKS, min_periods=1).mean()
    )
    same_week = weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    peer_sum = same_week.transform("sum")
    peer_count = same_week.transform("count")
    weekly["peer_baseline"] = (peer_sum - weekly["cost_per_tonne_km"]) / (peer_count - 1)
    weekly["vs_own_pct"] = (weekly["cost_per_tonne_km"] / weekly["own_baseline"] - 1) * 100
    weekly["vs_peer_pct"] = (weekly["cost_per_tonne_km"] / weekly["peer_baseline"] - 1) * 100
    return weekly


# ---------- step 3: note matching ----------

def load_notes():
    texts = pd.read_csv(NOTES_PATH).set_index("note_id")["note"]
    notes = []
    for item in json.loads(FACTS_PATH.read_text()):
        facts = item["facts"]
        notes.append({
            "note_id": item["note_id"],
            "applies_to": item["applies_to"],
            "date": date.fromisoformat(facts["start_date"]),
            "start": date.fromisoformat(facts["start_date"]),
            "end": date.fromisoformat(facts["end_date"]),
            "cost_effect": facts["cost_effect"],
            "magnitude": facts["magnitude_pct_max"],
            "text": texts[item["note_id"]],
        })
    return notes


def applies_to_route(note, route):
    return note["applies_to"] in (route, ALL_ROUTES)


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s", text.strip())[0]


def find_justification(route, week_start, notes):
    """A note that applies to the route, says costs rose, and overlaps the week."""
    week_end = week_start + timedelta(days=6)
    hits = [
        n for n in notes
        if applies_to_route(n, route)
        and n["cost_effect"] == "increase"
        and n["start"] <= week_end
        and n["end"] >= week_start
    ]
    # route-specific note beats an "All Routes" note; then note id for a stable tie-break
    hits.sort(key=lambda n: (n["applies_to"] == ALL_ROUTES, n["note_id"]))
    return hits[0] if hits else None


def closest_note(route, week_start, notes):
    near = [
        (abs((n["date"] - week_start).days), n["note_id"], n)
        for n in notes if applies_to_route(n, route)
    ]
    near = [x for x in near if x[0] <= NEAR_NOTE_DAYS]
    return min(near, key=lambda x: (x[0], x[1]))[2] if near else None


def explain(route, week_start, notes):
    """Return (matched_note_id, verdict, reason)."""
    hit = find_justification(route, week_start, notes)
    if hit:
        reason = (
            f'Matches note {hit["note_id"]} dated {hit["date"]}: '
            f'"{first_sentence(hit["text"])}" The note covers {hit["start"]} to '
            f'{hit["end"]}, which overlaps this week, so the cost rise has a '
            "documented explanation."
        )
        if hit["magnitude"] is not None:
            reason += f' The note states a rise of up to {hit["magnitude"]:g}%.'
        return hit["note_id"], "No (justified)", reason

    near = closest_note(route, week_start, notes)
    if near is None:
        return "", "Yes", (
            "No matching note found for this route or date range. "
            "Cost rise looks unexplained and worth a human review."
        )
    if near["cost_effect"] == "increase":
        why = (
            f'describes a cost increase, but for {near["start"]} to {near["end"]}, '
            "which does not cover this week."
        )
    else:
        why = (
            f'says "{first_sentence(near["text"])}" -- it does not describe a '
            "reason for a cost rise on this route."
        )
    return "", "Yes", (
        f'The closest note ({near["note_id"]}, {near["date"]}) {why} '
        "No genuine justification found; flagged for review."
    )


# ---------- step 4: report ----------

def build_report(weekly, notes):
    candidates = weekly[
        (weekly["vs_own_pct"] >= OWN_RISE_PCT) & (weekly["vs_peer_pct"] >= PEER_RISE_PCT)
    ]
    rows = []
    for r in candidates.itertuples():
        note_id, verdict, reason = explain(r.route, r.week_of.date(), notes)
        rows.append({
            "route": r.route,
            "week_of": r.week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(r.cost_per_tonne_km, 2),
            "vs_own_history": f"{r.vs_own_pct:+.1f}% vs this route's past average",
            "vs_similar_routes": f"{r.vs_peer_pct:+.1f}% vs similar-length routes this week",
            "flagged": verdict,
            "matched_note_id": note_id,
            "reason": reason,
        })
    report = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return report.sort_values(["week_of", "route"]).reset_index(drop=True)


def main():
    weekly = add_baselines(weekly_costs(pd.read_csv(SHIPMENTS_PATH)))
    report = build_report(weekly, load_notes())
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}: {len(report)} rows")
    print(report["flagged"].value_counts().to_string())


if __name__ == "__main__":
    main()