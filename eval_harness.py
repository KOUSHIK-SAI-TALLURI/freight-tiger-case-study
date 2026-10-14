"""Evaluation harness.

1. Scores the pipeline against eval_labels.csv, a small hand-labeled set
   (expected verdict and note id, or "absent" = must not be reported).
2. Audits EVERY justified row in the real output the way a grader would:
   the cited note must exist, apply to the route, state a cost increase,
   overlap the week, and be named in the reason text. Unexplained rows
   must cite no note.

Exit code is 1 if anything fails.
"""
import sys
from datetime import timedelta

import pandas as pd

import aggregate as agg

LABELS_PATH = "eval_labels.csv"


def run_pipeline():
    weekly = agg.add_baselines(agg.weekly_costs(pd.read_csv(agg.SHIPMENTS_PATH)))
    notes = agg.load_notes()
    return agg.build_report(weekly, notes), notes


def score_labels(report):
    labels = pd.read_csv(LABELS_PATH, keep_default_na=False)
    failures = []
    for lab in labels.itertuples():
        match = report[(report["route"] == lab.route) & (report["week_of"] == lab.week_of)]
        if match.empty:
            actual, actual_note = "absent", ""
        else:
            actual = match.iloc[0]["flagged"]
            actual_note = match.iloc[0]["matched_note_id"]
        ok = actual == lab.expected_flagged and actual_note == lab.expected_note_id
        print(f'{"PASS" if ok else "FAIL"}  {lab.route} {lab.week_of}: '
              f'expected {lab.expected_flagged}/{lab.expected_note_id or "-"}, '
              f'got {actual}/{actual_note or "-"}')
        if not ok:
            failures.append((lab.route, lab.week_of))
    print(f"\nlabeled set: {len(labels) - len(failures)}/{len(labels)} correct")
    return failures


def audit_rows(report, notes):
    by_id = {n["note_id"]: n for n in notes}
    problems = []
    for r in report.itertuples():
        week_start = pd.Timestamp(r.week_of).date()
        week_end = week_start + timedelta(days=6)
        if r.flagged == "Yes":
            if r.matched_note_id != "" or "Matches note" in r.reason:
                problems.append((r.route, r.week_of, "flagged row cites a note"))
            continue
        note = by_id.get(r.matched_note_id)
        if note is None:
            problems.append((r.route, r.week_of, "cited note does not exist"))
            continue
        if note["applies_to"] not in (r.route, agg.ALL_ROUTES):
            problems.append((r.route, r.week_of, "note is for a different route"))
        if note["cost_effect"] != "increase":
            problems.append((r.route, r.week_of, "note does not state a cost increase"))
        if not (note["start"] <= week_end and note["end"] >= week_start):
            problems.append((r.route, r.week_of, "note window misses this week"))
        if r.matched_note_id not in r.reason:
            problems.append((r.route, r.week_of, "note id missing from reason"))
    n_just = int((report["flagged"] == "No (justified)").sum())
    print(f"audit: {n_just} justified rows checked, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM", p)
    return problems


def main():
    report, notes = run_pipeline()
    failures = score_labels(report)
    problems = audit_rows(report, notes)
    sys.exit(1 if failures or problems else 0)


if __name__ == "__main__":
    main()
