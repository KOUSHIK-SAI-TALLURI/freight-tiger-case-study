"""Extract structured facts from each context note once, with a local LLM.

Results are cached in cache/note_facts.json so later runs read the same
facts and never call the model again (use --refresh to re-extract).

Division of labour:
- start_date  = the note's own date (all notes describe events starting that day)
- routes      = the CSV's applies_to column
- model reads = explicit end date / duration, cost effect, stated % magnitude
- code applies the fallback end (start + 6 days), treats a missing cost
  effect as "none", rejects any magnitude that does not literally appear
  in the note text as a percentage, and overrides the model's end date
  whenever the note literally states a "Mon D to Mon D" range.
"""
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import ollama
import pandas as pd

MODEL = "qwen2.5:3b"
OPTIONS = {"temperature": 0, "seed": 42}
NOTES_PATH = "data/context_notes.csv"
CACHE_PATH = Path("cache/note_facts.json")
EFFECTS = {"increase", "none", "decrease"}

PROMPT = """You extract structured facts from a short logistics note.
Note date: {note_date}
Note: {note}

Return JSON only, with exactly these keys:
- "end_date": YYYY-MM-DD, only if the note names an explicit end date for the
  event; otherwise null.
- "duration_days": the number of days the event lasts, only if the note
  states a duration ("about a week" = 7); otherwise null.
- "cost_effect": "increase" only if the note explicitly says costs rose;
  "decrease" only if it explicitly says costs fell; otherwise "none".
  Notes saying costs were not affected, nothing changed, demand was stable,
  or conditions improved or returned to normal are "none".
- "magnitude_pct_max": the largest percentage rise the note states as a
  number, or null if it states no percentage.
"""

PCT_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*%")

MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"],
        1,
    )
}
RANGE_PATTERN = re.compile(
    r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})"
    r"(?:\s+(?:to|through|until)\s+|\s*[-\u2013]\s*)"
    r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\b"
)


def stated_percentages(note_text):
    """All numbers that appear in the note as part of a percentage."""
    nums = set()
    for match in PCT_PATTERN.findall(note_text):
        nums.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", match))
    return nums


def stated_end_date(note_text, start):
    """End date if the note literally states a 'Mon D to Mon D' range
    that begins on the note's own date, else None."""
    m = RANGE_PATTERN.search(note_text)
    if not m:
        return None
    m1, d1 = m.group(1).lower(), int(m.group(2))
    m2, d2 = m.group(3).lower(), int(m.group(4))
    if m1 not in MONTHS or m2 not in MONTHS:
        return None
    try:
        if date(start.year, MONTHS[m1], d1) != start:
            return None  # range doesn't begin on the note's date; don't trust it
        year = start.year + (1 if MONTHS[m2] < MONTHS[m1] else 0)
        return date(year, MONTHS[m2], d2)
    except ValueError:
        return None  # impossible calendar date


def build_facts(row, raw):
    start = date.fromisoformat(row["date"])

    effect = raw.get("cost_effect")
    if effect is None:
        effect = "none"  # no explicit cost effect stated; only "increase" can justify a row
    if effect not in EFFECTS:
        raise ValueError(f"{row['note_id']}: bad cost_effect in {raw}")

    text_end = stated_end_date(row["note"], start)
    if text_end:
        end = text_end  # explicit range in the text beats the model
    elif raw.get("end_date"):
        end = date.fromisoformat(raw["end_date"])
    elif isinstance(raw.get("duration_days"), int) and raw["duration_days"] > 0:
        end = start + timedelta(days=raw["duration_days"] - 1)
    else:
        end = start + timedelta(days=6)
    if end < start:
        raise ValueError(f"{row['note_id']}: end before start in {raw}")

    mag = raw.get("magnitude_pct_max")
    if mag is not None and float(mag) not in stated_percentages(row["note"]):
        mag = None  # model produced a number the note does not state
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "cost_effect": effect,
        "magnitude_pct_max": mag,
    }


def extract_one(row):
    prompt = PROMPT.format(note_date=row["date"], note=row["note"])
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options=OPTIONS,
    )
    raw = json.loads(resp["message"]["content"])
    return {
        "note_id": row["note_id"],
        "applies_to": row["applies_to"],
        "facts": build_facts(row, raw),
        "raw_model_output": raw,
        "input_tokens": resp["prompt_eval_count"],
        "output_tokens": resp["eval_count"],
    }


def main():
    if CACHE_PATH.exists() and "--refresh" not in sys.argv:
        print(f"{CACHE_PATH} exists - using cached extraction (no LLM calls).")
        return
    CACHE_PATH.parent.mkdir(exist_ok=True)
    notes = pd.read_csv(NOTES_PATH)
    results = [extract_one(row) for _, row in notes.iterrows()]
    CACHE_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {CACHE_PATH}")
    print("LLM calls    :", len(results))
    print("input tokens :", sum(r["input_tokens"] for r in results))
    print("output tokens:", sum(r["output_tokens"] for r in results))


if __name__ == "__main__":
    main()