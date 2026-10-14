"""Score the cached LLM extraction against the hand-labeled gold set."""
import json
from pathlib import Path

cache = json.loads(Path("cache/note_facts.json").read_text())
gold = json.loads(Path("tests/gold_notes.json").read_text())

wrong = 0
for item in cache:
    nid = item["note_id"]
    for field, want in gold[nid].items():
        got = item["facts"][field]
        if got != want:
            wrong += 1
            print(f"MISMATCH {nid}.{field}: got {got!r}, expected {want!r}")

total = sum(len(g) for g in gold.values())
print(f"\n{total - wrong}/{total} fields correct across {len(gold)} notes")