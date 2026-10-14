# Shipping Cost Watcher

Flags routes whose weekly cost per tonne-km is rising in a way that is not
explained by the context notes, and explains every verdict from the notes
only.

Result on the supplied data (2,940 shipments, 7 routes, 104 weeks):
**17 rows reported, 14 flagged, 3 justified.** Output: `output/flagged_routes.csv`.

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python extract_notes.py        # uses cache/note_facts.json, no LLM call
python aggregate.py            # writes output/flagged_routes.csv
python eval_harness.py         # labeled set + audit of every justified row
python -m pytest -q            # 13 unit tests
python check_extraction.py     # scores note extraction against tests/gold_notes.json
```

`python extract_notes.py --refresh` re-runs the LLM step. It needs Ollama
with the model pulled (`ollama pull qwen2.5:3b`). Not needed to reproduce
the output, because the extraction is committed in `cache/`.

## How it works

1. **Weekly cost.** Shipments are grouped by route and Monday-to-Sunday
   week. Cost per tonne-km = `sum(freight_cost) / sum(tonnes * km)`. This is
   weighted by shipment size, not an average of per-shipment ratios.
2. **Baselines**, exactly as the brief defines them:
   - *Own history:* mean of the route's previous 8 weeks, strictly before
     the current week (`shift(1)` before the rolling mean, so no look-ahead).
     With fewer than 8 prior weeks, all available prior weeks are used. The
     first week of each route has no baseline and cannot be flagged.
   - *Similar routes:* same-week mean over the other routes of the same
     `route_type`, excluding the route itself.
3. **Candidate rule.** A route-week is a candidate when cost is at least
   **5%** above its own history **and** at least **15%** above similar
   routes. Both numbers are my choice (see Decisions).
4. **Note check.** A candidate is *justified* only if a note applies to the
   route (or "All Routes"), states a cost increase, and its date window
   overlaps that week. Otherwise it is *flagged*. A flagged row's reason
   mentions the nearest note (within 14 days) and why it does not qualify.

### Where the LLM is used

`extract_notes.py` reads each of the 10 notes once with a local model
(`qwen2.5:3b` via Ollama) and stores structured facts: end date or
duration, cost effect (increase / none / decrease), stated percentage. The
model proposes; code verifies:

- A percentage is kept only if it literally appears in the note text.
- If the note literally states a "Mon D to Mon D" range starting on the
  note's date, that range overrides the model's end date. (This fixed a real
  error: the model returned the recovery date for N001 instead of the last
  day of disruption.)
- Missing cost effect means "none". Missing end date means start + 6 days.

The result is cached in `cache/note_facts.json`. `aggregate.py` makes no LLM
calls, and its `reason` text is a template filled with the note's own first
sentence, so a reason cannot contain a fact that is not in a note.

Retrieval is a structured filter (route, cost effect, date overlap), not a
vector database. With 10 notes this is exact, deterministic, and cannot
return a wrong-route note. The trade-off is that it would not scale to
thousands of free-text notes, where embedding retrieval would be needed.

## Decisions and trade-offs

- **Thresholds (5% / 15%).** The brief gives no number. The sample row
  Mumbai-Pune 2025-09-15 (+9.2% own, +23.6% peers) is flagged, so the own
  threshold must be low. Counting rows at different settings: about 33 at
  5%/5%, 17 at 5%/15%, 7 at 10%/10%. Mumbai-Pune rises slowly from June 2025
  (peer gap about 21% growing to about 39%) while its rolling own baseline
  keeps chasing it upward, so a strict own-history rule misses the creep.
  Requiring a larger gap against peers removes ordinary noise (for example
  Delhi-Chennai weeks at +6 to +8%). This is a judgment, not a derived
  optimum.
- **Only candidate rows are written**, not all 728 route-weeks, matching
  the sample output (all three sample rows are candidates).
- **Strict date windows.** Chennai-Bangalore weeks of Mar 10 and Mar 17 2025
  are flagged even though costs stayed high after the flood: N001 ends
  Mar 8 and N009 says conditions were normal. A note must cover the week.
- **Explanations are templated, not LLM-written.** Less natural prose, but
  fully grounded and identical on every run.

## How I checked it

- **Reproducibility:** three runs of `aggregate.py`, compared with `fc`: no
  differences (`repro_check.txt`).
- **Labeled set** (`eval_labels.csv`, 10 cases): 10/10 correct. Three are
  the brief's sample rows; the other seven I labeled by reading the notes,
  so they are not independent of my matching logic.
- **Audit of every justified row:** cited note exists, applies to the route,
  states a cost increase, overlaps the week, and is named in the reason.
  3 rows checked, 0 problems.
- **Unit tests (13):** weighted cost, Monday weeks, strictly prior 8-week
  baseline, no look-ahead, peer excludes self and other route types, wrong
  route / wrong window / no-cost-effect notes are never justified, and the
  three sample rows reproduce exactly (cost, own %, peer %).
- **Extraction check:** 40/40 fields on the 10 notes. The date override was
  written after seeing the N001 miss on these same notes, so this shows the
  pipeline handles these notes, not that it generalizes.
- **Do the tests catch bugs?** I broke the matcher on purpose four ways:

| Deliberate bug | Labels + audit | Unit tests |
|---|---|---|
| Ignore route when matching notes | missed | caught |
| Ignore date window | caught | caught |
| Ignore cost effect | caught | caught |
| Include current week in own baseline | missed | caught |

  The labeled set alone is not enough (the real data has no wrong-route
  collision, and labels check verdicts, not numbers). The unit tests cover
  both gaps.

## Cost of one full run

From `cost_log.txt`: 10 LLM calls, 2,383 input tokens, 344 output tokens,
local open-source model, so no API fee (0 INR). Model settings:
temperature 0, seed 42.

## Known limitations

- Re-running `--refresh` was not diffed byte-for-byte; the committed cache
  is what makes the pipeline reproducible. Token counts were identical across
  my two refresh runs (2,383 in / 344 out).
- N003 (diesel, "starting this week", 5-7%) has no end date, so it gets the
  6-day fallback. No candidate row falls in that window, so it never fires.
- The stated magnitude is not compared to the observed rise: a 5-7% note
  could today justify a much larger jump.
- The stretch goal (plain-English Q&A over the findings) is not built.
- 10 labeled cases support "no known failures", not "proven correct".

## Files

| File | Purpose |
|---|---|
| `extract_notes.py` | one-time LLM extraction of note facts, with code guards |
| `cache/note_facts.json` | committed extraction result |
| `aggregate.py` | weekly cost, baselines, flag rule, note matching, CSV output |
| `output/flagged_routes.csv` | the deliverable |
| `eval_harness.py`, `eval_labels.csv` | labeled-set scoring and audit |
| `tests/` | unit tests and gold extraction labels |
| `check_extraction.py` | scores note extraction |
| `repro_check.txt`, `cost_log.txt` | 3-run diff and token log |
