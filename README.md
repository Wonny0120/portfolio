# Prompt Complexity Router — Classifier

Classifies each incoming prompt by complexity so the router can send it to the
cheaper adequate Claude model. The classifier is a small open-source model run
locally through Ollama and prompted few-shot; no training, no labelled training
data, no GPU.

```
easy  ->  Claude Haiku 4.5
hard  ->  Claude Opus 5
```

A task type (`lookup` / `judgment` / `classification` / `summarization`) is also
returned. It is logged for analysis but is not used for routing.

## Results

llama3.2:3b, 200-prompt evaluation set:

| Metric | Value |
|---|---|
| Complexity accuracy | **81.5%** (random baseline 50%) |
| Task accuracy | 62.0% (random baseline 25%) |
| Joint (both correct) | 51.5% |
| Latency | ~13 s per prompt on CPU |

Cost against sending every prompt to Opus:

| Output tokens | Always-Opus | Router | Saving |
|---|---|---|---|
| 100 | $0.5111 | $0.2845 | 44.3% |
| 300 | $1.5111 | $0.8405 | 44.4% |
| 1000 | $5.0111 | $2.7865 | 44.4% |

The saving barely moves across a 10× change in assumed output length, so it does
not depend on that assumption.

## Quality of Response

The classifier is right on 81.5% of prompts, so 18.5% are routed to the wrong
tier. The two directions are not equally costly:

| | Count | Effect |
|---|---|---|
| Under-routed (hard → Haiku) | 24 (12.0%) | Weaker answer than the prompt needed |
| Over-routed (easy → Opus) | 13 (6.5%) | Costs more, answer is fine |

Only under-routing affects quality. 12% of prompts get a cheaper model than they
should — that is the real trade for the 44% cost saving, and it is the number to
drive down rather than the accuracy figure as a whole.

This has not been verified end to end: the router records the tier decision
without calling the model, so no answers were generated to compare. Checking it
properly would mean running the under-routed prompts through both tiers and
comparing the outputs.

## Setup

```bash
ollama pull llama3.2:3b
pip install ollama pydantic pandas
```

## Run

**Route a single prompt:**

```bash
python router.py "Summarize this email thread into a list of decisions"
```

**Interactive:**

```bash
python router.py
```

Every decision is appended to `decisions.jsonl`:

```json
{"ts": "2026-09-09T14:22:31", "prompt": "...", "token_len": 9,
 "has_code": false, "complexity": "hard", "task": "summarization",
 "tier": "claude-opus-5", "latency_s": 12.4, "classifier_failed": false}
```

**Score the classifier:**

```bash
python eval2.py <run_name>
```

Writes `results_<run_name>.csv` and `misclassified_<run_name>.csv`, and appends a
row to `results_log.csv`.

**Cost comparison:**

```bash
python cost.py
```

## Files

| File | What it is |
|---|---|
| `classifier_v3.py` | The classifier prompt and `classify()` |
| `router.py` | Takes a prompt, returns a tier, logs the decision |
| `eval2.py` | Eval harness: accuracy, confusion matrix, share / recall / precision |
| `cost.py` | Router spend vs. always-Opus, plus misroute breakdown |
| `prompts-200-4task-2tier.csv` | Current evaluation set (100 easy / 100 hard, 50 per task) |
| `results_log.csv` | One row per eval run |

## What was tried

Eighteen configurations, all scored on the same evaluation set.

| # | Change | Complexity | Task | Joint |
|---|---|---|---|---|
| 1 | Original prompt, 6 buckets, 3 tiers | 55.5% | 58.0% | 36.0% |
| 2 | Rewritten prompt | 59.5% | 59.5% | 41.0% |
| 3 | Added "medium is the default" rule | 58.0% | 60.5% | 40.5% |
| 4 | Two-stage calls + renamed labels | 59.0% | 50.5% | 34.0% |
| 5 | Binary yes/no cascade | failed | failed | failed |
| 6 | Fixed truncated context (2048 → 4096) | 58.5% | 62.0% | 42.5% |
| 7 | qwen2.5:3b | 57.5% | 60.0% | 33.0% |
| 8 | phi3:mini | 47.0% | 61.0% | 30.5% |
| 9 | Reduced 6 buckets to 4 | 62.5% | 59.5% | 43.0% |
| 10 | Narrowed the "lookup" definition | 60.5% | 56.0% | 37.5% |
| 11 | Reordered tier checks | 57.5% | 57.0% | 38.5% |
| 12 | Added medium examples | 61.5% | 58.5% | 39.0% |
| 13 | Rebalanced examples across tiers | 63.0% | 61.0% | 42.5% |
| 14 | Self-consistency, 5 votes | 62.0% | 57.5% | 42.0% |
| 15 | qwen2.5:3b with the tuned prompt | 48.5% | 61.5% | 29.5% |
| 16 | qwen2.5:3b, default rule removed | 55.5% | 55.5% | 29.5% |
| 17 | Control: 100 clean prompts | 58.0% | 73.0% | 47.0% |
| 18 | **Three tiers folded into two** | **81.5%** | **62.0%** | **51.5%** |


## Design notes

**Two tiers, not three.** The original design had easy / medium / hard. Across
eighteen configurations, `medium` recall never exceeded 52% while easy and hard
both sat near 72% — every change that raised `medium` took the same amount back
out of its neighbours, and complexity accuracy stayed inside a 57–63% band.
Folding `medium` into `hard` took complexity from 63.0% to 81.5%. The random
baseline also rises from 33% to 50%, so the like-for-like gain is smaller than
the headline, but 81.5% is usable for routing where 63% was not.

**Four task buckets, not six.** `factual answer` and `research` were merged into
`lookup`; `reasoning` and `recommendation` into `judgment`. These were the pairs
the model confused most. The reduction gained 4 points on complexity.

**Ambiguity, not just capability.** While the taxonomy still had three tiers, the
same model and prompt were run against two evaluation sets that differed only in
how consistently the labels were applied. Task accuracy rose from 61.0% to 70.5%;
complexity stayed at 63.0% and 62.0%. So the task gap is largely label noise,
while the complexity gap was not which is why complexity only moved once the
taxonomy itself changed. Task accuracy is 62.0% on the current two-tier set, and
a labelling pass is the cheapest way to raise it.

**Failure mode to watch.** The classifier over-predicts `easy`: 55.5% of prompts
are routed to Haiku when only 50% are actually easy. That is why the router saves
44.4% while a perfect classifier would save 40.0% — the extra saving is borrowed
against answer quality. 24 prompts (12%) that needed Opus were sent to Haiku.
Over-routing in the other direction (13 prompts, 6.5%) costs money but not quality.

## Known limitations

- **CPU only.** Ollama runs 100% CPU on the development machine, so a larger model
  would be slower rather than faster. At ~13 s per classification the router
  currently spends more latency than it saves by routing to Haiku.
- **Task accuracy is weak.** `judgment` recall is 16% — most `judgment` prompts are
  predicted as `lookup`. This does not affect routing, which uses complexity only.
- **Output length is assumed, not measured.** The router records the tier decision
  without calling the model, so cost figures use a uniform output length.
- **Quality was not spot-checked end to end.** Whether a Haiku answer is close
  enough to an Opus answer on the under-routed prompts has not been verified.

## Context

Built as the intern-sized version of the full router spec: one provider, tiered
routing, an off-the-shelf classifier, and a measured result. See the project brief
for scope and success criteria.