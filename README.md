# LAD

## Milestone 4 · LLM-Driven Anomaly Summarization & Pipeline Completion

This milestone closes the loop for the LAD (Log Anomaly Detection) project. The Python
pipeline that mines templates, builds sequences, and runs Isolation Forest inference is
now paired with a Go-based summarization layer that translates raw anomaly scores into
plain-English incident briefs using Google’s Gemini 2.5 Flash model.

### End-to-End Workflow

1. **Parse & Template Mine** – `scripts/mine_templates.py` converts raw HDFS logs into
	structured templates and an event map.
2. **Sequence Building** – `scripts/build_sequences.py` builds block-level sequences and
	labels.
3. **Infer** – `ml_models/isolation_forest_training.py` + `ml_models/isolation_forest_inference.py`
	produce calibrated scores (`outputs/isolation_forest_predictions.csv`).
4. **Summarize** – `cmd/summarize` consumes the predictions, fetches matching log lines,
	constructs a prompt, and calls Gemini to create `outputs/anomaly_summary.txt`.

Run everything with:

```bash
make pipeline
```

Then summarize:

```bash
make summarize
```

Use a dry-run to inspect the LLM prompt without spending tokens:

```bash
make summarize LLM_FLAGS="--dry-run --max-lines 20"
```

Set your key once per session:

```bash
export GEMINI_API_KEY='YOUR_KEY'
```

or inline for a single invocation:

```bash
GEMINI_API_KEY='YOUR_KEY' make summarize
```