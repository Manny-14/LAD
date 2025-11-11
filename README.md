# LAD

## End-to-end log anomaly pipeline

Milestone 3 wires the entire flow — from raw logs to Isolation Forest predictions — into
reproducible scripts. A new `Makefile` now orchestrates the steps and captures a lightweight
summary for every run.

### Quick start

```bash
make pipeline
```

This target performs the full sequence:

1. **Template mining** – `scripts/mine_templates.py` assigns stable `EventId`s.
2. **Sequence building** – `scripts/build_sequences.py` produces `preprocessed/HDFS_sequences.npz`.
3. **Training** – `ml_models/isolation_forest_training.py` fits the (default) unsupervised
	 Isolation Forest with the tuned hyperparameters discovered during milestone analysis,
	 calibrates a recall-aware decision threshold, and writes
	 `isolation_forest_model.joblib`, `isolation_forest_event_vocab.json`, and
	 `isolation_forest_threshold.json` (with algorithm metadata). A supervised Random Forest
	 baseline remains available but now requires an explicit opt-in.
4. **Inference** – `ml_models/isolation_forest_inference.py` applies the stored threshold and metadata
	 when generating `outputs/isolation_forest_predictions.csv` (now including
	 `decision_threshold`, `score_direction`, and `algorithm` columns for traceability).
5. **Summary + validation** – `scripts/pipeline_summary.py` compares predictions against labels,
	 stores results in `outputs/pipeline_run_summary.json`, and unit tests are re-run to guard the
	 preprocessing steps.

All intermediate and final artifacts live under `preprocessed/`, `ml_models/`, and `outputs/`.

### Customising runs

- Set `TRAIN_FLAGS=` to remove `--skip-grid-search` and perform the full tuning sweep:

	```bash
	make train TRAIN_FLAGS=
	```

- Switch algorithms via `TRAIN_ALGO`. To try the supervised Random Forest baseline (which will drive
	 metrics close to 1.0), also pass `--allow-supervised` through `TRAIN_FLAGS` to acknowledge the
	 trade-off:

	```bash
	make pipeline TRAIN_ALGO=random-forest TRAIN_FLAGS="--allow-supervised --skip-grid-search"
	```

- Keep `TRAIN_ALGO=isolation-forest` (the default) to remain in the unsupervised regime:

	```bash
	make pipeline TRAIN_ALGO=isolation-forest
	```

- The calibrated threshold lives at `ml_models/isolation_forest_threshold.json`. Adjust or inspect it
	to explore different precision/recall trade-offs without retraining. The training CLI now exposes
	`--calibration-beta` (defaults to 1.0) to tilt the threshold towards recall (use values > 1) or
	precision (values < 1), and `--min-calibration-precision` to enforce a floor on precision when
	selecting the threshold.

- Use alternate datasets by overriding `DATA_CSV`/`LABELS_CSV` when invoking make:

	```bash
	make pipeline DATA_CSV=data/HDFS_2k.csv LABELS_CSV=data/anomaly_label.csv
	```

- To favour recall (e.g., for incident detection), tilt the calibration objective:

	```bash
	make pipeline TRAIN_FLAGS="--skip-grid-search --calibration-beta 2 --min-calibration-precision 0.75"
	```

	The `beta` value emphasises recall when greater than 1 while the optional precision floor keeps
	alert volume in check.

### Validation summary

The summary JSON captures basic health checks (prediction counts, anomaly totals, precision/recall
when labels are available) and highlights the top anomalous blocks, recording whether higher or
lower scores indicate anomalies. Inspect it after each run:

```bash
cat outputs/pipeline_run_summary.json | jq
```

### Related commands

| Command | Purpose |
| --- | --- |
| `make preprocess` | Run only template mining |
| `make sequences` | Rebuild sequences and block index |
| `make train` | Fit the selected model (Random Forest or Isolation Forest) |
| `make infer` | Regenerate predictions using the saved model + calibrated threshold |
| `make summary` | Refresh the JSON metrics report |
| `make tests` | Execute preprocessing + model smoke tests |
| `make clean` | Remove generated artifacts |

With these targets in place, the repository now ships a reproducible, script-first anomaly
detection pipeline ready for Milestone 4 (LLM summarisation + deployment).