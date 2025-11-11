PYTHON ?= python
DATA_CSV ?= data/HDFS.csv
LABELS_CSV ?= data/anomaly_label.csv
STRUCTURED_CSV ?= preprocessed/HDFS.log_structured.csv
TEMPLATES_JSON ?= preprocessed/HDFS_templates.json
EVENT_MAP_JSON ?= preprocessed/event_id_map.json
SEQUENCES_NPZ ?= preprocessed/HDFS_sequences.npz
BLOCK_INDEX_CSV ?= preprocessed/block_id_index.csv
MODEL_PATH ?= ml_models/isolation_forest_model.joblib
VOCAB_JSON ?= ml_models/isolation_forest_event_vocab.json
THRESHOLD_JSON ?= ml_models/isolation_forest_threshold.json
PREDICTIONS_CSV ?= outputs/isolation_forest_predictions.csv
SUMMARY_JSON ?= outputs/pipeline_run_summary.json
TRAIN_FLAGS ?= --skip-grid-search
TRAIN_ALGO ?= isolation-forest

.PHONY: pipeline preprocess sequences train infer summary tests clean

pipeline: preprocess sequences train infer summary tests

preprocess: $(STRUCTURED_CSV) $(EVENT_MAP_JSON)

$(STRUCTURED_CSV) $(EVENT_MAP_JSON): $(DATA_CSV)
	$(PYTHON) scripts/mine_templates.py \
		--input-csv $(DATA_CSV) \
		--output-structured $(STRUCTURED_CSV) \
		--templates-json $(TEMPLATES_JSON) \
		--event-map-json $(EVENT_MAP_JSON)

sequences: $(SEQUENCES_NPZ) $(BLOCK_INDEX_CSV)

$(SEQUENCES_NPZ) $(BLOCK_INDEX_CSV): $(STRUCTURED_CSV) $(EVENT_MAP_JSON)
	$(PYTHON) scripts/build_sequences.py \
		--structured-csv $(STRUCTURED_CSV) \
		--event-map-json $(EVENT_MAP_JSON) \
		--labels-csv $(LABELS_CSV) \
		--output-npz $(SEQUENCES_NPZ) \
		--block-index-csv $(BLOCK_INDEX_CSV)

train: $(MODEL_PATH) $(VOCAB_JSON) $(THRESHOLD_JSON)

$(MODEL_PATH) $(VOCAB_JSON) $(THRESHOLD_JSON): $(SEQUENCES_NPZ)
	$(PYTHON) -m ml_models.isolation_forest_training \
		--data-path $(SEQUENCES_NPZ) \
		--model-output-path $(MODEL_PATH) \
		--vocab-output-path $(VOCAB_JSON) \
		--threshold-output-path $(THRESHOLD_JSON) \
		--algorithm $(TRAIN_ALGO) \
		$(TRAIN_FLAGS)

infer: $(PREDICTIONS_CSV)

$(PREDICTIONS_CSV): $(MODEL_PATH) $(VOCAB_JSON) $(THRESHOLD_JSON) $(SEQUENCES_NPZ) $(BLOCK_INDEX_CSV)
	$(PYTHON) -m ml_models.isolation_forest_inference \
		--data-path $(SEQUENCES_NPZ) \
		--model-path $(MODEL_PATH) \
		--vocab-path $(VOCAB_JSON) \
		--threshold-path $(THRESHOLD_JSON) \
		--block-index-path $(BLOCK_INDEX_CSV) \
		--output-path $(PREDICTIONS_CSV)

summary: $(SUMMARY_JSON)

$(SUMMARY_JSON): $(SEQUENCES_NPZ) $(PREDICTIONS_CSV)
	$(PYTHON) scripts/pipeline_summary.py \
		--sequences $(SEQUENCES_NPZ) \
		--predictions $(PREDICTIONS_CSV) \
		--output $(SUMMARY_JSON)

tests:
	$(PYTHON) -m unittest scripts.test_mine_templates scripts.test_build_sequences ml_models.test_isolation_forest_training

clean:
	rm -f $(STRUCTURED_CSV) $(TEMPLATES_JSON) $(EVENT_MAP_JSON) \
		$(SEQUENCES_NPZ) $(BLOCK_INDEX_CSV) \
		$(MODEL_PATH) $(VOCAB_JSON) $(THRESHOLD_JSON) \
		$(PREDICTIONS_CSV) $(SUMMARY_JSON)
