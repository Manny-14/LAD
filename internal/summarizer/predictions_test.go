package summarizer

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAnomalies(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "predictions.csv")
	data := "BlockId,prediction,anomaly_score,decision_threshold,score_direction\n" +
		"blk_a,1,0.1,0.5,higher\n" +
		"blk_b,0,0.2,0.5,higher\n" +
		"blk_c,1,-0.2,0.0,lower\n"

	if err := os.WriteFile(csvPath, []byte(data), 0o644); err != nil {
		t.Fatalf("write csv: %v", err)
	}

	anomalies, err := LoadAnomalies(csvPath)
	if err != nil {
		t.Fatalf("LoadAnomalies returned error: %v", err)
	}

	if len(anomalies) != 2 {
		t.Fatalf("expected 2 anomalies, got %d", len(anomalies))
	}

	if anomalies[0].BlockID != "blk_a" || anomalies[1].BlockID != "blk_c" {
		t.Fatalf("unexpected block IDs: %+v", anomalies)
	}

	if anomalies[0].ScoreDirection != "higher" || anomalies[1].ScoreDirection != "lower" {
		t.Fatalf("unexpected score directions: %+v", anomalies)
	}
}

func TestRankAnomalies(t *testing.T) {
	anoms := []Prediction{
		{BlockID: "blk_low", AnomalyScore: -0.5, DecisionThreshold: 0.5, ScoreDirection: "lower"},
		{BlockID: "blk_high", AnomalyScore: 1.7, DecisionThreshold: 0.5, ScoreDirection: "higher"},
		{BlockID: "blk_medium", AnomalyScore: -0.1, DecisionThreshold: 0.5, ScoreDirection: "lower"},
	}

	top := RankAnomalies(anoms, 2)

	if len(top) != 2 {
		t.Fatalf("expected 2 anomalies, got %d", len(top))
	}

	if top[0].BlockID != "blk_high" {
		t.Fatalf("expected blk_high strongest, got %s", top[0].BlockID)
	}

	if top[1].BlockID != "blk_low" {
		t.Fatalf("expected blk_low second, got %s", top[1].BlockID)
	}
}
