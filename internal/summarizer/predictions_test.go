package summarizer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadAnomalies(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		csvData  string
		wantIDs  []string
		wantDirs []string
		wantErr  string
	}{
		{
			name: "withOptionalColumns",
			csvData: "BlockId,prediction,anomaly_score,decision_threshold,score_direction\n" +
				"blk_a,1,0.1,0.5,higher\n" +
				"blk_b,0,0.2,0.5,higher\n" +
				"blk_c,1,-0.2,0.0,lower\n",
			wantIDs:  []string{"blk_a", "blk_c"},
			wantDirs: []string{"higher", "lower"},
		},
		{
			name: "missingPredictionColumn",
			csvData: "BlockId,anomaly_score\n" +
				"blk_z,1\n",
			wantErr: "missing required column",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()
			csvPath := filepath.Join(dir, "predictions.csv")
			if err := os.WriteFile(csvPath, []byte(tc.csvData), 0o644); err != nil {
				t.Fatalf("write csv: %v", err)
			}

			anomalies, err := LoadAnomalies(csvPath)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("LoadAnomalies returned error: %v", err)
			}

			if len(anomalies) != len(tc.wantIDs) {
				t.Fatalf("expected %d anomalies, got %d", len(tc.wantIDs), len(anomalies))
			}

			for i, id := range tc.wantIDs {
				if anomalies[i].BlockID != id {
					t.Fatalf("unexpected block ID at %d: want %s got %s", i, id, anomalies[i].BlockID)
				}
			}

			for i, dir := range tc.wantDirs {
				if anomalies[i].ScoreDirection != dir {
					t.Fatalf("unexpected score direction at %d: want %s got %s", i, dir, anomalies[i].ScoreDirection)
				}
			}
		})
	}
}

func TestRankAnomalies(t *testing.T) {
	t.Parallel()

	base := []Prediction{
		{BlockID: "blk_low", AnomalyScore: -0.5, DecisionThreshold: 0.5, ScoreDirection: "lower"},
		{BlockID: "blk_high", AnomalyScore: 1.7, DecisionThreshold: 0.5, ScoreDirection: "higher"},
		{BlockID: "blk_medium", AnomalyScore: -0.1, DecisionThreshold: 0.5, ScoreDirection: "lower"},
	}

	cases := []struct {
		name string
		topN int
		want []string
	}{
		{
			name: "topTwo",
			topN: 2,
			want: []string{"blk_high", "blk_low"},
		},
		{
			name: "keepAllWhenTopZero",
			topN: 0,
			want: []string{"blk_high", "blk_low", "blk_medium"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Copy slice to avoid sharing mutations across tests.
			anoms := append([]Prediction(nil), base...)
			ranked := RankAnomalies(anoms, tc.topN)

			if len(ranked) != len(tc.want) {
				t.Fatalf("expected %d anomalies, got %d", len(tc.want), len(ranked))
			}

			for i, id := range tc.want {
				if ranked[i].BlockID != id {
					t.Fatalf("unexpected block order at %d: want %s got %s", i, id, ranked[i].BlockID)
				}
			}
		})
	}
}
