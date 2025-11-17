package summarizer

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateSummary(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		outputFile string
		expectFile bool
	}{
		{
			name:       "writesToDisk",
			outputFile: "summary.txt",
			expectFile: true,
		},
		{
			name:       "skipsWriteWhenOutputEmpty",
			outputFile: "",
			expectFile: false,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()
			predictions := filepath.Join(dir, "predictions.csv")
			logPath := filepath.Join(dir, "logs.log")
			predCSV := "BlockId,prediction,anomaly_score,decision_threshold,score_direction\nblk_1,1,-0.5,0.5,lower\n"
			logData := "INFO blk_1 Something happened\n"

			if err := os.WriteFile(predictions, []byte(predCSV), 0o644); err != nil {
				t.Fatalf("write predictions: %v", err)
			}
			if err := os.WriteFile(logPath, []byte(logData), 0o644); err != nil {
				t.Fatalf("write logs: %v", err)
			}

			outputPath := ""
			if tc.outputFile != "" {
				outputPath = filepath.Join(dir, tc.outputFile)
			}

			cfg := SummaryConfig{
				PredictionsPath:  predictions,
				LogPath:          logPath,
				OutputPath:       outputPath,
				TopN:             5,
				MaxLinesPerBlock: 10,
			}

			client := StaticClient{Response: "Example summary"}
			ctx := context.Background()

			summary, err := GenerateSummary(ctx, cfg, client)
			if err != nil {
				t.Fatalf("GenerateSummary returned error: %v", err)
			}

			if summary != client.Response {
				t.Fatalf("expected summary %q got %q", client.Response, summary)
			}

			if !tc.expectFile {
				if outputPath != "" {
					if _, err := os.Stat(outputPath); err == nil {
						t.Fatalf("did not expect output file at %s", outputPath)
					}
				}
				return
			}

			written, err := os.ReadFile(outputPath)
			if err != nil {
				t.Fatalf("read output: %v", err)
			}

			if string(written) != client.Response {
				t.Fatalf("output file mismatch: %q", string(written))
			}
		})
	}
}

func TestPreparePrompt(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		csvBody      string
		logsBody     string
		mutateConfig func(cfg *SummaryConfig)
		wantIncludes []string
		wantErr      string
	}{
		{
			name:         "happyPath",
			csvBody:      "BlockId,prediction\nblk_42,1\n",
			logsBody:     "INFO block blk_42 Disk failure\n",
			wantIncludes: []string{"blk_42", "Disk failure"},
		},
		{
			name:     "noAnomalies",
			csvBody:  "BlockId,prediction\nblk_normal,0\n",
			logsBody: "INFO block blk_normal OK\n",
			wantErr:  "no anomalies",
		},
		{
			name: "missingPredictionsPath",
			mutateConfig: func(cfg *SummaryConfig) {
				cfg.PredictionsPath = ""
			},
			wantErr: "predictions path is required",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()
			cfg := SummaryConfig{
				TopN:             1,
				MaxLinesPerBlock: 5,
			}

			if tc.csvBody != "" {
				predictions := filepath.Join(dir, "predictions.csv")
				if err := os.WriteFile(predictions, []byte(tc.csvBody), 0o644); err != nil {
					t.Fatalf("write predictions: %v", err)
				}
				cfg.PredictionsPath = predictions
			}

			if tc.logsBody != "" {
				logPath := filepath.Join(dir, "logs.log")
				if err := os.WriteFile(logPath, []byte(tc.logsBody), 0o644); err != nil {
					t.Fatalf("write logs: %v", err)
				}
				cfg.LogPath = logPath
			}

			if tc.mutateConfig != nil {
				tc.mutateConfig(&cfg)
			}

			prompt, err := PreparePrompt(cfg)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("PreparePrompt returned error: %v", err)
			}

			for _, snippet := range tc.wantIncludes {
				if !strings.Contains(prompt, snippet) {
					t.Fatalf("prompt missing %q: %q", snippet, prompt)
				}
			}
		})
	}
}
