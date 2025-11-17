package summarizer

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateSummaryWritesOutput(t *testing.T) {
	dir := t.TempDir()
	predictions := filepath.Join(dir, "predictions.csv")
	logPath := filepath.Join(dir, "logs.log")
	output := filepath.Join(dir, "summary.txt")

	predCSV := "BlockId,prediction,anomaly_score,decision_threshold,score_direction\nblk_1,1,-0.5,0.5,lower\n"
	logData := "INFO blk_1 Something happened\n"

	if err := os.WriteFile(predictions, []byte(predCSV), 0o644); err != nil {
		t.Fatalf("write predictions: %v", err)
	}
	if err := os.WriteFile(logPath, []byte(logData), 0o644); err != nil {
		t.Fatalf("write logs: %v", err)
	}

	cfg := SummaryConfig{
		PredictionsPath:  predictions,
		LogPath:          logPath,
		OutputPath:       output,
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

	written, err := os.ReadFile(output)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	if string(written) != client.Response {
		t.Fatalf("output file mismatch: %q", string(written))
	}
}

func TestPreparePrompt(t *testing.T) {
	dir := t.TempDir()
	predictions := filepath.Join(dir, "predictions.csv")
	logPath := filepath.Join(dir, "logs.log")

	predCSV := "BlockId,prediction\nblk_42,1\n"
	logData := "INFO block blk_42 Disk failure\n"

	if err := os.WriteFile(predictions, []byte(predCSV), 0o644); err != nil {
		t.Fatalf("write predictions: %v", err)
	}
	if err := os.WriteFile(logPath, []byte(logData), 0o644); err != nil {
		t.Fatalf("write logs: %v", err)
	}

	cfg := SummaryConfig{
		PredictionsPath:  predictions,
		LogPath:          logPath,
		TopN:             1,
		MaxLinesPerBlock: 5,
	}

	prompt, err := PreparePrompt(cfg)
	if err != nil {
		t.Fatalf("PreparePrompt returned error: %v", err)
	}

	if !strings.Contains(prompt, "blk_42") {
		t.Fatalf("prompt missing block ID: %q", prompt)
	}
	if !strings.Contains(prompt, "Disk failure") {
		t.Fatalf("prompt missing log line: %q", prompt)
	}
}
