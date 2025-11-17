package summarizer

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// SummaryConfig groups inputs needed for summarisation.
type SummaryConfig struct {
	PredictionsPath  string
	LogPath          string
	OutputPath       string
	TopN             int
	SystemPrompt     string
	MaxLinesPerBlock int
}

// GenerateSummary orchestrates anomaly extraction, log retrieval, prompt generation, and LLM invocation.
func GenerateSummary(ctx context.Context, cfg SummaryConfig, client LLMClient) (string, error) {
	prompt, err := PreparePrompt(cfg)
	if err != nil {
		return "", err
	}

	summary, err := client.Summarize(ctx, prompt)
	if err != nil {
		return "", err
	}

	summary = strings.TrimSpace(summary)
	if cfg.OutputPath != "" {
		if err := writeOutput(cfg.OutputPath, summary); err != nil {
			return "", err
		}
	}

	return summary, nil
}

// PreparePrompt assembles the anomaly prompt without invoking an external LLM.
func PreparePrompt(cfg SummaryConfig) (string, error) {
	if cfg.PredictionsPath == "" {
		return "", errors.New("predictions path is required")
	}
	if cfg.LogPath == "" {
		return "", errors.New("log path is required")
	}

	anomalies, err := LoadAnomalies(cfg.PredictionsPath)
	if err != nil {
		return "", err
	}
	if len(anomalies) == 0 {
		return "", errors.New("no anomalies detected in predictions")
	}

	top := RankAnomalies(anomalies, cfg.TopN)

	blockIDs := make([]string, 0, len(top))
	for _, anomaly := range top {
		blockIDs = append(blockIDs, anomaly.BlockID)
	}

	logs, err := CollectLogsForBlocks(cfg.LogPath, blockIDs, cfg.MaxLinesPerBlock)
	if err != nil {
		return "", err
	}

	prompt := BuildPrompt(top, logs, cfg.SystemPrompt)
	return prompt, nil
}

func writeOutput(path, summary string) error {
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create output directory: %w", err)
		}
	}

	if err := os.WriteFile(path, []byte(summary), 0o644); err != nil {
		return fmt.Errorf("write summary output: %w", err)
	}
	return nil
}
