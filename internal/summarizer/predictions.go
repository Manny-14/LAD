package summarizer

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Prediction captures a single row from the inference output.
type Prediction struct {
	BlockID           string
	Prediction        int
	AnomalyScore      float64
	DecisionThreshold float64
	ScoreDirection    string
}

// LoadAnomalies parses the predictions CSV and returns rows flagged as anomalous (prediction == 1).
func LoadAnomalies(path string) ([]Prediction, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open predictions csv: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	index := make(map[string]int, len(header))
	for i, col := range header {
		index[strings.ToLower(col)] = i
	}

	required := []string{"blockid", "prediction"}
	for _, col := range required {
		if _, ok := index[col]; !ok {
			return nil, fmt.Errorf("predictions csv missing required column %q", col)
		}
	}

	_, hasScore := index["anomaly_score"]
	_, hasThreshold := index["decision_threshold"]
	_, hasDirection := index["score_direction"]

	var anomalies []Prediction
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read record: %w", err)
		}

		predVal := strings.TrimSpace(record[index["prediction"]])
		if predVal == "" {
			continue
		}
		predInt, err := strconv.Atoi(predVal)
		if err != nil {
			return nil, fmt.Errorf("parse prediction value %q: %w", predVal, err)
		}
		if predInt != 1 {
			continue
		}

		prediction := Prediction{
			BlockID:    record[index["blockid"]],
			Prediction: predInt,
		}

		if hasScore {
			scoreStr := strings.TrimSpace(record[index["anomaly_score"]])
			if scoreStr != "" {
				if prediction.AnomalyScore, err = strconv.ParseFloat(scoreStr, 64); err != nil {
					return nil, fmt.Errorf("parse anomaly_score for block %s: %w", prediction.BlockID, err)
				}
			}
		}

		if hasThreshold {
			thrStr := strings.TrimSpace(record[index["decision_threshold"]])
			if thrStr != "" {
				if prediction.DecisionThreshold, err = strconv.ParseFloat(thrStr, 64); err != nil {
					return nil, fmt.Errorf("parse decision_threshold for block %s: %w", prediction.BlockID, err)
				}
			}
		}

		if hasDirection {
			prediction.ScoreDirection = strings.ToLower(strings.TrimSpace(record[index["score_direction"]]))
		} else {
			prediction.ScoreDirection = "lower"
		}

		anomalies = append(anomalies, prediction)
	}

	return anomalies, nil
}

// RankAnomalies sorts anomalies by severity and returns the top N (or all if N <= 0 or exceeds length).
func RankAnomalies(anomalies []Prediction, topN int) []Prediction {
	if len(anomalies) == 0 {
		return anomalies
	}

	sort.SliceStable(anomalies, func(i, j int) bool {
		return severity(anomalies[i]) > severity(anomalies[j])
	})

	if topN <= 0 || topN >= len(anomalies) {
		return anomalies
	}
	return anomalies[:topN]
}

func severity(p Prediction) float64 {
	direction := strings.ToLower(p.ScoreDirection)

	if direction == "" {
		direction = "lower"
	}

	switch direction {
	case "higher":
		if p.DecisionThreshold == 0 && p.AnomalyScore == 0 {
			return 0
		}
		return p.AnomalyScore - p.DecisionThreshold
	default:
		if p.DecisionThreshold == 0 && p.AnomalyScore == 0 {
			return 0
		}
		return p.DecisionThreshold - p.AnomalyScore
	}
}
