package summarizer

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
)

var blockIDPattern = regexp.MustCompile(`blk_[0-9\-]+`)

// CollectLogsForBlocks scans the provided log file and returns matching lines grouped by block ID.
func CollectLogsForBlocks(logPath string, blockIDs []string, maxLinesPerBlock int) (map[string][]string, error) {
	if len(blockIDs) == 0 {
		return map[string][]string{}, nil
	}

	set := make(map[string]struct{}, len(blockIDs))
	for _, id := range blockIDs {
		set[id] = struct{}{}
	}

	file, err := os.Open(logPath)
	if err != nil {
		return nil, fmt.Errorf("open log file: %w", err)
	}
	defer file.Close()

	logs := make(map[string][]string, len(blockIDs))
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		matches := blockIDPattern.FindAllString(line, -1)
		if len(matches) == 0 {
			continue
		}
		seen := make(map[string]struct{}, len(matches))
		for _, match := range matches {
			if _, ok := set[match]; !ok {
				continue
			}
			if _, already := seen[match]; already {
				continue
			}
			seen[match] = struct{}{}

			// respect max lines per block if requested
			if maxLinesPerBlock > 0 && len(logs[match]) >= maxLinesPerBlock {
				continue
			}
			logs[match] = append(logs[match], line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan log file: %w", err)
	}

	return logs, nil
}
