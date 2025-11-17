package summarizer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCollectLogsForBlocks(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	logPath := filepath.Join(dir, "hdfs.log")
	logContent := strings.Join([]string{
		"INFO block blk_1 message A",
		"WARN block blk_2 message B",
		"INFO block blk_1 message C",
		"ERROR block blk_3 message D",
	}, "\n")

	if err := os.WriteFile(logPath, []byte(logContent), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}

	cases := []struct {
		name      string
		blockIDs  []string
		maxLines  int
		wantCount map[string]int
	}{
		{
			name:     "respectsLimit",
			blockIDs: []string{"blk_1", "blk_3"},
			maxLines: 2,
			wantCount: map[string]int{
				"blk_1": 2,
				"blk_3": 1,
			},
		},
		{
			name:     "noLimitCapturesAll",
			blockIDs: []string{"blk_1"},
			maxLines: 0,
			wantCount: map[string]int{
				"blk_1": 2,
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			logs, err := CollectLogsForBlocks(logPath, tc.blockIDs, tc.maxLines)
			if err != nil {
				t.Fatalf("CollectLogsForBlocks returned error: %v", err)
			}

			for id, expected := range tc.wantCount {
				if got := len(logs[id]); got != expected {
					t.Fatalf("block %s expected %d lines, got %d", id, expected, got)
				}
			}

			for id := range logs {
				found := false
				for _, target := range tc.blockIDs {
					if id == target {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("unexpected logs for block %s", id)
				}
			}
		})
	}
}
