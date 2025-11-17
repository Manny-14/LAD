package summarizer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCollectLogsForBlocks(t *testing.T) {
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

	logs, err := CollectLogsForBlocks(logPath, []string{"blk_1", "blk_3"}, 2)
	if err != nil {
		t.Fatalf("CollectLogsForBlocks returned error: %v", err)
	}

	if len(logs["blk_1"]) != 2 {
		t.Fatalf("expected 2 lines for blk_1, got %d", len(logs["blk_1"]))
	}

	if len(logs["blk_3"]) != 1 {
		t.Fatalf("expected 1 line for blk_3, got %d", len(logs["blk_3"]))
	}

	if _, ok := logs["blk_2"]; ok {
		t.Fatalf("did not expect logs for blk_2")
	}
}
