package summarizer

import (
	"strings"
	"testing"
)

func TestBuildPrompt(t *testing.T) {
	anoms := []Prediction{{BlockID: "blk_1"}}
	logs := map[string][]string{
		"blk_1": {"line one", "line two"},
	}
	prompt := BuildPrompt(anoms, logs, "You are an assistant.")

	if len(prompt) == 0 {
		t.Fatal("prompt should not be empty")
	}

	if want, got := "Block 1: blk_1", prompt; !strings.Contains(prompt, want) {
		t.Fatalf("expected prompt to contain %q, got %q", want, got)
	}

	if !strings.Contains(prompt, "line one") || !strings.Contains(prompt, "line two") {
		t.Fatalf("prompt missing log lines: %q", prompt)
	}
}
