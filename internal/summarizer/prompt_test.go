package summarizer

import (
	"strings"
	"testing"
)

func TestBuildPrompt(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		anomalies    []Prediction
		logs         map[string][]string
		systemPrompt string
		wantSnippets []string
	}{
		{
			name:      "includesLogLines",
			anomalies: []Prediction{{BlockID: "blk_1"}},
			logs: map[string][]string{
				"blk_1": []string{"line one", "line two"},
			},
			systemPrompt: "You are an assistant.",
			wantSnippets: []string{"Block 1: blk_1", "line one", "line two", "You are an assistant."},
		},
		{
			name:         "omitsMissingLogs",
			anomalies:    []Prediction{{BlockID: "blk_2"}},
			logs:         map[string][]string{},
			systemPrompt: "",
			wantSnippets: []string{"Block 1: blk_2"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			prompt := BuildPrompt(tc.anomalies, tc.logs, tc.systemPrompt)
			if len(prompt) == 0 {
				t.Fatal("prompt should not be empty")
			}

			for _, snippet := range tc.wantSnippets {
				if !strings.Contains(prompt, snippet) {
					t.Fatalf("prompt missing %q: %q", snippet, prompt)
				}
			}
		})
	}
}
