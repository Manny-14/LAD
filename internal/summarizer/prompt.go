package summarizer

import (
	"fmt"
	"strings"
)

// BuildPrompt prepares a plain English instruction for the target LLM.
func BuildPrompt(anomalies []Prediction, logs map[string][]string, systemPrompt string) string {
	if len(anomalies) == 0 {
		return "Summarize the following anomalies: (no anomalies provided)"
	}

	var b strings.Builder
	if systemPrompt != "" {
		b.WriteString(systemPrompt)
		if !strings.HasSuffix(systemPrompt, "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	b.WriteString("Analyze the following anomalous Hadoop Distributed File System (HDFS) blocks and provide a concise incident-oriented summary for an on-call site reliability engineer. Highlight root causes, impacted components, and any remediation suggestions.\n\n")

	for idx, anomaly := range anomalies {
		b.WriteString(fmt.Sprintf("Block %d: %s\n", idx+1, anomaly.BlockID))
		if len(logs[anomaly.BlockID]) == 0 {
			b.WriteString("(No log lines found for this block)\n\n")
			continue
		}

		for _, line := range logs[anomaly.BlockID] {
			b.WriteString("  • ")
			b.WriteString(line)
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}

	b.WriteString("Summarize the most critical issues, probable causes, and recommended next steps.\n")
	return b.String()
}
