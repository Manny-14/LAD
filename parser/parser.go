package parser

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	logpb "github.com/Manny-14/LAD/gen/proto/log"
)

func ParseLogLine(line string) (*logpb.LogEntry, error) {
	parts := strings.Fields(line)
	if len(parts) < 5 {
		return nil, fmt.Errorf("invalid log format: not enough parts")
	}

	// Parse timestamp (YYMMDD HHMMSS)
	timeStampStr := parts[0] + " " + parts[1]
	const layout = "060102 150405"
	t, err := time.Parse(layout, timeStampStr)
	if err != nil {
		return nil, fmt.Errorf("could not parse timestamp: %w", err)
	}

	// Parse PID
	pid, err := strconv.Atoi(parts[2])
	if err != nil {
		return nil, fmt.Errorf("could not parse pid: %w", err)
	}

	// Parse Severity
	var severity logpb.Severity
	switch parts[3] {
	case "INFO":
		severity = logpb.Severity_INFO
	case "WARN":
		severity = logpb.Severity_WARN
	default:
		severity = logpb.Severity_UNKNOWN
	}

	// Parse Component and Message
	component := strings.TrimSuffix(parts[4], ":")
	message := strings.Join(parts[5:], " ")

	return &logpb.LogEntry{
		Timestamp: t.Unix(),
		Pid:       int32(pid),
		Severity:  severity,
		Component: component,
		Message:   message,
	}, nil
}
