package parser

import (
	"testing"
	"time"

	logpb "github.com/Manny-14/LAD/gen/proto/log"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestParseLogLine(t *testing.T) {
	// Define the layout for parsing the timestamp from the log file.
	const layout = "060102 150405"

	// Helper to create a timestamp for expected values.
	mustParseTime := func(value string) time.Time {
		t, err := time.Parse(layout, value)
		if err != nil {
			panic(err)
		}
		return t
	}

	testCases := []struct {
		name      string
		line      string
		want      *logpb.LogEntry
		expectErr bool
	}{
		{
			name: "valid info log line",
			line: "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
			want: &logpb.LogEntry{
				Timestamp: mustParseTime("081109 203615").Unix(),
				Pid:       148,
				Severity:  logpb.Severity_INFO,
				Component: "dfs.DataNode$PacketResponder",
				Message:   "PacketResponder 1 for block blk_38865049064139660 terminating",
			},
			expectErr: false,
		},
		{
			name: "valid warn log line",
			line: "081109 214043 2561 WARN dfs.DataNode$DataXceiver: 10.251.30.85:50010:Got exception while serving blk_-2918118818249673980 to /10.251.90.64:",
			want: &logpb.LogEntry{
				Timestamp: mustParseTime("081109 214043").Unix(),
				Pid:       2561,
				Severity:  logpb.Severity_WARN,
				Component: "dfs.DataNode$DataXceiver",
				Message:   "10.251.30.85:50010:Got exception while serving blk_-2918118818249673980 to /10.251.90.64:",
			},
			expectErr: false,
		},
		{
			name:      "line with too few parts",
			line:      "081109 203615 148 INFO",
			want:      nil,
			expectErr: true,
		},
		{
			name:      "line with a non-numeric PID",
			line:      "081109 203615 not-a-pid INFO dfs.DataNode$PacketResponder: message",
			want:      nil,
			expectErr: true,
		},
		{
			name:      "line with an invalid timestamp",
			line:      "not-a-date not-a-time 148 INFO dfs.DataNode$PacketResponder: message",
			want:      nil,
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseLogLine(tc.line)

			if tc.expectErr {
				if err == nil {
					t.Error("ParseLogLine() expected an error but got nil")
				}
				return // Test passes if an error was expected and received.
			}

			if err != nil {
				t.Errorf("ParseLogLine() returned an unexpected error: %v", err)
			}

			if diff := cmp.Diff(tc.want, got, protocmp.Transform()); diff != "" {
				t.Errorf("ParseLogLine() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
