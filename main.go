package main

import (
	"bufio"
	"encoding/csv"
	"log"
	"os"
	"strconv"

	logpb "github.com/Manny-14/LAD/gen/proto/log"
	"github.com/Manny-14/LAD/parser"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatal("Error: Please provide an input log file path and an output CSV file path.")
	}
	inputPath := os.Args[1]
	outputPath := os.Args[2]

	// Open input file
	inputFile, err := os.Open(inputPath)
	if err != nil {
		log.Fatalf("Error opening input file: %v", err)
	}
	defer inputFile.Close()

	// Create output file
	outputFile, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Error creating output CSV file: %v", err)
	}
	defer outputFile.Close()

	csvWriter := csv.NewWriter(outputFile)
	defer csvWriter.Flush()

	header := []string{"timestamp", "pid", "severity", "component", "message"}
	if err := csvWriter.Write(header); err != nil {
		log.Fatalf("Error writing CSV header: %v", err)
	}

	// process log file
	scanner := bufio.NewScanner(inputFile)
	for scanner.Scan() {
		line := scanner.Text()
		logEntry, err := parser.ParseLogLine(line)
		if err != nil {
			// Log errors to stderr instead of stdout
			log.Printf("Warning: Skipping malformed line: %q (%v)\n", line, err)
			continue
		}

		record := logEntryToRecord(logEntry)
		if err := csvWriter.Write(record); err != nil {
			log.Printf("Warning: Could not write record to CSV: %v", err)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading from file: %v", err)
	}

	log.Printf("Successfully processed log file and saved structured data to %s\n", outputPath)
}

func logEntryToRecord(entry *logpb.LogEntry) []string {
	return []string{
		strconv.Itoa(int(entry.Timestamp)),
		strconv.Itoa(int(entry.Pid)),
		entry.Severity.String(),
		entry.Component,
		entry.Message,
	}
}
