package summarizer

import (
	"context"
	"errors"
	"fmt"
	"strings"

	genai "github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// DefaultModel is the suggested Gemini model for summarisation.
const DefaultModel = "gemini-1.5-flash"

// LLMClient defines the behaviour required to obtain anomaly summaries.
type LLMClient interface {
	Summarize(ctx context.Context, prompt string) (string, error)
}

// GeminiClient implements LLMClient using Google Gemini.
type GeminiClient struct {
	client  *genai.Client
	model   *genai.GenerativeModel
	apiKey  string
	modelID string
}

// NewGeminiClient boots a Gemini API client using the provided API key and model id.
func NewGeminiClient(ctx context.Context, apiKey, modelID string) (*GeminiClient, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, errors.New("gemini api key is required")
	}
	if modelID == "" {
		modelID = DefaultModel
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("create gemini client: %w", err)
	}

	return &GeminiClient{
		client:  client,
		model:   client.GenerativeModel(modelID),
		apiKey:  apiKey,
		modelID: modelID,
	}, nil
}

// Close releases the underlying Gemini client.
func (g *GeminiClient) Close() error {
	if g == nil || g.client == nil {
		return nil
	}
	return g.client.Close()
}

// Summarize sends the prompt to Gemini and returns the aggregated text response.
func (g *GeminiClient) Summarize(ctx context.Context, prompt string) (string, error) {
	if g == nil || g.model == nil {
		return "", errors.New("gemini client is not initialised")
	}

	resp, err := g.model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("generate content: %w", err)
	}

	var b strings.Builder
	for _, cand := range resp.Candidates {
		for _, part := range cand.Content.Parts {
			if text, ok := part.(genai.Text); ok {
				if b.Len() > 0 {
					b.WriteString("\n")
				}
				b.WriteString(string(text))
			}
		}
	}

	if b.Len() == 0 {
		return "", errors.New("gemini response contained no text candidates")
	}

	return b.String(), nil
}

// StaticClient implements LLMClient by returning a pre-defined message.
type StaticClient struct {
	Response string
}

// Summarize returns the configured response without calling an external service.
func (s StaticClient) Summarize(_ context.Context, _ string) (string, error) {
	return s.Response, nil
}
