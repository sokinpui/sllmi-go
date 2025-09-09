package sllmigo

import (
	"context"
	"fmt"
	"io"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// GeminiModel is an implementation of the LLM interface for Google's Gemini models.
type GeminiModel struct {
	client    *genai.GenerativeModel
	tokenizer *genai.Tokenizer
}

// NewGeminiModel creates and initializes a new GeminiModel.
func NewGeminiModel(ctx context.Context, modelCode, apiKey string) (*GeminiModel, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("%w: GENAI_API_KEY is required for GeminiModel", ErrConfiguration)
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	model := client.GenerativeModel(modelCode)
	tokenizer := client.Tokenizer()

	return &GeminiModel{
		client:    model,
		tokenizer: tokenizer,
	}, nil
}

// Generate performs a non-streaming text generation.
func (m *GeminiModel) Generate(ctx context.Context, prompt string, config *Config) (string, error) {
	m.applyConfig(config)

	resp, err := m.client.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrGeneration, err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("%w: no content in response", ErrGeneration)
	}

	return fmt.Sprintf("%s", resp.Candidates[0].Content.Parts[0]), nil
}

// GenerateStream performs a streaming text generation.
func (m *GeminiModel) GenerateStream(ctx context.Context, prompt string, config *Config) (<-chan string, <-chan error) {
	m.applyConfig(config)

	outCh := make(chan string)
	errCh := make(chan error, 1)

	go func() {
		defer close(outCh)
		defer close(errCh)

		iter := m.client.GenerateContentStream(ctx, genai.Text(prompt))
		for {
			resp, err := iter.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				errCh <- fmt.Errorf("%w: %v", ErrGeneration, err)
				return
			}
			if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
				outCh <- fmt.Sprintf("%s", resp.Candidates[0].Content.Parts[0])
			}
		}
	}()

	return outCh, errCh
}

// CountTokens counts the number of tokens in a prompt.
func (m *GeminiModel) CountTokens(prompt string) (int, error) {
	resp, err := m.tokenizer.CountTokens(context.Background(), genai.Text(prompt))
	if err != nil {
		return 0, fmt.Errorf("token counting failed: %w", err)
	}
	return int(resp.TotalTokens), nil
}

// applyConfig applies the generation configuration to the model.
func (m *GeminiModel) applyConfig(config *Config) {
	if config == nil {
		return
	}
	if config.Temperature != nil {
		m.client.Temperature = config.Temperature
	}
	if config.TopP != nil {
		m.client.TopP = config.TopP
	}
	if config.TopK != nil {
		m.client.TopK = config.TopK
	}
	if config.OutputLength != nil {
		m.client.MaxOutputTokens = config.OutputLength
	}
}
