package sllmigo

import (
	"context"
	"fmt"
	"io"

	vgenai "cloud.google.com/go/vertexai/genai"
	"cloud.google.com/go/vertexai/genai/tokenizer"
	"google.golang.org/genai"
)

type GeminiModel struct {
	client *genai.Client
	model  string
}

func NewGeminiModel(ctx context.Context, modelCode, apiKey string) (*GeminiModel, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("%w: GENAI_API_KEY is required for GeminiModel", ErrConfiguration)
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	return &GeminiModel{
		client: client,
		model:  modelCode,
	}, nil
}

// Generate performs a non-streaming text generation.
func (m *GeminiModel) Generate(ctx context.Context, prompt string, config *Config) (string, error) {
	genConfig := getGenConfig(config)

	resp, err := m.client.Models.GenerateContent(ctx,
		m.model,
		genai.Text(prompt),
		genConfig,
	)
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
	genConfig := getGenConfig(config)

	outCh := make(chan string)
	errCh := make(chan error, 1)

	go func() {
		defer close(outCh)
		defer close(errCh)

		iter := m.client.Models.GenerateContentStream(ctx,
			m.model,
			genai.Text(prompt),
			genConfig,
		)
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
	tok, err := tokenizer.New("gemini-1.5-flash")
	if err != nil {
		return 0, fmt.Errorf("token counting failed: %w", err)
	}

	ntoks, err := tok.CountTokens(vgenai.Text(prompt))
	if err != nil {
		return 0, fmt.Errorf("token counting failed: %w", err)
	}

	return int(ntoks.TotalTokens), nil
}

func getGenConfig(config *Config) *genai.GenerateContentConfig {
	if config == nil {
		return &genai.GenerateContentConfig{}
	}

	if config.TopK != nil {
	}

	return &genai.GenerateContentConfig{
		Temperature:     config.Temperature,
		TopP:            config.TopP,
		TopK:            config.TopK,
		MaxOutputTokens: config.OutputLength,
	}
}
