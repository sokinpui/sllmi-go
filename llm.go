package sllmigo

import (
	"context"
	"fmt"
	"os"
)

// LLM defines the interface for a large language model.
type LLM interface {
	Generate(ctx context.Context, prompt string, config *Config) (string, error)
	GenerateStream(ctx context.Context, prompt string, config *Config) (<-chan string, <-chan error)
	CountTokens(prompt string) (int, error)
}

// Registry holds all available LLM implementations.
type Registry struct {
	models map[string]LLM
}

// New creates a new LLM registry and initializes the models.
func New() (*Registry, error) {
	apiKey := os.Getenv("GENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Warning: GENAI_API_KEY environment variable not set.")
	}

	models := make(map[string]LLM)

	modelCodes := []string{
		"gemini-1.5-pro-latest",
		"gemini-1.5-flash-latest",
		"gemini-1.0-pro",
	}

	for _, code := range modelCodes {
		model, err := NewGeminiModel(context.Background(), code, apiKey)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize model %s: %w", code, err)
		}
		models[code] = model
	}

	return &Registry{models: models}, nil
}

// GetModel retrieves a model from the registry by its code.
func (r *Registry) GetModel(modelCode string) (LLM, error) {
	model, ok := r.models[modelCode]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrModelNotFound, modelCode)
	}
	return model, nil
}

// ListModels returns a list of all available model codes.
func (r *Registry) ListModels() []string {
	keys := make([]string, 0, len(r.models))
	for k := range r.models {
		keys = append(keys, k)
	}
	return keys
}
