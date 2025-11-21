package sllmi

import (
	"context"
	"fmt"
	"os"
	"testing"
)

func TestOpenRouterModel_Generate(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping test")
	}

	ctx := context.Background()
	model, err := NewOpenRouterModel(ctx, "z-ai/glm-4.5-air:free", apiKey)
	if err != nil {
		t.Fatalf("Failed to create OpenRouterModel: %v", err)
	}

	prompt := "write a short story"
	config := &Config{}

	result, err := model.Generate(ctx, prompt, nil, config)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if result == "" {
		t.Error("Expected a non-empty result, but got an empty string.")
	}

	fmt.Println("--- OpenRouter Generate Test ---")
	fmt.Println("Prompt:", prompt)
	fmt.Println("Result:", result)
	fmt.Println("------------------------------")
}

func TestOpenRouterModel_GenerateStream(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping test")
	}

	ctx := context.Background()
	model, err := NewOpenRouterModel(ctx, "z-ai/glm-4.5-air:free", apiKey)
	if err != nil {
		t.Fatalf("Failed to create OpenRouterModel: %v", err)
	}

	prompt := "write a short story"
	config := &Config{}

	outCh, errCh := model.GenerateStream(ctx, prompt, nil, config)

	fmt.Println("--- OpenRouter GenerateStream Test ---")
	fmt.Println("Prompt:", prompt)
	fmt.Print("Result: ")

	var fullResponse string
	for {
		select {
		case token, ok := <-outCh:
			if !ok {
				outCh = nil
			} else {
				fullResponse += token
				fmt.Print(token)
			}
		case err, ok := <-errCh:
			if !ok {
				errCh = nil
			} else {
				t.Fatalf("GenerateStream failed: %v", err)
			}
		}
		if outCh == nil && errCh == nil {
			break
		}
	}
	fmt.Println("\n------------------------------------")

	if fullResponse == "" {
		t.Error("Expected a non-empty result from stream, but got an empty string.")
	}
}
