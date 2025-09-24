package sllmi_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/sokinpui/sllmi-go"
)

// checkAPIKey skips the test if the GENAI_API_KEY is not set.
func checkAPIKey(t *testing.T) {
	if os.Getenv("GENAI_API_KEYS") == "" {
		t.Skip("Skipping test: GENAI_API_KEYS is not set.")
	}
}

func TestLLMRegistry(t *testing.T) {
	registry, err := sllmi.New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	t.Run("ListModels", func(t *testing.T) {
		models := registry.ListModels()
		if len(models) == 0 {
			t.Error("ListModels() returned an empty list")
		}
		t.Logf("Available models: %v", models)
	})

	t.Run("GetModel", func(t *testing.T) {
		// Test getting a valid model
		models := registry.ListModels()
		if len(models) == 0 {
			t.Fatal("No models available to test GetModel")
		}
		modelName := models[0]
		model, err := registry.GetModel(modelName)
		if err != nil {
			t.Errorf("GetModel(%q) returned an error: %v", modelName, err)
		}
		if model == nil {
			t.Errorf("GetModel(%q) returned a nil model", modelName)
		}

		// Test getting an invalid model
		_, err = registry.GetModel("non-existent-model")
		if err == nil {
			t.Error("GetModel(\"non-existent-model\") was expected to fail, but it did not")
		}
	})
}

func TestGeminiModel(t *testing.T) {
	checkAPIKey(t)

	registry, err := sllmi.New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	models := registry.ListModels()
	if len(models) == 0 {
		t.Fatal("No models available to test")
	}
	modelName := "gemma-3-27b-it" // Use a specific, fast model for testing
	model, err := registry.GetModel(modelName)
	if err != nil {
		t.Fatalf("Failed to get model %s: %v", modelName, err)
	}

	prompt := "Write a short poem about the sea in exactly 4 lines."

	t.Run("CountTokens", func(t *testing.T) {
		count, err := model.CountTokens(prompt)
		if err != nil {
			t.Errorf("CountTokens() failed: %v", err)
		}
		if count <= 0 {
			t.Errorf("CountTokens() returned an invalid count: %d", count)
		}
		t.Logf("Token count for prompt: %d", count)
	})

	t.Run("Generate", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		response, err := model.Generate(ctx, prompt, nil)
		if err != nil {
			t.Fatalf("Generate() failed: %v", err)
		}
		if response == "" {
			t.Error("Generate() returned an empty response")
		}
		fmt.Printf("\n--- Generate() Response ---\n%s\n---------------------------\n", response)
		// t.Logf is still useful for verbose mode
	})

	t.Run("GenerateStream", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		outCh, errCh := model.GenerateStream(ctx, prompt, nil)

		var responseParts []string
		var streamErr error

	streamLoop:
		for {
			select {
			case chunk, ok := <-outCh:
				if !ok {
					break streamLoop
				}
				responseParts = append(responseParts, chunk)
			case err := <-errCh:
				streamErr = err
				break streamLoop
			case <-ctx.Done():
				t.Fatal("GenerateStream() timed out")
			}
		}

		if streamErr != nil {
			t.Fatalf("GenerateStream() returned an error: %v", streamErr)
		}

		if len(responseParts) == 0 {
			t.Error("GenerateStream() did not produce any output")
		}

		fullResponse := strings.Join(responseParts, "")
		if fullResponse == "" {
			t.Error("GenerateStream() resulted in an empty string")
		}
		fmt.Printf("\n--- GenerateStream() Response ---\n%s\n-------------------------------\n", fullResponse)
		// t.Logf is still useful for verbose mode
	})
}

func TestGeminiModel_NoAPIKey(t *testing.T) {
	originalAPIKeys := os.Getenv("GENAI_API_KEYS")
	os.Unsetenv("GENAI_API_KEYS")
	defer os.Setenv("GENAI_API_KEYS", originalAPIKeys)

	registry, err := sllmi.New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	modelName := "gemma-3-27b-it"
	model, err := registry.GetModel(modelName)
	if err != nil {
		t.Fatalf("Failed to get model %s: %v", modelName, err)
	}

	prompt := "This is a test prompt."

	t.Run("CountTokens_NoKey", func(t *testing.T) {
		count, err := model.CountTokens(prompt)
		if err != nil {
			t.Errorf("CountTokens() with no API key failed: %v", err)
		}
		if count <= 0 {
			t.Errorf("CountTokens() with no API key returned an invalid count: %d", count)
		}
	})

	t.Run("Generate_NoKey", func(t *testing.T) {
		_, err := model.Generate(context.Background(), prompt, nil)
		if err == nil {
			t.Error("Generate() with no API key was expected to fail, but it did not")
		}
		if !strings.Contains(err.Error(), "API key is required for generation") {
			t.Errorf("Generate() with no API key returned an unexpected error: %v", err)
		}
	})

	t.Run("GenerateStream_NoKey", func(t *testing.T) {
		outCh, errCh := model.GenerateStream(context.Background(), prompt, nil)

		// Drain the output channel to prevent goroutine leaks, though it should be empty.
		for range outCh {
		}

		err := <-errCh
		if err == nil {
			t.Error("GenerateStream() with no API key was expected to fail, but it did not")
		}
		if !strings.Contains(err.Error(), "API key is required for generation") {
			t.Errorf("GenerateStream() with no API key returned an unexpected error: %v", err)
		}
	})
}
