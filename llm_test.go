package sllmigo_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"sllmi-go"
)

// checkAPIKey skips the test if the GENAI_API_KEY is not set.
func checkAPIKey(t *testing.T) {
	if os.Getenv("GENAI_API_KEY") == "" {
		t.Skip("Skipping test: GENAI_API_KEY is not set.")
	}
}

func TestLLMRegistry(t *testing.T) {
	checkAPIKey(t)

	registry, err := sllmigo.New()
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

	registry, err := sllmigo.New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	models := registry.ListModels()
	if len(models) == 0 {
		t.Fatal("No models available to test")
	}
	modelName := "gemini-2.5-flash" // Use a specific, fast model for testing
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
		t.Logf("Generate() response:\n%s", response)
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
		t.Logf("GenerateStream() response:\n%s", fullResponse)
	})
}
