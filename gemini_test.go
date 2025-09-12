package sllmi_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sokinpui/sllmi-go"
)

func TestNewGeminiProvider(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		t.Setenv("GENAI_API_KEYS", "key1,key2")
		providers, err := newGeminiProvider()
		if err != nil {
			t.Fatalf("newGeminiProvider() error = %v, wantErr %v", err, false)
		}
		if len(providers) == 0 {
			t.Error("expected providers to be initialized, but got none")
		}
	})

	t.Run("FailureNoEnvVar", func(t *testing.T) {
		// Unset the variable if it exists
		t.Setenv("GENAI_API_KEYS", "")
		_, err := newGeminiProvider()
		if !errors.Is(err, ErrConfiguration) {
			t.Errorf("newGeminiProvider() error = %v, want %v", err, ErrConfiguration)
		}
		if !strings.Contains(err.Error(), "GENAI_API_KEYS environment variable not set") {
			t.Errorf("expected error message to contain 'not set', got '%s'", err.Error())
		}
	})

	t.Run("FailureEmptyEnvVar", func(t *testing.T) {
		t.Setenv("GENAI_API_KEYS", ",") // An empty or just comma var
		_, err := newGeminiProvider()
		if !errors.Is(err, ErrConfiguration) {
			t.Errorf("newGeminiProvider() error = %v, want %v", err, ErrConfiguration)
		}
		if !strings.Contains(err.Error(), "GENAI_API_KEYS environment variable is empty") {
			t.Errorf("expected error message to contain 'is empty', got '%s'", err.Error())
		}
	})
}

func TestNewGeminiModel(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		keys := []string{"key1", "key2"}
		model, err := NewGeminiModel(context.Background(), "gemini-2.5-pro", keys)
		if err != nil {
			t.Fatalf("NewGeminiModel() error = %v, wantErr %v", err, false)
		}
		if len(model.apiKeys) != len(keys) {
			t.Errorf("expected %d api keys, got %d", len(keys), len(model.apiKeys))
		}
	})

	t.Run("FailureNoKeys", func(t *testing.T) {
		_, err := NewGeminiModel(context.Background(), "gemini-2.5-pro", []string{})
		if !errors.Is(err, ErrConfiguration) {
			t.Errorf("NewGeminiModel() error = %v, want %v", err, ErrConfiguration)
		}
	})
}

func TestGeminiModel_Generate_AllKeysFail(t *testing.T) {
	// This is an integration test that relies on the remote service rejecting invalid keys.
	invalidKeys := []string{"invalid-key-1", "invalid-key-2"}
	model, err := NewGeminiModel(context.Background(), "gemini-2.5-pro", invalidKeys)
	if err != nil {
		t.Fatalf("failed to create model for testing: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	_, err = model.Generate(ctx, "test prompt", nil)

	if err == nil {
		t.Fatal("Generate() error = nil, want an error")
	}

	if !strings.Contains(err.Error(), "all API keys failed") {
		t.Errorf("expected error message to indicate all keys failed, but got: %v", err)
	}

	if !errors.Is(err, ErrGeneration) {
		t.Errorf("expected error to wrap ErrGeneration, but it did not")
	}
}

func TestGeminiModel_GenerateStream_AllKeysFail(t *testing.T) {
	// This is an integration test similar to the non-streaming version.
	invalidKeys := []string{"invalid-key-1", "invalid-key-2"}
	model, err := NewGeminiModel(context.Background(), "gemini-2.5-pro", invalidKeys)
	if err != nil {
		t.Fatalf("failed to create model for testing: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	_, errCh := model.GenerateStream(ctx, "test prompt", nil)

	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("GenerateStream() error = nil, want an error from channel")
		}
		if !strings.Contains(err.Error(), "all API keys failed") {
			t.Errorf("expected error message to indicate all keys failed, but got: %v", err)
		}
		if !errors.Is(err, ErrGeneration) {
			t.Errorf("expected error to wrap ErrGeneration, but it did not")
		}
	case <-ctx.Done():
		t.Fatal("test timed out waiting for an error from the error channel")
	}
}

func TestGeminiModel_CountTokens(t *testing.T) {
	model, err := NewGeminiModel(context.Background(), "gemini-2.5-pro", []string{"dummy-key"})
	if err != nil {
		t.Fatalf("failed to create model for testing: %v", err)
	}

	prompt := "hello world"
	count, err := model.CountTokens(prompt)
	if err != nil {
		t.Fatalf("CountTokens() error = %v, wantErr %v", err, false)
	}

	expectedCount := 2
	if count != expectedCount {
		t.Errorf("CountTokens() = %v, want %v", count, expectedCount)
	}
}
