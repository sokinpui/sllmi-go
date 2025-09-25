# sllmi-go

Simple LLM Interface for Go.

## Core Interfaces

**LLM Interface**

```go
type LLM interface {
    Generate(ctx context.Context, prompt string, imgPaths []string, config *Config) (string, error)
    GenerateStream(ctx context.Context, prompt string, imgPaths []string, config *Config) (<-chan string, <-chan error)
    CountTokens(prompt string) (int, error)
}
```

**Generation Config**

```go
type Config struct {
	Temperature  *float32 `json:"temperature,omitempty"`
	TopP         *float32 `json:"top_p,omitempty"`
	TopK         *float32 `json:"top_k,omitempty"`
	OutputLength int32    `json:"output_length,omitempty"`
}
```

## Installation

```sh
go get github.com/sokinpui/sllmi-go
```

## Usage

1.  For the default Gemini provider, set `GENAI_API_KEY`.

    ```sh
    export GENAI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

2.  **Initialize and Use a Model**:

    ```go
    package main

    import (
    	"context"
    	"fmt"
    	"log"

    	"github.com/sokinpui/sllmi-go"
    )

    func main() {
    	// New() automatically initializes all registered providers.
    	registry, err := sllmi.New()
    	if err != nil {
    		log.Fatalf("Failed to initialize registry: %v", err)
    	}

    	// List available models
    	fmt.Println("Available models:", registry.ListModels())

    	// Get a specific model
    	model, err := registry.GetModel("gemini-1.5-flash")
    	if err != nil {
    		log.Fatalf("Failed to get model: %v", err)
    	}

    	// Perform generation
    	ctx := context.Background()
    	result, err := model.Generate(ctx, "Explain the importance of APIs in one sentence.", nil)
    	if err != nil {
    		log.Fatalf("Generation failed: %v", err)
    	}
    	fmt.Printf("\nResult: %s\n", result)
    }
    ```

## Extending with a New Provider

To add support for a new LLM provider (e.g., OpenAI):

1.  **Create `openai.go`**: Inside this file, implement the `sllmi.LLM` interface for the new model.
2.  **Create a Provider Function**: This function initializes the client, creates instances of your model(s), and returns them in a `map[string]LLM`.
    ```go
    func newOpenAIProvider() (map[string]sllmi.LLM, error) {
        // 1. Check for OPENAI_API_KEY env var
        // 2. Initialize OpenAI client
        // 3. Create map and populate with OpenAI models
        // 4. Return map
    }
    ```
3.  **Register the Provider**: Use an `init()` function to register your provider with the library.
    ```go
    func init() {
    	sllmi.RegisterProvider(newOpenAIProvider)
    }
    ```
