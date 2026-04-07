package ollama

import (
	"encoding/json"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestBuildRequestBodyMapsThinkingMetadataToOllamaThink(t *testing.T) {
	p := New()

	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model: "qwen3.5:9b",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "hi"},
		},
		Metadata: map[string]any{"thinking": "off"},
	}, true)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	got, ok := payload["think"].(bool)
	if !ok {
		t.Fatalf("payload[think] type = %T, want bool; payload=%v", payload["think"], payload)
	}
	if got {
		t.Fatalf("payload[think] = true, want false")
	}
}

func TestBuildRequestBodyPreservesExplicitThinkingOn(t *testing.T) {
	p := New()

	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Metadata: map[string]any{"thinking": "on"},
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	got, ok := payload["think"].(bool)
	if !ok {
		t.Fatalf("payload[think] type = %T, want bool; payload=%v", payload["think"], payload)
	}
	if !got {
		t.Fatalf("payload[think] = false, want true")
	}
}
