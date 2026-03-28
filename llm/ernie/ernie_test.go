package ernie

import (
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestBuildRequest_SystemMessage(t *testing.T) {
	p := New("key", "secret")
	req := llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
	}
	data := p.buildRequest(req, false)
	if len(data) == 0 {
		t.Fatal("empty request body")
	}
	// system should be extracted to top-level field, not in messages
	s := string(data)
	if !contains(s, `"system":"You are helpful"`) {
		t.Errorf("system message not extracted: %s", s)
	}
}

func TestBuildRequest_UserFirst(t *testing.T) {
	p := New("key", "secret")
	req := llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "assistant", Content: "hi"},
			{Role: "user", Content: "hello"},
		},
	}
	data := p.buildRequest(req, false)
	s := string(data)
	// Messages should start with user (ERNIE requirement)
	if !contains(s, `"role":"user"`) {
		t.Errorf("first message should be user: %s", s)
	}
}

func TestParseResponse_Success(t *testing.T) {
	p := New("key", "secret")
	data := []byte(`{"id":"123","result":"Hello!","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	resp, err := p.parseResponse(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello!" {
		t.Errorf("got %q, want %q", resp.Content, "Hello!")
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("got %d tokens, want 15", resp.Usage.TotalTokens)
	}
}

func TestParseResponse_Error(t *testing.T) {
	p := New("key", "secret")
	data := []byte(`{"error_code":110,"error_msg":"Access token invalid"}`)
	_, err := p.parseResponse(data)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestName(t *testing.T) {
	p := New("key", "secret")
	if p.Name() != "ernie" {
		t.Errorf("got %q, want %q", p.Name(), "ernie")
	}
}

func TestModels(t *testing.T) {
	p := New("key", "secret")
	models := p.Models()
	if len(models) == 0 {
		t.Error("expected non-empty models list")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsStr(s, sub))
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
