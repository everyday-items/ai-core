package llm

import (
	"testing"
)

func TestNewMessages(t *testing.T) {
	// 带系统消息
	msgs := NewMessages("system prompt", "user1", "user2")
	if len(msgs) != 3 {
		t.Errorf("len(msgs) = %d, want 3", len(msgs))
	}
	if msgs[0].Role != RoleSystem {
		t.Errorf("msgs[0].Role = %q, want %q", msgs[0].Role, RoleSystem)
	}
	if msgs[0].Content != "system prompt" {
		t.Errorf("msgs[0].Content = %q, want %q", msgs[0].Content, "system prompt")
	}
	if msgs[1].Role != RoleUser {
		t.Errorf("msgs[1].Role = %q, want %q", msgs[1].Role, RoleUser)
	}

	// 不带系统消息
	msgs = NewMessages("", "user1")
	if len(msgs) != 1 {
		t.Errorf("len(msgs) = %d, want 1", len(msgs))
	}
	if msgs[0].Role != RoleUser {
		t.Errorf("msgs[0].Role = %q, want %q", msgs[0].Role, RoleUser)
	}
}

func TestNewMessage(t *testing.T) {
	msg := NewMessage(RoleUser, "hello")
	if msg.Role != RoleUser {
		t.Errorf("Role = %q, want %q", msg.Role, RoleUser)
	}
	if msg.Content != "hello" {
		t.Errorf("Content = %q, want %q", msg.Content, "hello")
	}
}

func TestMessageHelpers(t *testing.T) {
	tests := []struct {
		name    string
		msg     Message
		role    Role
		content string
	}{
		{"SystemMessage", SystemMessage("sys"), RoleSystem, "sys"},
		{"UserMessage", UserMessage("user"), RoleUser, "user"},
		{"AssistantMessage", AssistantMessage("assistant"), RoleAssistant, "assistant"},
		{"ToolMessage", ToolMessage("tool"), RoleTool, "tool"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.msg.Role != tt.role {
				t.Errorf("Role = %q, want %q", tt.msg.Role, tt.role)
			}
			if tt.msg.Content != tt.content {
				t.Errorf("Content = %q, want %q", tt.msg.Content, tt.content)
			}
		})
	}
}

func TestNewToolDefinition(t *testing.T) {
	schema := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"query": {Type: "string", Description: "搜索词"},
		},
	}

	def := NewToolDefinition("search", "搜索工具", schema)

	if def.Type != "function" {
		t.Errorf("Type = %q, want %q", def.Type, "function")
	}
	if def.Function.Name != "search" {
		t.Errorf("Function.Name = %q, want %q", def.Function.Name, "search")
	}
	if def.Function.Description != "搜索工具" {
		t.Errorf("Function.Description = %q, want %q", def.Function.Description, "搜索工具")
	}
	if def.Function.Parameters == nil {
		t.Error("Function.Parameters is nil")
	}
}

func TestCompletionResponse_HasToolCalls(t *testing.T) {
	// 无工具调用
	resp := &CompletionResponse{}
	if resp.HasToolCalls() {
		t.Error("HasToolCalls() should be false when no tool calls")
	}

	// 有工具调用
	resp = &CompletionResponse{
		ToolCalls: []ToolCall{{ID: "1", Name: "test"}},
	}
	if !resp.HasToolCalls() {
		t.Error("HasToolCalls() should be true when has tool calls")
	}
}

func TestModelInfo_HasFeature(t *testing.T) {
	model := ModelInfo{
		ID:       "gpt-4",
		Name:     "GPT-4",
		Features: []string{FeatureVision, FeatureFunctions, FeatureStreaming},
	}

	// 支持的特性
	if !model.HasFeature(FeatureVision) {
		t.Error("HasFeature(vision) should be true")
	}
	if !model.HasFeature(FeatureFunctions) {
		t.Error("HasFeature(functions) should be true")
	}

	// 不支持的特性
	if model.HasFeature(FeatureEmbedding) {
		t.Error("HasFeature(embedding) should be false")
	}
}

func TestFeatureConstants(t *testing.T) {
	// 确保特性常量有定义且不为空
	features := []string{
		FeatureVision,
		FeatureFunctions,
		FeatureJSON,
		FeatureStreaming,
		FeatureEmbedding,
	}

	for _, f := range features {
		if f == "" {
			t.Error("Feature constant should not be empty")
		}
	}
}

func TestRoleConstants(t *testing.T) {
	// 确保角色常量符合预期
	if RoleSystem != "system" {
		t.Errorf("RoleSystem = %q, want %q", RoleSystem, "system")
	}
	if RoleUser != "user" {
		t.Errorf("RoleUser = %q, want %q", RoleUser, "user")
	}
	if RoleAssistant != "assistant" {
		t.Errorf("RoleAssistant = %q, want %q", RoleAssistant, "assistant")
	}
	if RoleTool != "tool" {
		t.Errorf("RoleTool = %q, want %q", RoleTool, "tool")
	}
}

func TestCompletionRequest_Defaults(t *testing.T) {
	req := CompletionRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: "hello"},
		},
	}

	if req.Model != "gpt-4" {
		t.Errorf("Model = %q, want %q", req.Model, "gpt-4")
	}
	if len(req.Messages) != 1 {
		t.Errorf("len(Messages) = %d, want 1", len(req.Messages))
	}
	if req.MaxTokens != 0 {
		t.Errorf("MaxTokens = %d, want 0 (default)", req.MaxTokens)
	}
	if req.Temperature != nil {
		t.Error("Temperature should be nil by default")
	}
}

func TestCompletionRequest_WithOptions(t *testing.T) {
	temp := 0.7
	topP := 0.9

	req := CompletionRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: "hello"},
		},
		MaxTokens:   1000,
		Temperature: &temp,
		TopP:        &topP,
		Stop:        []string{"END"},
		User:        "user-123",
	}

	if req.MaxTokens != 1000 {
		t.Errorf("MaxTokens = %d, want 1000", req.MaxTokens)
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Error("Temperature should be 0.7")
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Error("TopP should be 0.9")
	}
	if len(req.Stop) != 1 || req.Stop[0] != "END" {
		t.Error("Stop should contain 'END'")
	}
	if req.User != "user-123" {
		t.Errorf("User = %q, want %q", req.User, "user-123")
	}
}

func TestCompletionRequest_WithTools(t *testing.T) {
	req := CompletionRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: "hello"},
		},
		Tools: []ToolDefinition{
			NewToolDefinition("search", "搜索", nil),
		},
		ToolChoice: "auto",
	}

	if len(req.Tools) != 1 {
		t.Errorf("len(Tools) = %d, want 1", len(req.Tools))
	}
	if req.ToolChoice != "auto" {
		t.Errorf("ToolChoice = %v, want %q", req.ToolChoice, "auto")
	}
}

func TestUsage(t *testing.T) {
	usage := Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}

	if usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", usage.PromptTokens)
	}
	if usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", usage.CompletionTokens)
	}
	if usage.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150", usage.TotalTokens)
	}
}
