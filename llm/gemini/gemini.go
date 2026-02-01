// Package gemini provides Google Gemini LLM provider implementation.
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/everyday-items/ai-core/llm"
	"github.com/everyday-items/ai-core/streamx"
)

const (
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	defaultModel   = "gemini-2.0-flash"
)

// Provider 实现 Google Gemini LLM 提供者
type Provider struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// Option 是 Provider 的配置选项
type Option func(*Provider)

// WithBaseURL 设置 API 基础 URL
func WithBaseURL(url string) Option {
	return func(p *Provider) {
		p.baseURL = url
	}
}

// WithModel 设置默认模型
func WithModel(model string) Option {
	return func(p *Provider) {
		p.model = model
	}
}

// WithHTTPClient 设置 HTTP 客户端
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// New 创建 Gemini Provider
// apiKey 可以为空，会从环境变量 GOOGLE_API_KEY 或 GEMINI_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
	}

	p := &Provider{
		apiKey:     apiKey,
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: http.DefaultClient,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "gemini"
}

// Complete 执行非流式补全请求
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", p.baseURL, req.Model, p.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return p.parseResponse(&result, req.Model), nil
}

// Stream 执行流式补全请求
func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?key=%s&alt=sse", p.baseURL, req.Model, p.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("gemini api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.GeminiFormat), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "gemini-2.0-flash",
			Name:        "Gemini 2.0 Flash",
			Description: "Next-gen fast and versatile model",
			MaxTokens:   1048576,
			InputCost:   0.10,
			OutputCost:  0.40,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-2.0-flash-thinking",
			Name:        "Gemini 2.0 Flash Thinking",
			Description: "Fast model with enhanced reasoning",
			MaxTokens:   1048576,
			InputCost:   0.10,
			OutputCost:  0.40,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-1.5-pro",
			Name:        "Gemini 1.5 Pro",
			Description: "Most capable model for complex tasks",
			MaxTokens:   2097152,
			InputCost:   1.25,
			OutputCost:  5.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-1.5-flash",
			Name:        "Gemini 1.5 Flash",
			Description: "Fast and efficient for most tasks",
			MaxTokens:   1048576,
			InputCost:   0.075,
			OutputCost:  0.30,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-1.5-flash-8b",
			Name:        "Gemini 1.5 Flash 8B",
			Description: "Lightweight and cost-effective",
			MaxTokens:   1048576,
			InputCost:   0.0375,
			OutputCost:  0.15,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
	}
}

// CountTokens 计算消息的 Token 数量（简化实现）
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	// 简化估算：约 4 字符一个 token
	var total int
	for _, msg := range messages {
		total += len(msg.Content) / 4
	}
	return total, nil
}

// buildRequestBody 构建请求体
// Gemini 使用独特的 API 格式
func (p *Provider) buildRequestBody(req llm.CompletionRequest) ([]byte, error) {
	// 转换消息格式
	var contents []geminiContent
	var systemInstruction *geminiContent

	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			systemInstruction = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
			continue
		}

		content := geminiContent{
			Role:  convertRole(msg.Role),
			Parts: []geminiPart{{Text: msg.Content}},
		}
		contents = append(contents, content)
	}

	payload := map[string]any{
		"contents": contents,
	}

	if systemInstruction != nil {
		payload["systemInstruction"] = systemInstruction
	}

	// 生成配置
	generationConfig := make(map[string]any)
	if req.MaxTokens > 0 {
		generationConfig["maxOutputTokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		generationConfig["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		generationConfig["topP"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		generationConfig["stopSequences"] = req.Stop
	}

	if len(generationConfig) > 0 {
		payload["generationConfig"] = generationConfig
	}

	// 工具支持
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0)
		functionDeclarations := make([]map[string]any, len(req.Tools))
		for i, tool := range req.Tools {
			functionDeclarations[i] = map[string]any{
				"name":        tool.Function.Name,
				"description": tool.Function.Description,
				"parameters":  tool.Function.Parameters,
			}
		}
		tools = append(tools, map[string]any{
			"functionDeclarations": functionDeclarations,
		})
		payload["tools"] = tools
	}

	return json.Marshal(payload)
}

// convertRole 转换角色名称
func convertRole(role llm.Role) string {
	switch role {
	case llm.RoleUser:
		return "user"
	case llm.RoleAssistant:
		return "model"
	default:
		return "user"
	}
}

// Gemini 数据结构
type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text         string             `json:"text,omitempty"`
	FunctionCall *geminiFunctionCall `json:"functionCall,omitempty"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// Gemini API 响应结构
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
			Role  string       `json:"role"`
		} `json:"content"`
		FinishReason  string `json:"finishReason"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *geminiResponse, model string) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		Model: model,
		Usage: llm.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		},
	}

	if len(resp.Candidates) > 0 {
		candidate := resp.Candidates[0]
		result.FinishReason = candidate.FinishReason

		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				result.Content += part.Text
			}
			if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
					ID:        fmt.Sprintf("call_%s", part.FunctionCall.Name),
					Type:      "function",
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				})
			}
		}
	}

	return result
}

// Embed 生成文本的向量嵌入
func (p *Provider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return p.EmbedWithModel(ctx, "text-embedding-004", texts)
}

// EmbedWithModel 使用指定模型生成嵌入
func (p *Provider) EmbedWithModel(ctx context.Context, model string, texts []string) ([][]float32, error) {
	// 批量嵌入
	requests := make([]map[string]any, len(texts))
	for i, text := range texts {
		requests[i] = map[string]any{
			"model": fmt.Sprintf("models/%s", model),
			"content": map[string]any{
				"parts": []map[string]string{{"text": text}},
			},
		}
	}

	payload := map[string]any{
		"requests": requests,
	}
	body, _ := json.Marshal(payload)

	url := fmt.Sprintf("%s/models/%s:batchEmbedContents?key=%s", p.baseURL, model, p.apiKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini embed error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result struct {
		Embeddings []struct {
			Values []float32 `json:"values"`
		} `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(result.Embeddings))
	for i, e := range result.Embeddings {
		embeddings[i] = e.Values
	}

	return embeddings, nil
}

// 确保实现了 Provider 和 EmbeddingProvider 接口
var _ llm.Provider = (*Provider)(nil)
var _ llm.EmbeddingProvider = (*Provider)(nil)
