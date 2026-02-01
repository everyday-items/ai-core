// Package qwen provides Alibaba Qwen (通义千问) LLM provider implementation.
package qwen

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
	defaultBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	defaultModel   = "qwen-max"
)

// Provider 实现通义千问 LLM 提供者
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

// New 创建通义千问 Provider
// apiKey 可以为空，会从环境变量 DASHSCOPE_API_KEY 或 QWEN_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("DASHSCOPE_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("QWEN_API_KEY")
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
	return "qwen"
}

// Complete 执行非流式补全请求
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("qwen api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result qwenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return p.parseResponse(&result), nil
}

// Stream 执行流式补全请求
func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("qwen api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	// 通义千问使用 OpenAI 兼容格式
	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.OpenAIFormat), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "qwen-max",
			Name:        "Qwen Max",
			Description: "通义千问最强模型，适合复杂任务",
			MaxTokens:   32768,
			InputCost:   20.00,
			OutputCost:  60.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "qwen-max-longcontext",
			Name:        "Qwen Max Long Context",
			Description: "通义千问长文本模型，支持更长上下文",
			MaxTokens:   30720,
			InputCost:   20.00,
			OutputCost:  60.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "qwen-plus",
			Name:        "Qwen Plus",
			Description: "通义千问平衡模型，性价比高",
			MaxTokens:   131072,
			InputCost:   0.80,
			OutputCost:  2.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "qwen-turbo",
			Name:        "Qwen Turbo",
			Description: "通义千问快速模型，成本最低",
			MaxTokens:   131072,
			InputCost:   0.30,
			OutputCost:  0.60,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "qwen-vl-max",
			Name:        "Qwen VL Max",
			Description: "通义千问视觉模型，支持图像理解",
			MaxTokens:   32768,
			InputCost:   20.00,
			OutputCost:  60.00,
			Features:    []string{llm.FeatureVision, llm.FeatureStreaming},
		},
		{
			ID:          "qwen-vl-plus",
			Name:        "Qwen VL Plus",
			Description: "通义千问视觉模型，性价比版",
			MaxTokens:   8192,
			InputCost:   8.00,
			OutputCost:  8.00,
			Features:    []string{llm.FeatureVision, llm.FeatureStreaming},
		},
	}
}

// CountTokens 计算消息的 Token 数量（简化实现）
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	// 简化估算：中文约 1.5 字符一个 token，英文约 4 字符一个 token
	// 这里使用保守估计
	var total int
	for _, msg := range messages {
		// 假设混合语言，使用 2.5 字符每 token
		total += len(msg.Content) * 2 / 5
	}
	return total, nil
}

// setHeaders 设置请求头
func (p *Provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}

// buildRequestBody 构建请求体
func (p *Provider) buildRequestBody(req llm.CompletionRequest, stream bool) ([]byte, error) {
	payload := map[string]any{
		"model":    req.Model,
		"messages": convertMessages(req.Messages),
		"stream":   stream,
	}

	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		payload["tool_choice"] = req.ToolChoice
	}
	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		payload["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		payload["stop"] = req.Stop
	}

	// 通义千问特有参数
	if req.Metadata != nil {
		if enableSearch, ok := req.Metadata["enable_search"].(bool); ok && enableSearch {
			payload["enable_search"] = true
		}
	}

	return json.Marshal(payload)
}

// convertMessages 转换消息格式
func convertMessages(messages []llm.Message) []map[string]any {
	result := make([]map[string]any, len(messages))
	for i, msg := range messages {
		m := map[string]any{
			"role":    string(msg.Role),
			"content": msg.Content,
		}
		if msg.Name != "" {
			m["name"] = msg.Name
		}
		result[i] = m
	}
	return result
}

// 通义千问 API 响应结构（兼容 OpenAI 格式）
type qwenResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *qwenResponse) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Created: resp.Created,
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		result.Content = choice.Message.Content
		result.FinishReason = choice.FinishReason

		for _, tc := range choice.Message.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
				ID:        tc.ID,
				Type:      tc.Type,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}

	return result
}

// 确保实现了 Provider 接口
var _ llm.Provider = (*Provider)(nil)
