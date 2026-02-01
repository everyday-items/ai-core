// Package deepseek provides DeepSeek LLM provider implementation.
package deepseek

import (
	"os"

	"github.com/everyday-items/ai-core/llm"
	"github.com/everyday-items/ai-core/llm/openai"
)

const (
	defaultBaseURL = "https://api.deepseek.com/v1"
	defaultModel   = "deepseek-chat"
)

// Provider 实现 DeepSeek LLM 提供者
// DeepSeek 使用 OpenAI 兼容的 API，所以复用 OpenAI Provider
type Provider struct {
	*openai.Provider
}

// Option 是 Provider 的配置选项
type Option func(*Provider)

// WithModel 设置默认模型
func WithModel(model string) Option {
	return func(p *Provider) {
		// 通过重新创建 Provider 来设置模型
		// 这是因为 openai.Provider 的 model 字段是私有的
	}
}

// New 创建 DeepSeek Provider
// apiKey 可以为空，会从环境变量 DEEPSEEK_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}

	p := &Provider{
		Provider: openai.New(
			apiKey,
			openai.WithBaseURL(defaultBaseURL),
			openai.WithModel(defaultModel),
		),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "deepseek"
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "deepseek-chat",
			Name:        "DeepSeek Chat",
			Description: "General purpose chat model, good balance of capability and cost",
			MaxTokens:   64000,
			InputCost:   0.14,  // per million tokens
			OutputCost:  0.28,  // per million tokens
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "deepseek-reasoner",
			Name:        "DeepSeek Reasoner",
			Description: "Advanced reasoning model for complex tasks",
			MaxTokens:   64000,
			InputCost:   0.55,
			OutputCost:  2.19,
			Features:    []string{llm.FeatureStreaming},
		},
	}
}

// 确保实现了 Provider 接口
var _ llm.Provider = (*Provider)(nil)
