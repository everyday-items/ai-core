package router

import (
	"context"
	"testing"
	"time"

	"github.com/everyday-items/ai-core/llm"
)

// mockProvider 模拟 LLM Provider
type mockProvider struct {
	name   string
	models []llm.ModelInfo
}

func (m *mockProvider) Name() string {
	return m.name
}

func (m *mockProvider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return &llm.CompletionResponse{
		ID:      "mock-response",
		Model:   req.Model,
		Content: "Mock response",
		Usage: llm.Usage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}, nil
}

func (m *mockProvider) Stream(ctx context.Context, req llm.CompletionRequest) (*llm.Stream, error) {
	return nil, nil
}

func (m *mockProvider) Models() []llm.ModelInfo {
	return m.models
}

func (m *mockProvider) CountTokens(messages []llm.Message) (int, error) {
	return 100, nil
}

// TestNewSmartRouter 测试创建智能路由器
func TestNewSmartRouter(t *testing.T) {
	baseRouter := New()
	smartRouter := NewSmartRouter(baseRouter)

	if smartRouter == nil {
		t.Fatal("NewSmartRouter 返回 nil")
	}

	if smartRouter.Router == nil {
		t.Error("基础路由器为 nil")
	}

	if smartRouter.classifier == nil {
		t.Error("分类器为 nil")
	}

	if len(smartRouter.modelProfiles) == 0 {
		t.Error("默认模型档案未加载")
	}
}

// TestSmartRouterRoute 测试智能路由
func TestSmartRouterRoute(t *testing.T) {
	// 创建模拟 Provider
	openaiProvider := &mockProvider{
		name: "openai",
		models: []llm.ModelInfo{
			{
				ID:          "gpt-4o",
				Name:        "GPT-4o",
				InputCost:   2.5,
				OutputCost:  10.0,
				MaxTokens:   128000,
				Features:    []string{"vision", "functions", "streaming"},
			},
			{
				ID:          "gpt-4o-mini",
				Name:        "GPT-4o Mini",
				InputCost:   0.15,
				OutputCost:  0.6,
				MaxTokens:   128000,
				Features:    []string{"vision", "functions", "streaming"},
			},
		},
	}

	deepseekProvider := &mockProvider{
		name: "deepseek",
		models: []llm.ModelInfo{
			{
				ID:          "deepseek-chat",
				Name:        "DeepSeek Chat",
				InputCost:   0.14,
				OutputCost:  0.28,
				MaxTokens:   64000,
				Features:    []string{"functions", "streaming"},
			},
		},
	}

	// 创建路由器
	baseRouter := New()
	baseRouter.Register("openai", openaiProvider)
	baseRouter.Register("deepseek", deepseekProvider)

	smartRouter := NewSmartRouter(baseRouter)

	// 测试用例
	tests := []struct {
		name       string
		req        llm.CompletionRequest
		routingCtx *RoutingContext
		wantErr    bool
	}{
		{
			name: "编程任务路由",
			req: llm.CompletionRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: "请写一个快速排序算法"},
				},
			},
			routingCtx: NewRoutingContext(TaskTypeCoding, ComplexityMedium),
			wantErr:    false,
		},
		{
			name: "推理任务路由",
			req: llm.CompletionRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: "分析一下这个问题的原因"},
				},
			},
			routingCtx: NewRoutingContext(TaskTypeReasoning, ComplexityComplex),
			wantErr:    false,
		},
		{
			name: "成本优先路由",
			req: llm.CompletionRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: "你好"},
				},
			},
			routingCtx: NewRoutingContext(TaskTypeChat, ComplexitySimple).
				WithPriority(PriorityCost),
			wantErr: false,
		},
		{
			name: "视觉任务路由",
			req: llm.CompletionRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: "描述这张图片"},
				},
			},
			routingCtx: NewRoutingContext(TaskTypeVision, ComplexityMedium).
				RequireVision(),
			wantErr: false,
		},
		{
			name: "自动分类路由",
			req: llm.CompletionRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: "帮我写一个 Python 函数"},
				},
			},
			routingCtx: nil, // 自动分类
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision, err := smartRouter.Route(context.Background(), tt.req, tt.routingCtx)

			if (err != nil) != tt.wantErr {
				t.Errorf("Route() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if decision == nil {
					t.Error("Route() 返回 nil decision")
					return
				}

				if decision.Provider == nil {
					t.Error("decision.Provider 为 nil")
				}

				if decision.ModelID == "" {
					t.Error("decision.ModelID 为空")
				}

				if decision.Score <= 0 {
					t.Errorf("decision.Score = %v, 期望 > 0", decision.Score)
				}

				t.Logf("路由决策: %s (%s), 得分: %.2f, 原因: %s",
					decision.ModelID, decision.ProviderName, decision.Score, decision.Reason)
			}
		})
	}
}

// TestSmartRouterCompleteWithRouting 测试带路由的补全
func TestSmartRouterCompleteWithRouting(t *testing.T) {
	provider := &mockProvider{
		name: "openai",
		models: []llm.ModelInfo{
			{
				ID:          "gpt-4o-mini",
				Name:        "GPT-4o Mini",
				InputCost:   0.15,
				OutputCost:  0.6,
				MaxTokens:   128000,
				Features:    []string{"functions", "streaming"},
			},
		},
	}

	baseRouter := New()
	baseRouter.Register("openai", provider)

	smartRouter := NewSmartRouter(baseRouter)

	req := llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "你好"},
		},
	}

	routingCtx := NewRoutingContext(TaskTypeChat, ComplexitySimple)

	resp, decision, err := smartRouter.CompleteWithRouting(context.Background(), req, routingCtx)
	if err != nil {
		t.Fatalf("CompleteWithRouting() error = %v", err)
	}

	if resp == nil {
		t.Error("响应为 nil")
	}

	if decision == nil {
		t.Error("决策为 nil")
	}

	// 检查路由历史
	history := smartRouter.GetRoutingHistory()
	if len(history) == 0 {
		t.Error("路由历史为空")
	}
}

// TestRuleBasedClassifier 测试基于规则的分类器
func TestRuleBasedClassifier(t *testing.T) {
	classifier := NewRuleBasedClassifier()

	tests := []struct {
		name           string
		content        string
		wantTaskType   TaskType
		wantComplexity TaskComplexity
	}{
		{
			name:           "编程任务",
			content:        "请帮我写一个 Python 函数来排序数组",
			wantTaskType:   TaskTypeCoding,
			wantComplexity: ComplexityMedium,
		},
		{
			name:           "推理任务",
			content:        "Let's think step by step, why does this happen?",
			wantTaskType:   TaskTypeReasoning,
			wantComplexity: ComplexityMedium,
		},
		{
			name:           "数学任务",
			content:        "计算 x^2 + 2x + 1 = 0 的解",
			wantTaskType:   TaskTypeMath,
			wantComplexity: ComplexityMedium,
		},
		{
			name:           "翻译任务",
			content:        "把这段话翻译成英文",
			wantTaskType:   TaskTypeTranslation,
			wantComplexity: ComplexityMedium,
		},
		{
			name:           "摘要任务",
			content:        "请总结一下这篇文章的要点",
			wantTaskType:   TaskTypeSummarization,
			wantComplexity: ComplexityMedium,
		},
		{
			name:           "创意写作",
			content:        "帮我写一个关于太空旅行的故事",
			wantTaskType:   TaskTypeCreative,
			wantComplexity: ComplexityMedium,
		},
		{
			name:           "信息提取",
			content:        "从这段文本中提取所有人名",
			wantTaskType:   TaskTypeExtraction,
			wantComplexity: ComplexityMedium,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := llm.CompletionRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: tt.content},
				},
			}

			taskType, complexity := classifier.Classify(context.Background(), req)

			if taskType != tt.wantTaskType {
				t.Errorf("Classify() taskType = %v, want %v", taskType, tt.wantTaskType)
			}

			t.Logf("分类结果: %s, 复杂度: %s", taskType, complexity)
		})
	}
}

// TestRoutingContext 测试路由上下文构建
func TestRoutingContext(t *testing.T) {
	ctx := NewRoutingContext(TaskTypeCoding, ComplexityMedium).
		WithCapabilities("functions", "streaming").
		WithMaxLatency(5000).
		WithMaxCost(0.01).
		WithPriority(PriorityQuality).
		WithPreferredProviders("openai", "anthropic").
		RequireVision().
		RequireFunctions()

	if ctx.TaskType != TaskTypeCoding {
		t.Errorf("TaskType = %v, want %v", ctx.TaskType, TaskTypeCoding)
	}

	if ctx.Complexity != ComplexityMedium {
		t.Errorf("Complexity = %v, want %v", ctx.Complexity, ComplexityMedium)
	}

	if ctx.Constraints.MaxLatencyMs != 5000 {
		t.Errorf("MaxLatencyMs = %v, want 5000", ctx.Constraints.MaxLatencyMs)
	}

	if ctx.Constraints.MaxCostPerRequest != 0.01 {
		t.Errorf("MaxCostPerRequest = %v, want 0.01", ctx.Constraints.MaxCostPerRequest)
	}

	if ctx.Priority != PriorityQuality {
		t.Errorf("Priority = %v, want %v", ctx.Priority, PriorityQuality)
	}

	if !ctx.Constraints.RequireVision {
		t.Error("RequireVision should be true")
	}

	if !ctx.Constraints.RequireFunctionCalling {
		t.Error("RequireFunctionCalling should be true")
	}
}

// TestModelProfile 测试模型档案
func TestModelProfile(t *testing.T) {
	profiles := GetDefaultProfiles()

	// 测试 GPT-4o 档案
	gpt4o := profiles["gpt-4o"]
	if gpt4o == nil {
		t.Fatal("GPT-4o 档案不存在")
	}

	if gpt4o.ID != "gpt-4o" {
		t.Errorf("ID = %v, want gpt-4o", gpt4o.ID)
	}

	if !gpt4o.HasCapability("vision") {
		t.Error("GPT-4o 应该支持 vision")
	}

	// 测试任务得分
	codingScore := gpt4o.GetTaskScore(TaskTypeCoding)
	if codingScore <= 0 || codingScore > 1 {
		t.Errorf("编程任务得分异常: %v", codingScore)
	}

	// 测试 DeepSeek 档案
	deepseek := profiles["deepseek-chat"]
	if deepseek == nil {
		t.Fatal("DeepSeek 档案不存在")
	}

	// DeepSeek 编程得分应该较高
	if deepseek.GetTaskScore(TaskTypeCoding) < 0.9 {
		t.Errorf("DeepSeek 编程得分应该 >= 0.9, got %v", deepseek.GetTaskScore(TaskTypeCoding))
	}
}

// TestProfileBuilder 测试档案构建器
func TestProfileBuilder(t *testing.T) {
	profile := NewProfileBuilder("custom-model", "custom-provider").
		WithDisplayName("Custom Model").
		WithDescription("A custom test model").
		WithCapabilities("functions", "streaming").
		WithTaskScore(TaskTypeCoding, 0.9).
		WithTaskScore(TaskTypeChat, 0.8).
		WithComplexityScore(ComplexitySimple, 0.95).
		WithComplexityScore(ComplexityComplex, 0.7).
		WithLatency(1000).
		WithCost(1.0, 2.0).
		WithContextLength(32000).
		WithTiers(4, 4, 2).
		Build()

	if profile.ID != "custom-model" {
		t.Errorf("ID = %v, want custom-model", profile.ID)
	}

	if profile.Provider != "custom-provider" {
		t.Errorf("Provider = %v, want custom-provider", profile.Provider)
	}

	if profile.GetTaskScore(TaskTypeCoding) != 0.9 {
		t.Errorf("TaskScore(coding) = %v, want 0.9", profile.GetTaskScore(TaskTypeCoding))
	}

	if profile.AverageLatencyMs != 1000 {
		t.Errorf("AverageLatencyMs = %v, want 1000", profile.AverageLatencyMs)
	}
}

// TestSmartRoutingStats 测试路由统计
func TestSmartRoutingStats(t *testing.T) {
	provider := &mockProvider{
		name: "openai",
		models: []llm.ModelInfo{
			{
				ID:          "gpt-4o-mini",
				Name:        "GPT-4o Mini",
				InputCost:   0.15,
				OutputCost:  0.6,
				MaxTokens:   128000,
				Features:    []string{"functions", "streaming"},
			},
		},
	}

	baseRouter := New()
	baseRouter.Register("openai", provider)

	smartRouter := NewSmartRouter(baseRouter)

	// 执行几次路由
	for i := 0; i < 5; i++ {
		req := llm.CompletionRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "测试消息"},
			},
		}
		_, _, _ = smartRouter.CompleteWithRouting(context.Background(), req, nil)
	}

	stats := smartRouter.GetRoutingStats()

	if stats.TotalRequests != 5 {
		t.Errorf("TotalRequests = %v, want 5", stats.TotalRequests)
	}

	if stats.SuccessfulRequests != 5 {
		t.Errorf("SuccessfulRequests = %v, want 5", stats.SuccessfulRequests)
	}

	if stats.ModelUsage["gpt-4o-mini"] != 5 {
		t.Errorf("ModelUsage[gpt-4o-mini] = %v, want 5", stats.ModelUsage["gpt-4o-mini"])
	}
}

// TestRoutingWithConstraints 测试带约束的路由
func TestRoutingWithConstraints(t *testing.T) {
	// 创建两个 Provider，一个成本高一个成本低
	expensiveProvider := &mockProvider{
		name: "expensive",
		models: []llm.ModelInfo{
			{
				ID:          "expensive-model",
				Name:        "Expensive Model",
				InputCost:   100.0, // 非常高的成本
				OutputCost:  200.0,
				MaxTokens:   128000,
				Features:    []string{"functions", "streaming"},
			},
		},
	}

	cheapProvider := &mockProvider{
		name: "cheap",
		models: []llm.ModelInfo{
			{
				ID:          "cheap-model",
				Name:        "Cheap Model",
				InputCost:   0.1,
				OutputCost:  0.2,
				MaxTokens:   128000,
				Features:    []string{"functions", "streaming"},
			},
		},
	}

	baseRouter := New()
	baseRouter.Register("expensive", expensiveProvider)
	baseRouter.Register("cheap", cheapProvider)

	smartRouter := NewSmartRouter(baseRouter)

	// 注册模型档案
	smartRouter.RegisterProfile("expensive-model", NewProfileBuilder("expensive-model", "expensive").
		WithCost(100.0, 200.0).
		WithTaskScore(TaskTypeChat, 0.9).
		Build())

	smartRouter.RegisterProfile("cheap-model", NewProfileBuilder("cheap-model", "cheap").
		WithCost(0.1, 0.2).
		WithTaskScore(TaskTypeChat, 0.8).
		Build())

	req := llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "你好"},
		},
	}

	// 测试成本优先路由
	routingCtx := NewRoutingContext(TaskTypeChat, ComplexitySimple).
		WithPriority(PriorityCost)

	decision, err := smartRouter.Route(context.Background(), req, routingCtx)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}

	// 成本优先应该选择便宜的模型
	if decision.ModelID != "cheap-model" {
		t.Errorf("成本优先应选择 cheap-model, got %v", decision.ModelID)
	}

	t.Logf("成本优先路由结果: %s, 得分: %.2f", decision.ModelID, decision.Score)
}

// TestAllTaskTypes 测试所有任务类型
func TestAllTaskTypes(t *testing.T) {
	types := AllTaskTypes()

	expectedTypes := []TaskType{
		TaskTypeChat,
		TaskTypeReasoning,
		TaskTypeCoding,
		TaskTypeAnalysis,
		TaskTypeSummarization,
		TaskTypeExtraction,
		TaskTypeCreative,
		TaskTypeTranslation,
		TaskTypeMath,
		TaskTypeVision,
	}

	if len(types) != len(expectedTypes) {
		t.Errorf("AllTaskTypes() 返回 %d 个类型, 期望 %d", len(types), len(expectedTypes))
	}

	for i, tt := range expectedTypes {
		if types[i] != tt {
			t.Errorf("AllTaskTypes()[%d] = %v, want %v", i, types[i], tt)
		}
	}
}

// TestComplexityScore 测试复杂度得分
func TestComplexityScore(t *testing.T) {
	tests := []struct {
		complexity TaskComplexity
		wantScore  float64
	}{
		{ComplexitySimple, 0.25},
		{ComplexityMedium, 0.5},
		{ComplexityComplex, 0.75},
		{ComplexityExpert, 1.0},
	}

	for _, tt := range tests {
		t.Run(string(tt.complexity), func(t *testing.T) {
			score := tt.complexity.Score()
			if score != tt.wantScore {
				t.Errorf("Score() = %v, want %v", score, tt.wantScore)
			}
		})
	}
}

// BenchmarkSmartRouterRoute 基准测试智能路由
func BenchmarkSmartRouterRoute(b *testing.B) {
	provider := &mockProvider{
		name: "openai",
		models: []llm.ModelInfo{
			{ID: "gpt-4o", Features: []string{"vision", "functions"}},
			{ID: "gpt-4o-mini", Features: []string{"vision", "functions"}},
		},
	}

	baseRouter := New()
	baseRouter.Register("openai", provider)

	smartRouter := NewSmartRouter(baseRouter)

	req := llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "请写一个排序算法"},
		},
	}
	routingCtx := NewRoutingContext(TaskTypeCoding, ComplexityMedium)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = smartRouter.Route(context.Background(), req, routingCtx)
	}
}

// BenchmarkRuleBasedClassifier 基准测试规则分类器
func BenchmarkRuleBasedClassifier(b *testing.B) {
	classifier := NewRuleBasedClassifier()
	req := llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "请帮我写一个 Python 函数来实现快速排序算法"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		classifier.Classify(context.Background(), req)
	}
}

// TestSmartRouterConcurrency 测试并发安全性
func TestSmartRouterConcurrency(t *testing.T) {
	provider := &mockProvider{
		name: "openai",
		models: []llm.ModelInfo{
			{ID: "gpt-4o-mini", Features: []string{"functions", "streaming"}},
		},
	}

	baseRouter := New()
	baseRouter.Register("openai", provider)

	smartRouter := NewSmartRouter(baseRouter)

	// 并发执行路由
	const goroutines = 10
	const iterations = 100

	done := make(chan bool, goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			for j := 0; j < iterations; j++ {
				req := llm.CompletionRequest{
					Messages: []llm.Message{
						{Role: llm.RoleUser, Content: "测试消息"},
					},
				}
				_, _, _ = smartRouter.CompleteWithRouting(context.Background(), req, nil)
			}
			done <- true
		}()
	}

	// 等待所有协程完成
	timeout := time.After(30 * time.Second)
	for i := 0; i < goroutines; i++ {
		select {
		case <-done:
		case <-timeout:
			t.Fatal("并发测试超时")
		}
	}

	// 验证统计
	stats := smartRouter.GetRoutingStats()
	expectedRequests := goroutines * iterations
	if stats.TotalRequests != expectedRequests {
		t.Errorf("TotalRequests = %v, want %v", stats.TotalRequests, expectedRequests)
	}
}
