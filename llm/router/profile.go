// Package router 提供多 Provider LLM 路由功能
//
// 本文件定义模型能力档案（ModelProfile），用于描述各模型在不同任务上的能力表现。
// 档案信息用于智能路由决策，帮助选择最适合特定任务的模型。
package router

// ModelProfile 模型能力档案
// 描述模型在各种任务和复杂度上的表现
type ModelProfile struct {
	// ID 模型标识符
	ID string

	// Provider 提供者名称
	Provider string

	// DisplayName 显示名称
	DisplayName string

	// Description 模型描述
	Description string

	// Capabilities 支持的能力列表
	// 如: vision, functions, streaming, json_mode, embedding
	Capabilities []string

	// TaskScores 各任务类型的得分 (0-1)
	// 分数越高表示该模型越适合该类型任务
	TaskScores map[TaskType]float64

	// ComplexityScores 各复杂度的得分 (0-1)
	// 分数越高表示该模型越适合该复杂度的任务
	ComplexityScores map[TaskComplexity]float64

	// AverageLatencyMs 平均响应延迟（毫秒）
	AverageLatencyMs int

	// InputCostPerMillion 输入成本（每百万 Token，美元）
	InputCostPerMillion float64

	// OutputCostPerMillion 输出成本（每百万 Token，美元）
	OutputCostPerMillion float64

	// MaxContextLength 最大上下文长度
	MaxContextLength int

	// QualityTier 质量等级 (1-5, 5 最高)
	QualityTier int

	// SpeedTier 速度等级 (1-5, 5 最快)
	SpeedTier int

	// CostTier 成本等级 (1-5, 1 最便宜)
	CostTier int

	// Strengths 模型优势描述
	Strengths []string

	// Weaknesses 模型劣势描述
	Weaknesses []string

	// RecommendedFor 推荐用途
	RecommendedFor []string

	// NotRecommendedFor 不推荐用途
	NotRecommendedFor []string
}

// HasCapability 检查是否具有某能力
func (p *ModelProfile) HasCapability(cap string) bool {
	for _, c := range p.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// GetTaskScore 获取任务得分，不存在则返回默认值
func (p *ModelProfile) GetTaskScore(taskType TaskType) float64 {
	if score, ok := p.TaskScores[taskType]; ok {
		return score
	}
	return 0.5 // 默认中等得分
}

// GetComplexityScore 获取复杂度得分，不存在则返回默认值
func (p *ModelProfile) GetComplexityScore(complexity TaskComplexity) float64 {
	if score, ok := p.ComplexityScores[complexity]; ok {
		return score
	}
	return 0.5
}

// TotalCostPerMillion 总成本（每百万 Token）
func (p *ModelProfile) TotalCostPerMillion() float64 {
	return p.InputCostPerMillion + p.OutputCostPerMillion
}

// ============== 默认模型档案 ==============

// GetDefaultProfiles 获取默认模型档案
// 包含主流 LLM 模型的能力档案
func GetDefaultProfiles() map[string]*ModelProfile {
	return map[string]*ModelProfile{
		// ===== OpenAI 模型 =====
		"gpt-4o": {
			ID:          "gpt-4o",
			Provider:    "openai",
			DisplayName: "GPT-4o",
			Description: "OpenAI 旗舰多模态模型，性能与 GPT-4 Turbo 相当，成本更低",
			Capabilities: []string{
				"vision", "functions", "streaming", "json_mode",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.95,
				TaskTypeReasoning:     0.95,
				TaskTypeCoding:        0.90,
				TaskTypeAnalysis:      0.92,
				TaskTypeSummarization: 0.90,
				TaskTypeExtraction:    0.88,
				TaskTypeCreative:      0.90,
				TaskTypeTranslation:   0.92,
				TaskTypeMath:          0.88,
				TaskTypeVision:        0.95,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.95,
				ComplexityMedium:  0.95,
				ComplexityComplex: 0.90,
				ComplexityExpert:  0.85,
			},
			AverageLatencyMs:     1500,
			InputCostPerMillion:  2.5,
			OutputCostPerMillion: 10.0,
			MaxContextLength:     128000,
			QualityTier:          5,
			SpeedTier:            4,
			CostTier:             3,
			Strengths:            []string{"多模态", "推理能力强", "上下文长"},
			RecommendedFor:       []string{"复杂对话", "图像理解", "代码生成"},
		},

		"gpt-4o-mini": {
			ID:          "gpt-4o-mini",
			Provider:    "openai",
			DisplayName: "GPT-4o Mini",
			Description: "GPT-4o 的轻量版本，成本效益高，适合简单任务",
			Capabilities: []string{
				"vision", "functions", "streaming", "json_mode",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.88,
				TaskTypeReasoning:     0.80,
				TaskTypeCoding:        0.82,
				TaskTypeAnalysis:      0.78,
				TaskTypeSummarization: 0.85,
				TaskTypeExtraction:    0.85,
				TaskTypeCreative:      0.80,
				TaskTypeTranslation:   0.85,
				TaskTypeMath:          0.75,
				TaskTypeVision:        0.80,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.95,
				ComplexityMedium:  0.85,
				ComplexityComplex: 0.70,
				ComplexityExpert:  0.55,
			},
			AverageLatencyMs:     800,
			InputCostPerMillion:  0.15,
			OutputCostPerMillion: 0.6,
			MaxContextLength:     128000,
			QualityTier:          4,
			SpeedTier:            5,
			CostTier:             1,
			Strengths:            []string{"速度快", "成本低", "上下文长"},
			RecommendedFor:       []string{"简单对话", "快速响应", "大批量处理"},
		},

		"gpt-4-turbo": {
			ID:          "gpt-4-turbo",
			Provider:    "openai",
			DisplayName: "GPT-4 Turbo",
			Description: "GPT-4 的增强版本，支持更长上下文",
			Capabilities: []string{
				"vision", "functions", "streaming", "json_mode",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.93,
				TaskTypeReasoning:     0.95,
				TaskTypeCoding:        0.92,
				TaskTypeAnalysis:      0.93,
				TaskTypeSummarization: 0.90,
				TaskTypeExtraction:    0.88,
				TaskTypeCreative:      0.92,
				TaskTypeTranslation:   0.90,
				TaskTypeMath:          0.90,
				TaskTypeVision:        0.90,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.95,
				ComplexityComplex: 0.92,
				ComplexityExpert:  0.88,
			},
			AverageLatencyMs:     2000,
			InputCostPerMillion:  10.0,
			OutputCostPerMillion: 30.0,
			MaxContextLength:     128000,
			QualityTier:          5,
			SpeedTier:            3,
			CostTier:             4,
			Strengths:            []string{"推理能力强", "代码能力好"},
			RecommendedFor:       []string{"复杂推理", "代码审查", "深度分析"},
		},

		"o1": {
			ID:          "o1",
			Provider:    "openai",
			DisplayName: "o1",
			Description: "OpenAI 推理模型，专为复杂推理任务优化",
			Capabilities: []string{
				"streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.75,
				TaskTypeReasoning:     0.98,
				TaskTypeCoding:        0.95,
				TaskTypeAnalysis:      0.92,
				TaskTypeSummarization: 0.80,
				TaskTypeExtraction:    0.85,
				TaskTypeCreative:      0.70,
				TaskTypeTranslation:   0.80,
				TaskTypeMath:          0.98,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.70,
				ComplexityMedium:  0.85,
				ComplexityComplex: 0.95,
				ComplexityExpert:  0.98,
			},
			AverageLatencyMs:     10000,
			InputCostPerMillion:  15.0,
			OutputCostPerMillion: 60.0,
			MaxContextLength:     200000,
			QualityTier:          5,
			SpeedTier:            1,
			CostTier:             5,
			Strengths:            []string{"顶级推理", "数学能力", "复杂问题"},
			RecommendedFor:       []string{"复杂推理", "数学证明", "算法设计"},
			NotRecommendedFor:    []string{"简单对话", "快速响应"},
		},

		"o1-mini": {
			ID:          "o1-mini",
			Provider:    "openai",
			DisplayName: "o1-mini",
			Description: "o1 的轻量版本，推理能力强但成本更低",
			Capabilities: []string{
				"streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.70,
				TaskTypeReasoning:     0.92,
				TaskTypeCoding:        0.90,
				TaskTypeAnalysis:      0.85,
				TaskTypeSummarization: 0.75,
				TaskTypeExtraction:    0.80,
				TaskTypeCreative:      0.65,
				TaskTypeTranslation:   0.75,
				TaskTypeMath:          0.92,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.75,
				ComplexityMedium:  0.88,
				ComplexityComplex: 0.90,
				ComplexityExpert:  0.85,
			},
			AverageLatencyMs:     5000,
			InputCostPerMillion:  3.0,
			OutputCostPerMillion: 12.0,
			MaxContextLength:     128000,
			QualityTier:          4,
			SpeedTier:            2,
			CostTier:             3,
			Strengths:            []string{"推理能力", "性价比"},
			RecommendedFor:       []string{"中等复杂度推理", "编程任务"},
		},

		// ===== DeepSeek 模型 =====
		"deepseek-chat": {
			ID:          "deepseek-chat",
			Provider:    "deepseek",
			DisplayName: "DeepSeek Chat",
			Description: "DeepSeek 对话模型，编程能力出色，性价比极高",
			Capabilities: []string{
				"functions", "streaming", "json_mode",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.90,
				TaskTypeReasoning:     0.88,
				TaskTypeCoding:        0.95,
				TaskTypeAnalysis:      0.85,
				TaskTypeSummarization: 0.85,
				TaskTypeExtraction:    0.88,
				TaskTypeCreative:      0.82,
				TaskTypeTranslation:   0.88,
				TaskTypeMath:          0.90,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.90,
				ComplexityComplex: 0.85,
				ComplexityExpert:  0.75,
			},
			AverageLatencyMs:     1200,
			InputCostPerMillion:  0.14,
			OutputCostPerMillion: 0.28,
			MaxContextLength:     64000,
			QualityTier:          4,
			SpeedTier:            4,
			CostTier:             1,
			Strengths:            []string{"编程能力强", "成本极低", "中文能力好"},
			RecommendedFor:       []string{"代码生成", "代码审查", "中文对话"},
		},

		"deepseek-reasoner": {
			ID:          "deepseek-reasoner",
			Provider:    "deepseek",
			DisplayName: "DeepSeek Reasoner",
			Description: "DeepSeek 推理模型，专注复杂推理任务",
			Capabilities: []string{
				"streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.75,
				TaskTypeReasoning:     0.95,
				TaskTypeCoding:        0.92,
				TaskTypeAnalysis:      0.90,
				TaskTypeSummarization: 0.80,
				TaskTypeExtraction:    0.85,
				TaskTypeCreative:      0.70,
				TaskTypeTranslation:   0.80,
				TaskTypeMath:          0.95,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.75,
				ComplexityMedium:  0.85,
				ComplexityComplex: 0.92,
				ComplexityExpert:  0.90,
			},
			AverageLatencyMs:     8000,
			InputCostPerMillion:  0.55,
			OutputCostPerMillion: 2.19,
			MaxContextLength:     64000,
			QualityTier:          5,
			SpeedTier:            2,
			CostTier:             1,
			Strengths:            []string{"推理能力强", "性价比极高"},
			RecommendedFor:       []string{"复杂推理", "数学问题", "编程"},
		},

		// ===== Anthropic 模型 =====
		"claude-3-5-sonnet-20241022": {
			ID:          "claude-3-5-sonnet-20241022",
			Provider:    "anthropic",
			DisplayName: "Claude 3.5 Sonnet",
			Description: "Anthropic 最新模型，综合能力强",
			Capabilities: []string{
				"vision", "functions", "streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.95,
				TaskTypeReasoning:     0.93,
				TaskTypeCoding:        0.95,
				TaskTypeAnalysis:      0.93,
				TaskTypeSummarization: 0.92,
				TaskTypeExtraction:    0.90,
				TaskTypeCreative:      0.95,
				TaskTypeTranslation:   0.90,
				TaskTypeMath:          0.88,
				TaskTypeVision:        0.92,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.95,
				ComplexityMedium:  0.95,
				ComplexityComplex: 0.92,
				ComplexityExpert:  0.88,
			},
			AverageLatencyMs:     1500,
			InputCostPerMillion:  3.0,
			OutputCostPerMillion: 15.0,
			MaxContextLength:     200000,
			QualityTier:          5,
			SpeedTier:            4,
			CostTier:             3,
			Strengths:            []string{"创意写作", "代码能力", "安全性"},
			RecommendedFor:       []string{"创意写作", "代码生成", "深度分析"},
		},

		"claude-3-opus-20240229": {
			ID:          "claude-3-opus-20240229",
			Provider:    "anthropic",
			DisplayName: "Claude 3 Opus",
			Description: "Anthropic 旗舰模型，顶级能力",
			Capabilities: []string{
				"vision", "functions", "streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.95,
				TaskTypeReasoning:     0.95,
				TaskTypeCoding:        0.93,
				TaskTypeAnalysis:      0.95,
				TaskTypeSummarization: 0.93,
				TaskTypeExtraction:    0.90,
				TaskTypeCreative:      0.98,
				TaskTypeTranslation:   0.92,
				TaskTypeMath:          0.90,
				TaskTypeVision:        0.93,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.95,
				ComplexityComplex: 0.95,
				ComplexityExpert:  0.92,
			},
			AverageLatencyMs:     3000,
			InputCostPerMillion:  15.0,
			OutputCostPerMillion: 75.0,
			MaxContextLength:     200000,
			QualityTier:          5,
			SpeedTier:            2,
			CostTier:             5,
			Strengths:            []string{"创意写作", "深度分析", "复杂任务"},
			RecommendedFor:       []string{"高难度任务", "创意写作", "研究分析"},
		},

		"claude-3-haiku-20240307": {
			ID:          "claude-3-haiku-20240307",
			Provider:    "anthropic",
			DisplayName: "Claude 3 Haiku",
			Description: "Anthropic 轻量模型，速度快成本低",
			Capabilities: []string{
				"vision", "functions", "streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.85,
				TaskTypeReasoning:     0.78,
				TaskTypeCoding:        0.80,
				TaskTypeAnalysis:      0.75,
				TaskTypeSummarization: 0.85,
				TaskTypeExtraction:    0.85,
				TaskTypeCreative:      0.80,
				TaskTypeTranslation:   0.82,
				TaskTypeMath:          0.72,
				TaskTypeVision:        0.78,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.95,
				ComplexityMedium:  0.82,
				ComplexityComplex: 0.65,
				ComplexityExpert:  0.50,
			},
			AverageLatencyMs:     500,
			InputCostPerMillion:  0.25,
			OutputCostPerMillion: 1.25,
			MaxContextLength:     200000,
			QualityTier:          3,
			SpeedTier:            5,
			CostTier:             1,
			Strengths:            []string{"速度极快", "成本低"},
			RecommendedFor:       []string{"简单任务", "快速响应", "大批量"},
		},

		// ===== Google 模型 =====
		"gemini-1.5-pro": {
			ID:          "gemini-1.5-pro",
			Provider:    "google",
			DisplayName: "Gemini 1.5 Pro",
			Description: "Google 旗舰模型，超长上下文",
			Capabilities: []string{
				"vision", "functions", "streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.92,
				TaskTypeReasoning:     0.90,
				TaskTypeCoding:        0.88,
				TaskTypeAnalysis:      0.90,
				TaskTypeSummarization: 0.92,
				TaskTypeExtraction:    0.88,
				TaskTypeCreative:      0.88,
				TaskTypeTranslation:   0.90,
				TaskTypeMath:          0.85,
				TaskTypeVision:        0.92,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.92,
				ComplexityComplex: 0.88,
				ComplexityExpert:  0.82,
			},
			AverageLatencyMs:     2000,
			InputCostPerMillion:  1.25,
			OutputCostPerMillion: 5.0,
			MaxContextLength:     2000000,
			QualityTier:          5,
			SpeedTier:            3,
			CostTier:             2,
			Strengths:            []string{"超长上下文", "多模态"},
			RecommendedFor:       []string{"长文档处理", "视频理解"},
		},

		"gemini-1.5-flash": {
			ID:          "gemini-1.5-flash",
			Provider:    "google",
			DisplayName: "Gemini 1.5 Flash",
			Description: "Google 快速模型，平衡性能与速度",
			Capabilities: []string{
				"vision", "functions", "streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.85,
				TaskTypeReasoning:     0.82,
				TaskTypeCoding:        0.80,
				TaskTypeAnalysis:      0.82,
				TaskTypeSummarization: 0.85,
				TaskTypeExtraction:    0.85,
				TaskTypeCreative:      0.80,
				TaskTypeTranslation:   0.85,
				TaskTypeMath:          0.78,
				TaskTypeVision:        0.85,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.95,
				ComplexityMedium:  0.85,
				ComplexityComplex: 0.72,
				ComplexityExpert:  0.60,
			},
			AverageLatencyMs:     800,
			InputCostPerMillion:  0.075,
			OutputCostPerMillion: 0.3,
			MaxContextLength:     1000000,
			QualityTier:          4,
			SpeedTier:            5,
			CostTier:             1,
			Strengths:            []string{"速度快", "长上下文", "成本低"},
			RecommendedFor:       []string{"简单任务", "大批量处理"},
		},

		// ===== 通义千问 =====
		"qwen-max": {
			ID:          "qwen-max",
			Provider:    "qwen",
			DisplayName: "通义千问 Max",
			Description: "阿里云旗舰模型，中文能力强",
			Capabilities: []string{
				"functions", "streaming", "json_mode",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.92,
				TaskTypeReasoning:     0.88,
				TaskTypeCoding:        0.88,
				TaskTypeAnalysis:      0.88,
				TaskTypeSummarization: 0.90,
				TaskTypeExtraction:    0.88,
				TaskTypeCreative:      0.90,
				TaskTypeTranslation:   0.92,
				TaskTypeMath:          0.85,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.90,
				ComplexityComplex: 0.85,
				ComplexityExpert:  0.78,
			},
			AverageLatencyMs:     1500,
			InputCostPerMillion:  2.0,
			OutputCostPerMillion: 6.0,
			MaxContextLength:     32000,
			QualityTier:          4,
			SpeedTier:            4,
			CostTier:             2,
			Strengths:            []string{"中文能力强", "翻译好"},
			RecommendedFor:       []string{"中文对话", "翻译", "创意写作"},
		},

		"qwen-turbo": {
			ID:          "qwen-turbo",
			Provider:    "qwen",
			DisplayName: "通义千问 Turbo",
			Description: "阿里云快速模型，性价比高",
			Capabilities: []string{
				"functions", "streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.85,
				TaskTypeReasoning:     0.78,
				TaskTypeCoding:        0.80,
				TaskTypeAnalysis:      0.78,
				TaskTypeSummarization: 0.85,
				TaskTypeExtraction:    0.82,
				TaskTypeCreative:      0.82,
				TaskTypeTranslation:   0.85,
				TaskTypeMath:          0.75,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.82,
				ComplexityComplex: 0.68,
				ComplexityExpert:  0.55,
			},
			AverageLatencyMs:     600,
			InputCostPerMillion:  0.3,
			OutputCostPerMillion: 0.6,
			MaxContextLength:     131072,
			QualityTier:          3,
			SpeedTier:            5,
			CostTier:             1,
			Strengths:            []string{"速度快", "成本低", "长上下文"},
			RecommendedFor:       []string{"简单对话", "快速响应"},
		},

		// ===== 本地模型 (Ollama) =====
		"llama3.1:70b": {
			ID:          "llama3.1:70b",
			Provider:    "ollama",
			DisplayName: "Llama 3.1 70B",
			Description: "Meta 开源大模型，本地部署",
			Capabilities: []string{
				"streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.88,
				TaskTypeReasoning:     0.85,
				TaskTypeCoding:        0.85,
				TaskTypeAnalysis:      0.82,
				TaskTypeSummarization: 0.85,
				TaskTypeExtraction:    0.82,
				TaskTypeCreative:      0.85,
				TaskTypeTranslation:   0.82,
				TaskTypeMath:          0.78,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.90,
				ComplexityMedium:  0.85,
				ComplexityComplex: 0.78,
				ComplexityExpert:  0.68,
			},
			AverageLatencyMs:     3000,
			InputCostPerMillion:  0.0,
			OutputCostPerMillion: 0.0,
			MaxContextLength:     128000,
			QualityTier:          4,
			SpeedTier:            2,
			CostTier:             1,
			Strengths:            []string{"本地部署", "隐私安全", "无 API 成本"},
			RecommendedFor:       []string{"隐私敏感场景", "离线使用"},
		},

		"llama3.1:8b": {
			ID:          "llama3.1:8b",
			Provider:    "ollama",
			DisplayName: "Llama 3.1 8B",
			Description: "Meta 开源轻量模型，本地部署快",
			Capabilities: []string{
				"streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.78,
				TaskTypeReasoning:     0.72,
				TaskTypeCoding:        0.75,
				TaskTypeAnalysis:      0.70,
				TaskTypeSummarization: 0.78,
				TaskTypeExtraction:    0.75,
				TaskTypeCreative:      0.75,
				TaskTypeTranslation:   0.72,
				TaskTypeMath:          0.68,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.88,
				ComplexityMedium:  0.75,
				ComplexityComplex: 0.58,
				ComplexityExpert:  0.45,
			},
			AverageLatencyMs:     800,
			InputCostPerMillion:  0.0,
			OutputCostPerMillion: 0.0,
			MaxContextLength:     128000,
			QualityTier:          3,
			SpeedTier:            4,
			CostTier:             1,
			Strengths:            []string{"本地快速", "资源占用低"},
			RecommendedFor:       []string{"简单任务", "边缘设备"},
		},

		"qwen2.5:72b": {
			ID:          "qwen2.5:72b",
			Provider:    "ollama",
			DisplayName: "Qwen 2.5 72B",
			Description: "通义千问开源大模型，本地部署，中文强",
			Capabilities: []string{
				"streaming",
			},
			TaskScores: map[TaskType]float64{
				TaskTypeChat:          0.90,
				TaskTypeReasoning:     0.85,
				TaskTypeCoding:        0.88,
				TaskTypeAnalysis:      0.85,
				TaskTypeSummarization: 0.88,
				TaskTypeExtraction:    0.85,
				TaskTypeCreative:      0.88,
				TaskTypeTranslation:   0.90,
				TaskTypeMath:          0.82,
				TaskTypeVision:        0.50,
			},
			ComplexityScores: map[TaskComplexity]float64{
				ComplexitySimple:  0.92,
				ComplexityMedium:  0.88,
				ComplexityComplex: 0.80,
				ComplexityExpert:  0.70,
			},
			AverageLatencyMs:     3500,
			InputCostPerMillion:  0.0,
			OutputCostPerMillion: 0.0,
			MaxContextLength:     131072,
			QualityTier:          4,
			SpeedTier:            2,
			CostTier:             1,
			Strengths:            []string{"中文强", "本地部署", "编程能力"},
			RecommendedFor:       []string{"中文场景", "隐私敏感"},
		},
	}
}

// ============== Profile 构建器 ==============

// ProfileBuilder 模型档案构建器
type ProfileBuilder struct {
	profile *ModelProfile
}

// NewProfileBuilder 创建档案构建器
func NewProfileBuilder(id, provider string) *ProfileBuilder {
	return &ProfileBuilder{
		profile: &ModelProfile{
			ID:               id,
			Provider:         provider,
			TaskScores:       make(map[TaskType]float64),
			ComplexityScores: make(map[TaskComplexity]float64),
			Capabilities:     make([]string, 0),
		},
	}
}

// WithDisplayName 设置显示名称
func (b *ProfileBuilder) WithDisplayName(name string) *ProfileBuilder {
	b.profile.DisplayName = name
	return b
}

// WithDescription 设置描述
func (b *ProfileBuilder) WithDescription(desc string) *ProfileBuilder {
	b.profile.Description = desc
	return b
}

// WithCapabilities 设置能力
func (b *ProfileBuilder) WithCapabilities(caps ...string) *ProfileBuilder {
	b.profile.Capabilities = append(b.profile.Capabilities, caps...)
	return b
}

// WithTaskScore 设置任务得分
func (b *ProfileBuilder) WithTaskScore(taskType TaskType, score float64) *ProfileBuilder {
	b.profile.TaskScores[taskType] = score
	return b
}

// WithComplexityScore 设置复杂度得分
func (b *ProfileBuilder) WithComplexityScore(complexity TaskComplexity, score float64) *ProfileBuilder {
	b.profile.ComplexityScores[complexity] = score
	return b
}

// WithLatency 设置延迟
func (b *ProfileBuilder) WithLatency(ms int) *ProfileBuilder {
	b.profile.AverageLatencyMs = ms
	return b
}

// WithCost 设置成本
func (b *ProfileBuilder) WithCost(input, output float64) *ProfileBuilder {
	b.profile.InputCostPerMillion = input
	b.profile.OutputCostPerMillion = output
	return b
}

// WithContextLength 设置上下文长度
func (b *ProfileBuilder) WithContextLength(length int) *ProfileBuilder {
	b.profile.MaxContextLength = length
	return b
}

// WithTiers 设置等级
func (b *ProfileBuilder) WithTiers(quality, speed, cost int) *ProfileBuilder {
	b.profile.QualityTier = quality
	b.profile.SpeedTier = speed
	b.profile.CostTier = cost
	return b
}

// Build 构建档案
func (b *ProfileBuilder) Build() *ModelProfile {
	return b.profile
}
