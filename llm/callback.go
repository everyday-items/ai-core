// Package llm 的回调接口定义
//
// Callback 提供 Provider 级别的可观测性钩子，
// 允许上层框架（如 hexagon）注入追踪、日志、指标等功能。
package llm

import "context"

// Callback 定义 Provider 级别的请求回调接口
//
// 通过 WithCallback 中间件注入到 Provider 调用链中，
// 在每次 LLM 请求的开始和结束时触发。
//
// 使用示例:
//
//	callback := &MyCallback{}
//	provider = llm.Chain(provider, llm.WithCallback(callback))
type Callback interface {
	// OnStart 在 LLM 请求开始前调用
	//
	// 参数:
	//   - ctx: 请求上下文
	//   - event: 请求开始事件，包含请求参数
	OnStart(ctx context.Context, event *CallbackStartEvent)

	// OnEnd 在 LLM 请求完成后调用（无论成功或失败）
	//
	// 参数:
	//   - ctx: 请求上下文
	//   - event: 请求结束事件，包含响应和错误信息
	OnEnd(ctx context.Context, event *CallbackEndEvent)
}

// CallbackStartEvent 请求开始事件
type CallbackStartEvent struct {
	// Provider 提供者名称
	Provider string

	// Request 补全请求参数
	Request *CompletionRequest
}

// CallbackEndEvent 请求结束事件
type CallbackEndEvent struct {
	// Provider 提供者名称
	Provider string

	// Request 原始请求参数
	Request *CompletionRequest

	// Response 补全响应（成功时非 nil）
	Response *CompletionResponse

	// Error 请求错误（失败时非 nil）
	Error error

	// DurationMs 请求耗时（毫秒）
	DurationMs int64

	// Stream 是否为流式请求
	Stream bool
}

// CallbackFunc 函数式 Callback 实现
//
// 便捷的 Callback 包装器，允许只实现需要的钩子。
type CallbackFunc struct {
	// StartFn 请求开始回调函数
	StartFn func(ctx context.Context, event *CallbackStartEvent)

	// EndFn 请求结束回调函数
	EndFn func(ctx context.Context, event *CallbackEndEvent)
}

// OnStart 实现 Callback 接口
func (f *CallbackFunc) OnStart(ctx context.Context, event *CallbackStartEvent) {
	if f.StartFn != nil {
		f.StartFn(ctx, event)
	}
}

// OnEnd 实现 Callback 接口
func (f *CallbackFunc) OnEnd(ctx context.Context, event *CallbackEndEvent) {
	if f.EndFn != nil {
		f.EndFn(ctx, event)
	}
}

// MultiCallback 组合多个 Callback
//
// 按注册顺序依次调用所有 Callback。
type MultiCallback struct {
	callbacks []Callback
}

// NewMultiCallback 创建组合 Callback
func NewMultiCallback(callbacks ...Callback) *MultiCallback {
	return &MultiCallback{callbacks: callbacks}
}

// OnStart 依次调用所有 Callback 的 OnStart
func (m *MultiCallback) OnStart(ctx context.Context, event *CallbackStartEvent) {
	for _, cb := range m.callbacks {
		cb.OnStart(ctx, event)
	}
}

// OnEnd 依次调用所有 Callback 的 OnEnd
func (m *MultiCallback) OnEnd(ctx context.Context, event *CallbackEndEvent) {
	for _, cb := range m.callbacks {
		cb.OnEnd(ctx, event)
	}
}
