package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/everyday-items/ai-core/streamx"
)

// Middleware 定义 Provider 装饰器函数
//
// Middleware 接收一个 Provider 并返回一个增强后的 Provider，
// 可用于注入缓存、限流、重试、可观测性等横切关注点。
//
// 使用示例:
//
//	provider := openai.New("key")
//	enhanced := llm.Chain(provider,
//	    llm.WithRetry(3, time.Second),
//	    llm.WithRateLimit(10),
//	    llm.WithTimeout(30 * time.Second),
//	    llm.WithCallback(myCallback),
//	)
type Middleware func(Provider) Provider

// Chain 将多个中间件应用到 Provider 上
//
// 中间件按顺序应用，第一个中间件在最外层。
// 例如 Chain(p, A, B, C) 的调用顺序为: A → B → C → Provider
//
// 参数:
//   - provider: 原始 Provider
//   - middlewares: 中间件列表
//
// 返回:
//   - Provider: 增强后的 Provider
func Chain(provider Provider, middlewares ...Middleware) Provider {
	for i := len(middlewares) - 1; i >= 0; i-- {
		provider = middlewares[i](provider)
	}
	return provider
}

// ============== 重试中间件 ==============

// WithRetry 创建重试中间件
//
// 当请求失败时自动重试，使用指数退避策略。
// 退避时间为 backoff * 2^attempt，上限为 30 秒。
//
// 参数:
//   - maxRetries: 最大重试次数（不含首次请求）
//   - backoff: 初始退避时间
func WithRetry(maxRetries int, backoff time.Duration) Middleware {
	return func(next Provider) Provider {
		return &retryProvider{
			inner:      next,
			maxRetries: maxRetries,
			backoff:    backoff,
		}
	}
}

type retryProvider struct {
	inner      Provider
	maxRetries int
	backoff    time.Duration
}

func (p *retryProvider) Name() string { return p.inner.Name() }
func (p *retryProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *retryProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *retryProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	var lastErr error
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			delay := p.backoff * time.Duration(1<<uint(attempt-1))
			delay = min(delay, 30*time.Second)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		resp, err := p.inner.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("重试 %d 次后仍然失败: %w", p.maxRetries, lastErr)
}

func (p *retryProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	var lastErr error
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			delay := p.backoff * time.Duration(1<<uint(attempt-1))
			delay = min(delay, 30*time.Second)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		stream, err := p.inner.Stream(ctx, req)
		if err == nil {
			return stream, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("重试 %d 次后仍然失败: %w", p.maxRetries, lastErr)
}

// ============== 限流中间件 ==============

// WithRateLimit 创建限流中间件
//
// 使用令牌桶算法限制每秒请求数。
// 当请求超过限制时会等待直到有可用令牌或上下文取消。
//
// 参数:
//   - rps: 每秒允许的最大请求数
func WithRateLimit(rps float64) Middleware {
	return func(next Provider) Provider {
		return &rateLimitProvider{
			inner:    next,
			rps:      rps,
			tokens:   rps,
			lastTime: time.Now(),
		}
	}
}

type rateLimitProvider struct {
	inner    Provider
	rps      float64
	tokens   float64
	lastTime time.Time
	mu       sync.Mutex
}

func (p *rateLimitProvider) Name() string { return p.inner.Name() }
func (p *rateLimitProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *rateLimitProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *rateLimitProvider) wait(ctx context.Context) error {
	p.mu.Lock()
	now := time.Now()
	elapsed := now.Sub(p.lastTime).Seconds()
	p.tokens += elapsed * p.rps
	if p.tokens > p.rps {
		p.tokens = p.rps
	}
	p.lastTime = now

	if p.tokens >= 1 {
		p.tokens--
		p.mu.Unlock()
		return nil
	}

	// 计算需要等待的时间
	waitTime := time.Duration((1 - p.tokens) / p.rps * float64(time.Second))
	p.tokens = 0
	p.lastTime = now.Add(waitTime)
	p.mu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(waitTime):
		return nil
	}
}

func (p *rateLimitProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	if err := p.wait(ctx); err != nil {
		return nil, err
	}
	return p.inner.Complete(ctx, req)
}

func (p *rateLimitProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	if err := p.wait(ctx); err != nil {
		return nil, err
	}
	return p.inner.Stream(ctx, req)
}

// ============== 超时中间件 ==============

// WithTimeout 创建超时中间件
//
// 为每个请求设置超时时间。如果请求在超时前未完成，
// 将返回 context.DeadlineExceeded 错误。
//
// 参数:
//   - timeout: 请求超时时间
func WithTimeout(timeout time.Duration) Middleware {
	return func(next Provider) Provider {
		return &timeoutProvider{
			inner:   next,
			timeout: timeout,
		}
	}
}

type timeoutProvider struct {
	inner   Provider
	timeout time.Duration
}

func (p *timeoutProvider) Name() string { return p.inner.Name() }
func (p *timeoutProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *timeoutProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *timeoutProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	return p.inner.Complete(ctx, req)
}

func (p *timeoutProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	// 注意：流式请求的 cancel 不能立即调用，需要在流关闭后调用
	// 这里让 context 在超时后自动取消
	_ = cancel
	return p.inner.Stream(ctx, req)
}

// ============== 回调中间件 ==============

// WithCallback 创建回调中间件
//
// 在每次 LLM 请求的开始和结束时触发回调，
// 用于实现可观测性（追踪、日志、指标等）。
//
// 参数:
//   - callback: 回调接口实现
func WithCallback(callback Callback) Middleware {
	return func(next Provider) Provider {
		return &callbackProvider{
			inner:    next,
			callback: callback,
		}
	}
}

type callbackProvider struct {
	inner    Provider
	callback Callback
}

func (p *callbackProvider) Name() string { return p.inner.Name() }
func (p *callbackProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *callbackProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *callbackProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	p.callback.OnStart(ctx, &CallbackStartEvent{
		Provider: p.inner.Name(),
		Request:  &req,
	})

	start := time.Now()
	resp, err := p.inner.Complete(ctx, req)
	duration := time.Since(start).Milliseconds()

	p.callback.OnEnd(ctx, &CallbackEndEvent{
		Provider:   p.inner.Name(),
		Request:    &req,
		Response:   resp,
		Error:      err,
		DurationMs: duration,
		Stream:     false,
	})

	return resp, err
}

func (p *callbackProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	p.callback.OnStart(ctx, &CallbackStartEvent{
		Provider: p.inner.Name(),
		Request:  &req,
	})

	start := time.Now()
	stream, err := p.inner.Stream(ctx, req)
	duration := time.Since(start).Milliseconds()

	p.callback.OnEnd(ctx, &CallbackEndEvent{
		Provider:   p.inner.Name(),
		Request:    &req,
		Error:      err,
		DurationMs: duration,
		Stream:     true,
	})

	return stream, err
}

// ============== 缓存中间件 ==============

// Cache 定义 LLM 响应缓存接口
//
// 实现此接口以提供不同的缓存后端（内存、Redis 等）。
type Cache interface {
	// Get 获取缓存的响应
	//
	// 参数:
	//   - ctx: 上下文
	//   - key: 缓存键
	//
	// 返回:
	//   - *CompletionResponse: 缓存命中时返回响应，未命中返回 nil
	//   - error: 缓存访问错误
	Get(ctx context.Context, key string) (*CompletionResponse, error)

	// Set 缓存响应
	//
	// 参数:
	//   - ctx: 上下文
	//   - key: 缓存键
	//   - resp: 要缓存的响应
	//
	// 返回:
	//   - error: 缓存写入错误
	Set(ctx context.Context, key string, resp *CompletionResponse) error
}

// CacheKeyFunc 自定义缓存键生成函数
type CacheKeyFunc func(req *CompletionRequest) string

// WithCache 创建缓存中间件
//
// 对相同的请求返回缓存的响应，避免重复调用 LLM。
// 仅缓存非流式请求（Complete），流式请求（Stream）直接透传。
//
// 参数:
//   - cache: 缓存后端实现
//   - keyFn: 缓存键生成函数（可为 nil，使用默认键生成）
func WithCache(cache Cache, keyFn CacheKeyFunc) Middleware {
	return func(next Provider) Provider {
		return &cacheProvider{
			inner: next,
			cache: cache,
			keyFn: keyFn,
		}
	}
}

type cacheProvider struct {
	inner Provider
	cache Cache
	keyFn CacheKeyFunc
}

func (p *cacheProvider) Name() string { return p.inner.Name() }
func (p *cacheProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *cacheProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *cacheProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	// 生成缓存键
	var key string
	if p.keyFn != nil {
		key = p.keyFn(&req)
	} else {
		key = defaultCacheKey(&req)
	}

	// 查找缓存
	if cached, err := p.cache.Get(ctx, key); err == nil && cached != nil {
		return cached, nil
	}

	// 调用后端
	resp, err := p.inner.Complete(ctx, req)
	if err != nil {
		return nil, err
	}

	// 写入缓存（忽略缓存写入错误）
	_ = p.cache.Set(ctx, key, resp)

	return resp, nil
}

func (p *cacheProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	// 流式请求不走缓存
	return p.inner.Stream(ctx, req)
}

// defaultCacheKey 默认的缓存键生成
// 基于模型名和消息内容生成简单的键
func defaultCacheKey(req *CompletionRequest) string {
	var b strings.Builder
	b.WriteString(req.Model)
	for _, msg := range req.Messages {
		b.WriteByte('|')
		b.WriteString(string(msg.Role))
		b.WriteByte(':')
		b.WriteString(msg.Content)
	}
	return b.String()
}
