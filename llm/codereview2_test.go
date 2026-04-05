package llm

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ============================================================================
// BUG-5: defaultCacheKey 还忽略了 TopP、Stop、User、ToolChoice 字段
// ============================================================================

func TestDefaultCacheKey_IgnoresTopP(t *testing.T) {
	topP1 := 0.1
	topP2 := 0.9
	req1 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		TopP:     &topP1,
	}
	req2 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		TopP:     &topP2,
	}

	if defaultCacheKey(req1) == defaultCacheKey(req2) {
		t.Error("BUG: 不同 TopP 的请求产生相同缓存键，TopP=0.1 会返回高创意的缓存结果")
	}
}

func TestDefaultCacheKey_IgnoresStop(t *testing.T) {
	req1 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		Stop:     []string{"STOP"},
	}
	req2 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
	}

	if defaultCacheKey(req1) == defaultCacheKey(req2) {
		t.Error("BUG: 有 Stop 和无 Stop 的请求产生相同缓存键")
	}
}

func TestDefaultCacheKey_IgnoresUser(t *testing.T) {
	req1 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		User:     "user-A",
	}
	req2 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		User:     "user-B",
	}

	if defaultCacheKey(req1) == defaultCacheKey(req2) {
		t.Error("BUG: 不同 User 的请求产生相同缓存键，可能导致用户数据交叉污染")
	}
}

func TestDefaultCacheKey_IgnoresToolChoice(t *testing.T) {
	req1 := &CompletionRequest{
		Model:      "gpt-4",
		Messages:   []Message{{Role: RoleUser, Content: "hello"}},
		ToolChoice: "auto",
	}
	req2 := &CompletionRequest{
		Model:      "gpt-4",
		Messages:   []Message{{Role: RoleUser, Content: "hello"}},
		ToolChoice: "none",
	}

	if defaultCacheKey(req1) == defaultCacheKey(req2) {
		t.Error("BUG: 不同 ToolChoice 产生相同缓存键")
	}
}

// ============================================================================
// BUG-6: retryProvider 不区分可重试和不可重试错误
// ============================================================================

func TestRetry_RetriesNonRetryableErrors(t *testing.T) {
	mock := &mockProvider{
		name:        "mock",
		completeErr: fmt.Errorf("401 Unauthorized: invalid API key"),
	}

	provider := Chain(mock, WithRetry(3, time.Millisecond))
	_, _ = provider.Complete(context.Background(), CompletionRequest{})

	if mock.callCount.Load() > 1 {
		t.Logf("BUG: 认证错误不应重试，但实际调用了 %d 次。"+
			"浪费资源且可能触发账号限制", mock.callCount.Load())
	}
}

// ============================================================================
// BUG-7: rateLimitProvider.wait 在 tokens 不足时的等待精度
// ============================================================================

func TestRateLimit_SubTokenWait(t *testing.T) {
	rl := &rateLimitProvider{
		inner:    &mockProvider{name: "mock", completeResp: &CompletionResponse{}},
		rps:      2,
		tokens:   0.5,
		lastTime: time.Now(),
	}

	start := time.Now()
	_ = rl.wait(context.Background())
	elapsed := time.Since(start)

	// tokens=0.5, rps=2 → waitTime = (1-0.5)/2 * 1s = 0.25s
	if elapsed < 200*time.Millisecond {
		t.Logf("BUG: tokens=0.5 时应等待 ~250ms，但实际只等了 %v", elapsed)
	}
}

// ============================================================================
// BUG-8: rateLimitProvider 并发快速调用时令牌超发
// ============================================================================

func TestRateLimit_TokenOverflow(t *testing.T) {
	var callCount atomic.Int32
	wrapper := &cr2CallCountingWrapper{
		inner:     &mockProvider{name: "mock", completeResp: &CompletionResponse{}},
		callCount: &callCount,
	}

	provider := Chain(wrapper, WithRateLimit(1))

	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			provider.Complete(ctx, CompletionRequest{})
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)

	completed := callCount.Load()
	if completed > 2 && elapsed < 200*time.Millisecond {
		t.Logf("WARNING: 1 QPS 限制下 100ms 内完成了 %d 个请求", completed)
	}
}

// ============================================================================
// BUG-9: callbackProvider.Stream 的 DurationMs 只测量连接时间
// ============================================================================

func TestCallback_StreamDurationMisleading(t *testing.T) {
	var events []CallbackEndEvent
	var mu sync.Mutex

	callback := &CallbackFunc{
		EndFn: func(ctx context.Context, event *CallbackEndEvent) {
			mu.Lock()
			events = append(events, *event)
			mu.Unlock()
		},
	}

	mock := &mockProvider{
		name:         "mock",
		completeResp: &CompletionResponse{Content: "done"},
	}

	provider := Chain(mock, WithCallback(callback))
	provider.Complete(context.Background(), CompletionRequest{Model: "test"})
	provider.Stream(context.Background(), CompletionRequest{Model: "test"})

	mu.Lock()
	defer mu.Unlock()

	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}

	t.Logf("Complete duration: %dms, Stream duration: %dms",
		events[0].DurationMs, events[1].DurationMs)
	t.Log("DESIGN: Stream 的 DurationMs 只反映连接建立时间，不含数据传输")
}

// ============================================================================
// BUG-10: WithCache + WithRetry 组合的缓存写入行为
// ============================================================================

func TestCacheRetryCombo_DuplicateWrites(t *testing.T) {
	var backendCalls atomic.Int32
	retryMock := &cr2RetryCountProvider{
		callCount: &backendCalls,
		failFirst: true,
	}

	var setCalls atomic.Int32
	cache := &cr2CountingCache{
		data:     make(map[string]*CompletionResponse),
		setCalls: &setCalls,
	}

	provider := Chain(retryMock, WithRetry(2, time.Millisecond), WithCache(cache, nil))

	resp, err := provider.Complete(context.Background(), CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Response: %s, Backend calls: %d, Cache sets: %d",
		resp.Content, backendCalls.Load(), setCalls.Load())
}

// ============================================================================
// BENCHMARK: defaultCacheKey 内存分配
// ============================================================================

func BenchmarkDefaultCacheKey_Allocs(b *testing.B) {
	req := &CompletionRequest{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleSystem, Content: "You are a helpful assistant."},
			{Role: RoleUser, Content: "Hello, how are you?"},
			{Role: RoleAssistant, Content: "I'd be happy to help!"},
			{Role: RoleUser, Content: "Write a function to sort integers."},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = defaultCacheKey(req)
	}
}

// ============================================================================
// BENCHMARK: rateLimitProvider.wait 锁竞争
// ============================================================================

func BenchmarkRateLimit_WaitContention(b *testing.B) {
	rl := &rateLimitProvider{
		inner:    &mockProvider{name: "mock", completeResp: &CompletionResponse{}},
		rps:      1000000,
		tokens:   1000000,
		lastTime: time.Now(),
	}

	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			_ = rl.wait(ctx)
		}
	})
}

// ============================================================================
// 辅助类型（使用 cr2 前缀避免同包冲突）
// ============================================================================

type cr2CallCountingWrapper struct {
	inner     Provider
	callCount *atomic.Int32
}

func (w *cr2CallCountingWrapper) Name() string        { return w.inner.Name() }
func (w *cr2CallCountingWrapper) Models() []ModelInfo { return w.inner.Models() }
func (w *cr2CallCountingWrapper) CountTokens(messages []Message) (int, error) {
	return w.inner.CountTokens(messages)
}
func (w *cr2CallCountingWrapper) Stream(ctx context.Context, req CompletionRequest) (*Stream, error) {
	return w.inner.Stream(ctx, req)
}
func (w *cr2CallCountingWrapper) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	w.callCount.Add(1)
	return w.inner.Complete(ctx, req)
}

type cr2RetryCountProvider struct {
	callCount *atomic.Int32
	failFirst bool
}

func (p *cr2RetryCountProvider) Name() string                                { return "retry-mock" }
func (p *cr2RetryCountProvider) Models() []ModelInfo                         { return nil }
func (p *cr2RetryCountProvider) CountTokens(messages []Message) (int, error) { return 0, nil }
func (p *cr2RetryCountProvider) Stream(ctx context.Context, req CompletionRequest) (*Stream, error) {
	return nil, nil
}
func (p *cr2RetryCountProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	n := p.callCount.Add(1)
	if p.failFirst && n == 1 {
		return nil, fmt.Errorf("temporary failure")
	}
	return &CompletionResponse{Content: "success"}, nil
}

type cr2CountingCache struct {
	mu       sync.Mutex
	data     map[string]*CompletionResponse
	setCalls *atomic.Int32
}

func (c *cr2CountingCache) Get(_ context.Context, key string) (*CompletionResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.data[key], nil
}

func (c *cr2CountingCache) Set(_ context.Context, key string, resp *CompletionResponse) error {
	c.setCalls.Add(1)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = resp
	return nil
}
