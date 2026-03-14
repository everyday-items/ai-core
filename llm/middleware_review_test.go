package llm

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/everyday-items/ai-core/streamx"
)

// ========== 并发竞态测试 ==========

// TestRateLimitProvider_Concurrent 测试限流中间件在高并发下的竞态条件
func TestRateLimitProvider_Concurrent(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}
	p := Chain(mock, WithRateLimit(1000))

	var wg sync.WaitGroup
	errs := make(chan error, 100)

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := p.Complete(context.Background(), CompletionRequest{})
			if err != nil {
				errs <- err
			}
		}()
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("并发调用失败: %v", err)
	}
}

// TestCacheProvider_Concurrent 测试缓存中间件在高并发下的竞态条件
func TestCacheProvider_Concurrent(t *testing.T) {
	callCount := atomic.Int32{}
	mock := &countingProvider{
		resp:      &CompletionResponse{Content: "cached"},
		callCount: &callCount,
	}

	cache := &inMemoryTestCache{data: make(map[string]*CompletionResponse)}
	p := Chain(mock, WithCache(cache, nil))

	var wg sync.WaitGroup
	req := CompletionRequest{Model: "gpt-4o", Messages: []Message{{Role: RoleUser, Content: "hi"}}}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = p.Complete(context.Background(), req)
		}()
	}
	wg.Wait()
	// 如果 inMemoryTestCache 不是线程安全的，这里会 panic
}

type countingProvider struct {
	resp      *CompletionResponse
	callCount *atomic.Int32
}

func (p *countingProvider) Name() string                                                              { return "counting" }
func (p *countingProvider) Models() []ModelInfo                                                       { return nil }
func (p *countingProvider) CountTokens(messages []Message) (int, error)                               { return 0, nil }
func (p *countingProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) { return nil, nil }
func (p *countingProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	p.callCount.Add(1)
	return p.resp, nil
}

// ========== 边界情况测试 ==========

// TestRetry_ContextCancelled 测试重试中上下文取消
func TestRetry_ContextCancelled(t *testing.T) {
	mock := &mockProvider{
		name:        "test",
		completeErr: errors.New("fail"),
	}
	p := Chain(mock, WithRetry(100, 100*time.Millisecond))

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := p.Complete(ctx, CompletionRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	// 应该在第一次退避等待时因 context 取消退出，而不是重试 100 次
	if mock.callCount.Load() > 2 {
		t.Fatalf("expected at most 2 calls due to context cancel, got %d", mock.callCount.Load())
	}
}

// TestRetry_ZeroRetries 测试 maxRetries=0 时只调用一次
func TestRetry_ZeroRetries(t *testing.T) {
	mock := &mockProvider{
		name:        "test",
		completeErr: errors.New("fail"),
	}
	p := Chain(mock, WithRetry(0, time.Millisecond))

	_, err := p.Complete(context.Background(), CompletionRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if mock.callCount.Load() != 1 {
		t.Fatalf("expected 1 call, got %d", mock.callCount.Load())
	}
}

// TestRateLimit_ContextCancelled 测试限流中上下文取消
func TestRateLimit_ContextCancelled(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}
	// 1 RPS，消耗掉令牌后第二次调用需要等待约 1 秒
	p := Chain(mock, WithRateLimit(1))

	// 消耗掉初始令牌
	_, _ = p.Complete(context.Background(), CompletionRequest{})

	// 第二次调用应该被限流，50ms 后 context 取消
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := p.Complete(ctx, CompletionRequest{})
	if err == nil {
		t.Fatal("expected context cancelled error")
	}
}

// TestTimeout_StreamLeaks 测试超时中间件对流式请求的处理
func TestTimeout_StreamLeaks(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}
	p := Chain(mock, WithTimeout(time.Second))

	// Stream 请求不应该立即 cancel context
	_, err := p.Stream(context.Background(), CompletionRequest{})
	if err != nil {
		t.Fatalf("stream should not fail: %v", err)
	}
}

// TestCache_EmptyMessages 测试空消息的缓存键
func TestCache_EmptyMessages(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "empty"},
	}
	cache := &inMemoryTestCache{data: make(map[string]*CompletionResponse)}
	p := Chain(mock, WithCache(cache, nil))

	// 空消息应该能正常工作
	_, err := p.Complete(context.Background(), CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestChain_NoMiddleware 测试空中间件链
func TestChain_NoMiddleware(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "hello"},
	}

	p := Chain(mock)
	resp, err := p.Complete(context.Background(), CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "hello" {
		t.Fatalf("expected 'hello', got '%s'", resp.Content)
	}
}

// TestCallback_PanicInCallback 测试回调中的 panic 是否会传播
func TestCallback_PanicInCallback(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}

	cb := &CallbackFunc{
		StartFn: func(ctx context.Context, event *CallbackStartEvent) {
			panic("callback panic")
		},
	}

	p := Chain(mock, WithCallback(cb))

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic to propagate")
		}
	}()

	_, _ = p.Complete(context.Background(), CompletionRequest{})
}

// TestMultiCallback_Empty 测试空 MultiCallback
func TestMultiCallback_Empty(t *testing.T) {
	mc := NewMultiCallback()
	// 不应该 panic
	mc.OnStart(context.Background(), &CallbackStartEvent{})
	mc.OnEnd(context.Background(), &CallbackEndEvent{})
}

// TestDefaultCacheKey_Collision 测试默认缓存键是否存在碰撞
func TestDefaultCacheKey_Collision(t *testing.T) {
	key1 := defaultCacheKey(&CompletionRequest{
		Model:    "gpt-4o",
		Messages: []Message{{Role: RoleUser, Content: "a|user:b"}},
	})
	key2 := defaultCacheKey(&CompletionRequest{
		Model:    "gpt-4o",
		Messages: []Message{{Role: RoleUser, Content: "a"}, {Role: RoleUser, Content: "b"}},
	})

	if key1 == key2 {
		t.Fatalf("缓存键碰撞！不同的请求生成了相同的键:\n  key1=%s\n  key2=%s", key1, key2)
	}
}

// ========== 基准测试 ==========

func BenchmarkChain_NoMiddleware(b *testing.B) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}
	p := Chain(mock)
	ctx := context.Background()
	req := CompletionRequest{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = p.Complete(ctx, req)
	}
}

func BenchmarkChain_WithCallback(b *testing.B) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}
	cb := &CallbackFunc{}
	p := Chain(mock, WithCallback(cb))
	ctx := context.Background()
	req := CompletionRequest{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = p.Complete(ctx, req)
	}
}

func BenchmarkDefaultCacheKey(b *testing.B) {
	req := &CompletionRequest{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleSystem, Content: "You are a helpful assistant."},
			{Role: RoleUser, Content: "Hello, how are you?"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = defaultCacheKey(req)
	}
}

func BenchmarkRateLimit_Parallel(b *testing.B) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}
	p := Chain(mock, WithRateLimit(1000000)) // 极高 RPS，不限流
	ctx := context.Background()
	req := CompletionRequest{}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = p.Complete(ctx, req)
		}
	})
}
