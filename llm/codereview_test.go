package llm

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/streamx"
)

// ============== BUG-1: defaultCacheKey 忽略关键字段导致缓存错误命中 ==============

func TestDefaultCacheKey_IgnoresMaxTokens(t *testing.T) {
	req1 := &CompletionRequest{
		Model:     "gpt-4",
		Messages:  []Message{{Role: RoleUser, Content: "hello"}},
		MaxTokens: 100,
	}
	req2 := &CompletionRequest{
		Model:     "gpt-4",
		Messages:  []Message{{Role: RoleUser, Content: "hello"}},
		MaxTokens: 4000,
	}

	key1 := defaultCacheKey(req1)
	key2 := defaultCacheKey(req2)

	if key1 == key2 {
		t.Errorf("BUG: 不同 MaxTokens 的请求产生相同缓存键 %q，会导致缓存错误命中", key1)
	}
}

func TestDefaultCacheKey_IgnoresTemperature(t *testing.T) {
	temp1 := 0.0
	temp2 := 1.5
	req1 := &CompletionRequest{
		Model:       "gpt-4",
		Messages:    []Message{{Role: RoleUser, Content: "hello"}},
		Temperature: &temp1,
	}
	req2 := &CompletionRequest{
		Model:       "gpt-4",
		Messages:    []Message{{Role: RoleUser, Content: "hello"}},
		Temperature: &temp2,
	}

	key1 := defaultCacheKey(req1)
	key2 := defaultCacheKey(req2)

	if key1 == key2 {
		t.Errorf("BUG: Temperature=0 和 Temperature=1.5 产生相同缓存键，会返回错误的缓存结果")
	}
}

func TestDefaultCacheKey_IgnoresTools(t *testing.T) {
	req1 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "what's the weather?"}},
	}
	req2 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "what's the weather?"}},
		Tools: []ToolDefinition{
			NewToolDefinition("get_weather", "获取天气", nil),
		},
	}

	key1 := defaultCacheKey(req1)
	key2 := defaultCacheKey(req2)

	if key1 == key2 {
		t.Errorf("BUG: 带工具和不带工具的请求产生相同缓存键，带工具的请求不会返回 tool_calls")
	}
}

func TestDefaultCacheKey_IgnoresResponseFormat(t *testing.T) {
	req1 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "list items"}},
	}
	req2 := &CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "list items"}},
		ResponseFormat: &ResponseFormat{
			Type: "json_object",
		},
	}

	key1 := defaultCacheKey(req1)
	key2 := defaultCacheKey(req2)

	if key1 == key2 {
		t.Errorf("BUG: 纯文本格式和 JSON 格式的请求产生相同缓存键")
	}
}

// ============== BUG-2: 缓存中间件 Complete 成功后缓存穿透问题 ==============

func TestCacheMiddleware_ThundringHerd(t *testing.T) {
	var callCount int
	var mu sync.Mutex

	mock := &mockProvider{
		name: "mock",
		completeResp: &CompletionResponse{Content: "cached"},
	}

	// 用自定义 completeFn 行为覆盖 completeResp
	origComplete := mock.completeResp
	mock.completeResp = nil
	mock.completeErr = nil

	// 使用包装 provider 来计数
	countingMock := &countingMockProvider{
		inner:    mock,
		mu:       &mu,
		count:    &callCount,
		response: origComplete,
	}

	memCache := &simpleTestCache{data: make(map[string]*CompletionResponse)}
	provider := Chain(countingMock, WithCache(memCache, nil))

	var wg sync.WaitGroup
	req := CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
	}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			provider.Complete(context.Background(), req)
		}()
	}
	wg.Wait()

	mu.Lock()
	calls := callCount
	mu.Unlock()

	if calls > 1 {
		t.Logf("WARNING: 缓存击穿 — 100 并发请求产生 %d 次后端调用（理想值为 1）", calls)
	}
}

// ============== BUG-3: Stream 回调时机错误 — OnEnd 在流开始前就触发 ==============

func TestCallbackMiddleware_StreamDurationIsConnectionTimeOnly(t *testing.T) {
	var endEvent *CallbackEndEvent

	callback := &CallbackFunc{
		EndFn: func(ctx context.Context, event *CallbackEndEvent) {
			endEvent = event
		},
	}

	streamMock := &streamOnlyProvider{
		name: "mock",
		streamFn: func(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
			time.Sleep(5 * time.Millisecond)
			return nil, nil
		},
	}

	provider := Chain(streamMock, WithCallback(callback))
	provider.Stream(context.Background(), CompletionRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
	})

	if endEvent == nil {
		t.Fatal("回调未触发")
	}

	if endEvent.Stream != true {
		t.Error("Stream 字段应该为 true")
	}

	if endEvent.Response != nil {
		t.Log("如果 Response 不为 nil，说明 bug 已修复")
	} else {
		t.Log("BUG确认: Stream 回调的 Response 为 nil，与 Complete 行为不一致")
	}
}

// ============== BUG-4: RateLimit 令牌桶 — 等待后不扣减令牌 ==============

func TestRateLimit_WaitConsumesToken(t *testing.T) {
	rl := &rateLimitProvider{
		inner:    &mockProvider{name: "mock", completeResp: &CompletionResponse{}},
		rps:      1,
		tokens:   0,
		lastTime: time.Now(),
	}

	start := time.Now()
	ctx := context.Background()
	err := rl.wait(ctx)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatal(err)
	}

	if elapsed < 900*time.Millisecond {
		t.Errorf("令牌桶应等待约1秒，实际只等了 %v", elapsed)
	}
	if elapsed > 1200*time.Millisecond {
		t.Errorf("令牌桶等待时间过长: %v", elapsed)
	}
}

// ============== 辅助类型 ==============

// countingMockProvider 计数 provider 调用次数
type countingMockProvider struct {
	inner    Provider
	mu       *sync.Mutex
	count    *int
	response *CompletionResponse
}

func (m *countingMockProvider) Name() string { return "counting-mock" }
func (m *countingMockProvider) Models() []ModelInfo {
	return []ModelInfo{{ID: "mock-model"}}
}
func (m *countingMockProvider) CountTokens(messages []Message) (int, error) {
	return 0, nil
}
func (m *countingMockProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	m.mu.Lock()
	*m.count++
	m.mu.Unlock()
	time.Sleep(10 * time.Millisecond)
	return m.response, nil
}
func (m *countingMockProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	return nil, nil
}

// streamOnlyProvider 仅用于 Stream 测试
type streamOnlyProvider struct {
	name     string
	streamFn func(ctx context.Context, req CompletionRequest) (*streamx.Stream, error)
}

func (m *streamOnlyProvider) Name() string { return m.name }
func (m *streamOnlyProvider) Models() []ModelInfo {
	return []ModelInfo{{ID: "mock-model"}}
}
func (m *streamOnlyProvider) CountTokens(messages []Message) (int, error) {
	return 0, nil
}
func (m *streamOnlyProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	return &CompletionResponse{Content: "mock"}, nil
}
func (m *streamOnlyProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	if m.streamFn != nil {
		return m.streamFn(ctx, req)
	}
	return nil, nil
}

type simpleTestCache struct {
	mu   sync.Mutex
	data map[string]*CompletionResponse
}

func (c *simpleTestCache) Get(_ context.Context, key string) (*CompletionResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	resp := c.data[key]
	return resp, nil
}
func (c *simpleTestCache) Set(_ context.Context, key string, resp *CompletionResponse) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = resp
	return nil
}
