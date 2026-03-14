package cache

import (
	"context"
	"testing"
	"time"

	"github.com/everyday-items/ai-core/llm"
)

func TestMemoryCache_GetSet(t *testing.T) {
	c := NewMemoryCache()

	ctx := context.Background()
	key := "test-key"
	resp := &llm.CompletionResponse{Content: "hello"}

	// 缓存未命中
	got, err := c.Get(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Fatal("expected nil on cache miss")
	}

	// 写入缓存
	if err := c.Set(ctx, key, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 缓存命中
	got, err = c.Get(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("expected cache hit")
	}
	if got.Content != "hello" {
		t.Fatalf("expected 'hello', got '%s'", got.Content)
	}
}

func TestMemoryCache_TTL(t *testing.T) {
	c := NewMemoryCache(WithTTL(50 * time.Millisecond))

	ctx := context.Background()
	key := "ttl-key"
	resp := &llm.CompletionResponse{Content: "hello"}

	_ = c.Set(ctx, key, resp)

	// 立即获取应命中
	got, _ := c.Get(ctx, key)
	if got == nil {
		t.Fatal("expected cache hit")
	}

	// 等待过期
	time.Sleep(60 * time.Millisecond)

	// 过期后应未命中
	got, _ = c.Get(ctx, key)
	if got != nil {
		t.Fatal("expected cache miss after TTL")
	}
}

func TestMemoryCache_LRU(t *testing.T) {
	c := NewMemoryCache(WithMaxEntries(3))

	ctx := context.Background()

	// 插入 3 个条目
	for i := 0; i < 3; i++ {
		key := string(rune('a' + i))
		_ = c.Set(ctx, key, &llm.CompletionResponse{Content: key})
	}

	// 访问 "a" 使其最近使用
	_, _ = c.Get(ctx, "a")

	// 插入第 4 个条目，应淘汰 "b"（最久未使用）
	_ = c.Set(ctx, "d", &llm.CompletionResponse{Content: "d"})

	// "b" 应被淘汰
	got, _ := c.Get(ctx, "b")
	if got != nil {
		t.Fatal("expected 'b' to be evicted")
	}

	// "a" 应仍在缓存中
	got, _ = c.Get(ctx, "a")
	if got == nil {
		t.Fatal("expected 'a' to be in cache")
	}

	// "d" 应在缓存中
	got, _ = c.Get(ctx, "d")
	if got == nil {
		t.Fatal("expected 'd' to be in cache")
	}
}

func TestMemoryCache_Stats(t *testing.T) {
	c := NewMemoryCache()
	ctx := context.Background()

	_ = c.Set(ctx, "key", &llm.CompletionResponse{Content: "val"})

	// 1 hit + 1 miss
	_, _ = c.Get(ctx, "key")     // hit
	_, _ = c.Get(ctx, "missing") // miss

	stats := c.Stats()
	if stats.Hits != 1 {
		t.Fatalf("expected 1 hit, got %d", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Fatalf("expected 1 miss, got %d", stats.Misses)
	}
	if stats.HitRate != 0.5 {
		t.Fatalf("expected hit rate 0.5, got %f", stats.HitRate)
	}
	if stats.Size != 1 {
		t.Fatalf("expected size 1, got %d", stats.Size)
	}
}

func TestMemoryCache_Clear(t *testing.T) {
	c := NewMemoryCache()
	ctx := context.Background()

	_ = c.Set(ctx, "k1", &llm.CompletionResponse{Content: "v1"})
	_ = c.Set(ctx, "k2", &llm.CompletionResponse{Content: "v2"})

	c.Clear()

	got, _ := c.Get(ctx, "k1")
	if got != nil {
		t.Fatal("expected nil after clear")
	}

	stats := c.Stats()
	if stats.Size != 0 {
		t.Fatalf("expected size 0, got %d", stats.Size)
	}
}
