package cache

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// ========== 并发竞态测试 ==========

// TestMemoryCache_Concurrent 测试并发读写安全
func TestMemoryCache_Concurrent(t *testing.T) {
	c := NewMemoryCache(WithMaxEntries(100))
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", i)
			_ = c.Set(ctx, key, &llm.CompletionResponse{Content: key})
			_, _ = c.Get(ctx, key)
			_ = c.Stats()
		}(i)
	}
	wg.Wait()
}

// TestMemoryCache_ConcurrentLRU 测试并发下 LRU 淘汰的正确性
func TestMemoryCache_ConcurrentLRU(t *testing.T) {
	c := NewMemoryCache(WithMaxEntries(10))
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", i)
			_ = c.Set(ctx, key, &llm.CompletionResponse{Content: key})
		}(i)
	}
	wg.Wait()

	stats := c.Stats()
	if stats.Size > 10 {
		t.Fatalf("缓存大小超过限制: 期望 ≤10, 实际 %d", stats.Size)
	}
}

// ========== 边界情况测试 ==========

// TestMemoryCache_MaxEntries_Zero 测试 MaxEntries=0 时的行为
func TestMemoryCache_MaxEntries_Zero(t *testing.T) {
	c := NewMemoryCache(WithMaxEntries(0))
	ctx := context.Background()

	// maxEntries=0 意味着无法存储任何条目
	err := c.Set(ctx, "key", &llm.CompletionResponse{Content: "val"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 应该立即被淘汰
	got, _ := c.Get(ctx, "key")
	if got != nil {
		t.Fatal("maxEntries=0 时不应保留任何缓存条目")
	}
}

// TestMemoryCache_MaxEntries_Negative 测试 MaxEntries 为负值时的行为
func TestMemoryCache_MaxEntries_Negative(t *testing.T) {
	c := NewMemoryCache(WithMaxEntries(-1))
	ctx := context.Background()

	err := c.Set(ctx, "key", &llm.CompletionResponse{Content: "val"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 负值 maxEntries 会导致什么行为？
	got, _ := c.Get(ctx, "key")
	if got != nil {
		t.Log("警告: maxEntries=-1 仍然存储了缓存条目")
	}
}

// TestMemoryCache_TTL_Zero 测试 TTL=0 时的行为
//
// TTL=0 表示不启用 TTL（零值语义），条目永不过期
func TestMemoryCache_TTL_Zero(t *testing.T) {
	c := NewMemoryCache(WithTTL(0))
	ctx := context.Background()

	_ = c.Set(ctx, "key", &llm.CompletionResponse{Content: "val"})
	got, _ := c.Get(ctx, "key")
	// TTL=0 → 不启用过期，条目应该存在
	if got == nil {
		t.Fatal("TTL=0 时不启用过期，缓存条目应存在")
	}
}

// TestMemoryCache_LRU_OrderConsistency 测试 LRU order 与 entries 的一致性
func TestMemoryCache_LRU_OrderConsistency(t *testing.T) {
	c := NewMemoryCache(WithMaxEntries(3))
	ctx := context.Background()

	// 添加 3 个条目
	_ = c.Set(ctx, "a", &llm.CompletionResponse{Content: "a"})
	_ = c.Set(ctx, "b", &llm.CompletionResponse{Content: "b"})
	_ = c.Set(ctx, "c", &llm.CompletionResponse{Content: "c"})

	// 更新已存在的条目
	_ = c.Set(ctx, "a", &llm.CompletionResponse{Content: "a-updated"})

	// 添加新条目，应淘汰 "b"（最久未使用）
	_ = c.Set(ctx, "d", &llm.CompletionResponse{Content: "d"})

	// 验证 "b" 被淘汰
	got, _ := c.Get(ctx, "b")
	if got != nil {
		t.Fatal("expected 'b' to be evicted")
	}

	// 验证 "a" 仍在（因为被更新过）
	got, _ = c.Get(ctx, "a")
	if got == nil {
		t.Fatal("expected 'a' to be in cache")
	}
	if got.Content != "a-updated" {
		t.Fatalf("expected 'a-updated', got '%s'", got.Content)
	}
}

// TestMemoryCache_SameKey_Overwrite 测试相同键覆盖
func TestMemoryCache_SameKey_Overwrite(t *testing.T) {
	c := NewMemoryCache()
	ctx := context.Background()

	_ = c.Set(ctx, "key", &llm.CompletionResponse{Content: "v1"})
	_ = c.Set(ctx, "key", &llm.CompletionResponse{Content: "v2"})

	got, _ := c.Get(ctx, "key")
	if got == nil || got.Content != "v2" {
		t.Fatalf("expected 'v2', got '%v'", got)
	}

	stats := c.Stats()
	if stats.Size != 1 {
		t.Fatalf("expected size 1, got %d", stats.Size)
	}
}

// ========== 基准测试 ==========

func BenchmarkMemoryCache_Get(b *testing.B) {
	c := NewMemoryCache(WithMaxEntries(10000))
	ctx := context.Background()

	// 预填充
	for i := 0; i < 1000; i++ {
		_ = c.Set(ctx, fmt.Sprintf("key-%d", i), &llm.CompletionResponse{Content: "val"})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.Get(ctx, fmt.Sprintf("key-%d", i%1000))
	}
}

func BenchmarkMemoryCache_Set(b *testing.B) {
	c := NewMemoryCache(WithMaxEntries(10000))
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.Set(ctx, fmt.Sprintf("key-%d", i), &llm.CompletionResponse{Content: "val"})
	}
}

func BenchmarkMemoryCache_LRU_Eviction(b *testing.B) {
	c := NewMemoryCache(WithMaxEntries(100))
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.Set(ctx, fmt.Sprintf("key-%d", i), &llm.CompletionResponse{Content: "val"})
	}
}

func BenchmarkMemoryCache_Get_Parallel(b *testing.B) {
	c := NewMemoryCache(WithMaxEntries(10000))
	ctx := context.Background()

	for i := 0; i < 1000; i++ {
		_ = c.Set(ctx, fmt.Sprintf("key-%d", i), &llm.CompletionResponse{Content: "val"})
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			_, _ = c.Get(ctx, fmt.Sprintf("key-%d", i%1000))
			i++
		}
	})
}

func BenchmarkMemoryCache_Mixed_Parallel(b *testing.B) {
	c := NewMemoryCache(WithMaxEntries(1000))
	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("key-%d", i%500)
			if i%3 == 0 {
				_ = c.Set(ctx, key, &llm.CompletionResponse{Content: "val"})
			} else {
				_, _ = c.Get(ctx, key)
			}
			i++
		}
	})
}

// BenchmarkMemoryCache_LRU_MoveToBack_LargeCache 测试大缓存下 moveToBack 的性能
func BenchmarkMemoryCache_LRU_MoveToBack_LargeCache(b *testing.B) {
	c := NewMemoryCache(WithMaxEntries(10000), WithTTL(time.Hour))
	ctx := context.Background()

	// 预填充 10000 个条目
	for i := 0; i < 10000; i++ {
		_ = c.Set(ctx, fmt.Sprintf("key-%d", i), &llm.CompletionResponse{Content: "val"})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 每次 Get 都会触发 moveToBack，这是 O(n) 操作
		_, _ = c.Get(ctx, fmt.Sprintf("key-%d", i%10000))
	}
}
