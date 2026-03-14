// Package cache 提供 LLM 响应缓存实现
//
// 本包实现了 llm.Cache 接口，提供内存缓存和分层缓存能力，
// 用于避免对相同请求重复调用 LLM，节省成本和延迟。
//
// 使用示例:
//
//	// 创建内存缓存
//	c := cache.NewMemoryCache(
//	    cache.WithMaxEntries(1000),
//	    cache.WithTTL(time.Hour),
//	)
//
//	// 通过中间件应用到 Provider
//	provider = llm.Chain(provider, llm.WithCache(c, nil))
package cache

import (
	"context"
	"sync"
	"time"

	"github.com/everyday-items/ai-core/llm"
)

// ============== 内存缓存实现 ==============

// MemoryCache 基于内存的 LLM 响应缓存
//
// 使用 LRU 淘汰策略，支持 TTL 过期和最大条目数限制。
// 线程安全，适用于单进程场景。
//
// 特性:
//   - LRU 淘汰：超过 MaxEntries 时淘汰最久未使用的条目
//   - TTL 过期：条目在指定时间后自动失效
//   - 命中统计：提供命中率等缓存统计信息
//   - 零外部依赖：纯内存实现
type MemoryCache struct {
	mu         sync.RWMutex
	entries    map[string]*cacheEntry
	order      []string // LRU 顺序，尾部是最新的
	maxEntries int
	ttl        time.Duration
	hits       int64
	misses     int64
}

// cacheEntry 缓存条目
type cacheEntry struct {
	response  *llm.CompletionResponse
	createdAt time.Time
}

// MemoryCacheOption 内存缓存配置选项
type MemoryCacheOption func(*MemoryCache)

// WithMaxEntries 设置最大缓存条目数
//
// 超过此数量时使用 LRU 策略淘汰最旧的条目。
// 默认值: 1000
func WithMaxEntries(n int) MemoryCacheOption {
	return func(c *MemoryCache) {
		c.maxEntries = n
	}
}

// WithTTL 设置缓存过期时间
//
// 超过此时间的条目会在访问时自动失效。
// 默认值: 1 小时
func WithTTL(ttl time.Duration) MemoryCacheOption {
	return func(c *MemoryCache) {
		c.ttl = ttl
	}
}

// NewMemoryCache 创建内存缓存
//
// 默认配置:
//   - 最大条目数: 1000
//   - TTL: 1 小时
func NewMemoryCache(opts ...MemoryCacheOption) *MemoryCache {
	c := &MemoryCache{
		entries:    make(map[string]*cacheEntry),
		maxEntries: 1000,
		ttl:        time.Hour,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Get 获取缓存的响应
//
// 如果缓存命中且未过期，返回缓存的响应并更新 LRU 顺序。
// 如果缓存未命中或已过期，返回 nil。
func (c *MemoryCache) Get(_ context.Context, key string) (*llm.CompletionResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, ok := c.entries[key]
	if !ok {
		c.misses++
		return nil, nil
	}

	// 检查 TTL
	if time.Since(entry.createdAt) > c.ttl {
		// 过期删除
		delete(c.entries, key)
		c.removeFromOrder(key)
		c.misses++
		return nil, nil
	}

	// 命中，移到 LRU 尾部
	c.moveToBack(key)
	c.hits++

	return entry.response, nil
}

// Set 缓存响应
//
// 如果缓存已满（超过 MaxEntries），淘汰最久未使用的条目。
func (c *MemoryCache) Set(_ context.Context, key string, resp *llm.CompletionResponse) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// 如果已存在，更新并移到尾部
	if _, ok := c.entries[key]; ok {
		c.entries[key] = &cacheEntry{
			response:  resp,
			createdAt: time.Now(),
		}
		c.moveToBack(key)
		return nil
	}

	// LRU 淘汰
	for len(c.entries) >= c.maxEntries && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.entries, oldest)
	}

	// 插入新条目
	c.entries[key] = &cacheEntry{
		response:  resp,
		createdAt: time.Now(),
	}
	c.order = append(c.order, key)

	return nil
}

// Stats 返回缓存统计信息
func (c *MemoryCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	var hitRate float64
	if total > 0 {
		hitRate = float64(c.hits) / float64(total)
	}

	return CacheStats{
		Size:    len(c.entries),
		Hits:    c.hits,
		Misses:  c.misses,
		HitRate: hitRate,
	}
}

// Clear 清空缓存
func (c *MemoryCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*cacheEntry)
	c.order = c.order[:0]
	c.hits = 0
	c.misses = 0
}

// moveToBack 将 key 移动到 LRU 尾部（最近使用）
// 调用者需持有写锁
func (c *MemoryCache) moveToBack(key string) {
	c.removeFromOrder(key)
	c.order = append(c.order, key)
}

// removeFromOrder 从 LRU 顺序中移除 key
// 调用者需持有写锁
func (c *MemoryCache) removeFromOrder(key string) {
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

// CacheStats 缓存统计信息
type CacheStats struct {
	// Size 当前缓存条目数
	Size int

	// Hits 缓存命中次数
	Hits int64

	// Misses 缓存未命中次数
	Misses int64

	// HitRate 缓存命中率 (0-1)
	HitRate float64
}

// 确保实现了 llm.Cache 接口
var _ llm.Cache = (*MemoryCache)(nil)
