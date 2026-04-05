// Package cache 提供 LLM 响应缓存实现
//
// 本包实现了 llm.Cache 接口，提供内存缓存能力，
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
	"container/list"
	"context"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// ============== 内存缓存实现 ==============

// MemoryCache 基于内存的 LLM 响应缓存
//
// 使用 LRU 淘汰策略，支持 TTL 过期和最大条目数限制。
// 线程安全，适用于单进程场景。
//
// 特性:
//   - LRU 淘汰：O(1) 时间复杂度，基于 container/list 双向链表
//   - TTL 过期：条目在指定时间后自动失效（惰性清理）
//   - 命中统计：提供命中率等缓存统计信息
//   - 零外部依赖：纯内存实现
type MemoryCache struct {
	mu         sync.Mutex
	entries    map[string]*list.Element // key → list.Element (值为 *cacheEntry)
	evictList  *list.List               // LRU 链表，前端是最旧的
	maxEntries int
	ttl        time.Duration
	hits       int64
	misses     int64
}

// cacheEntry 缓存条目
type cacheEntry struct {
	key       string
	response  *llm.CompletionResponse
	createdAt time.Time
}

// MemoryCacheOption 内存缓存配置选项
type MemoryCacheOption func(*MemoryCache)

// WithMaxEntries 设置最大缓存条目数
//
// 超过此数量时使用 LRU 策略淘汰最旧的条目。
// 值 ≤ 0 时禁用缓存（所有 Set 操作不存储）。
// 默认值: 1000
func WithMaxEntries(n int) MemoryCacheOption {
	return func(c *MemoryCache) {
		c.maxEntries = n
	}
}

// WithTTL 设置缓存过期时间
//
// 超过此时间的条目会在访问时自动失效（惰性清理）。
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
		entries:    make(map[string]*list.Element),
		evictList:  list.New(),
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
// 如果缓存命中且未过期，返回缓存的响应并更新 LRU 顺序（O(1)）。
// 如果缓存未命中或已过期，返回 nil。
func (c *MemoryCache) Get(_ context.Context, key string) (*llm.CompletionResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.entries[key]
	if !ok {
		c.misses++
		return nil, nil
	}

	entry := elem.Value.(*cacheEntry)

	// 检查 TTL
	if c.ttl > 0 && time.Since(entry.createdAt) > c.ttl {
		// 过期删除
		c.removeElement(elem)
		c.misses++
		return nil, nil
	}

	// 命中，移到 LRU 尾部（最近使用）— O(1)
	c.evictList.MoveToBack(elem)
	c.hits++

	return entry.response, nil
}

// Set 缓存响应
//
// 如果缓存已满（超过 MaxEntries），淘汰最久未使用的条目。
// 当 MaxEntries ≤ 0 时，不缓存任何条目。
func (c *MemoryCache) Set(_ context.Context, key string, resp *llm.CompletionResponse) error {
	// maxEntries ≤ 0 时禁用缓存
	if c.maxEntries <= 0 {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// 如果已存在，更新并移到尾部
	if elem, ok := c.entries[key]; ok {
		entry := elem.Value.(*cacheEntry)
		entry.response = resp
		entry.createdAt = time.Now()
		c.evictList.MoveToBack(elem)
		return nil
	}

	// LRU 淘汰
	for c.evictList.Len() >= c.maxEntries {
		oldest := c.evictList.Front()
		if oldest == nil {
			break
		}
		c.removeElement(oldest)
	}

	// 插入新条目
	entry := &cacheEntry{
		key:       key,
		response:  resp,
		createdAt: time.Now(),
	}
	elem := c.evictList.PushBack(entry)
	c.entries[key] = elem

	return nil
}

// Stats 返回缓存统计信息
func (c *MemoryCache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()

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

	c.entries = make(map[string]*list.Element)
	c.evictList.Init()
	c.hits = 0
	c.misses = 0
}

// removeElement 从缓存中移除元素
// 调用者需持有锁
func (c *MemoryCache) removeElement(elem *list.Element) {
	entry := elem.Value.(*cacheEntry)
	delete(c.entries, entry.key)
	c.evictList.Remove(elem)
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
