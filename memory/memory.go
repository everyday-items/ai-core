package memory

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Memory 定义记忆系统的核心接口
type Memory interface {
	// Save 保存单条记忆条目
	Save(ctx context.Context, entry Entry) error

	// SaveBatch 批量保存记忆条目
	SaveBatch(ctx context.Context, entries []Entry) error

	// Get 根据 ID 获取记忆条目
	Get(ctx context.Context, id string) (*Entry, error)

	// Search 搜索记忆条目
	Search(ctx context.Context, query SearchQuery) ([]Entry, error)

	// Delete 删除指定 ID 的记忆条目
	Delete(ctx context.Context, id string) error

	// Clear 清空所有记忆
	Clear(ctx context.Context) error

	// Stats 返回记忆统计信息
	Stats() MemoryStats
}

// Entry 表示一条记忆条目
type Entry struct {
	// ID 唯一标识符
	ID string `json:"id"`

	// Role 角色（user、assistant、system、tool）
	Role string `json:"role"`

	// Content 内容
	Content string `json:"content"`

	// Metadata 元数据
	Metadata map[string]any `json:"metadata,omitempty"`

	// Embedding 向量嵌入（用于语义搜索）
	Embedding []float32 `json:"embedding,omitempty"`

	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt 更新时间
	UpdatedAt time.Time `json:"updated_at,omitempty"`
}

// SearchQuery 定义搜索查询参数
type SearchQuery struct {
	// Query 查询文本（用于语义搜索）
	Query string `json:"query,omitempty"`

	// Embedding 查询向量（用于向量搜索）
	Embedding []float32 `json:"embedding,omitempty"`

	// Limit 返回结果数量限制
	Limit int `json:"limit,omitempty"`

	// Offset 分页偏移量
	Offset int `json:"offset,omitempty"`

	// Roles 过滤角色
	Roles []string `json:"roles,omitempty"`

	// Since 时间范围开始
	Since *time.Time `json:"since,omitempty"`

	// Until 时间范围结束
	Until *time.Time `json:"until,omitempty"`

	// Metadata 元数据过滤
	Metadata map[string]any `json:"metadata,omitempty"`

	// OrderBy 排序字段
	OrderBy string `json:"order_by,omitempty"`

	// OrderDesc 是否降序
	OrderDesc bool `json:"order_desc,omitempty"`
}

// MemoryStats 包含记忆统计信息
type MemoryStats struct {
	// EntryCount 条目总数
	EntryCount int `json:"entry_count"`

	// TokenCount 总 Token 数（如果可计算）
	TokenCount int `json:"token_count,omitempty"`

	// OldestEntry 最早条目时间
	OldestEntry *time.Time `json:"oldest_entry,omitempty"`

	// NewestEntry 最新条目时间
	NewestEntry *time.Time `json:"newest_entry,omitempty"`
}

// ============== 缓冲记忆实现 ==============

// BufferMemory 是一个简单的内存缓冲区实现
// 使用 FIFO 策略，当超过容量时移除最旧的条目
type BufferMemory struct {
	entries  []Entry
	capacity int
	mu       sync.RWMutex
	idGen    func() string
}

// BufferOption 是 BufferMemory 的配置选项
type BufferOption func(*BufferMemory)

// WithIDGenerator 设置 ID 生成器
func WithIDGenerator(gen func() string) BufferOption {
	return func(m *BufferMemory) {
		m.idGen = gen
	}
}

// NewBuffer 创建缓冲记忆
// capacity 是最大条目数，超过后会移除最旧的条目
func NewBuffer(capacity int, opts ...BufferOption) *BufferMemory {
	if capacity <= 0 {
		capacity = 100
	}
	m := &BufferMemory{
		entries:  make([]Entry, 0, capacity),
		capacity: capacity,
		idGen:    defaultIDGen,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

var idCounter atomic.Uint64

func defaultIDGen() string {
	// 使用原子计数器 + 随机数确保唯一性，避免时间戳冲突
	counter := idCounter.Add(1)
	randomBytes := make([]byte, 4)
	_, _ = rand.Read(randomBytes)
	return fmt.Sprintf("mem-%d-%s", counter, hex.EncodeToString(randomBytes))
}

// Save 保存单条记忆条目
func (m *BufferMemory) Save(ctx context.Context, entry Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if entry.ID == "" {
		entry.ID = m.idGen()
	}
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now()
	}

	// 如果已达容量，移除最旧的条目
	if len(m.entries) >= m.capacity {
		m.entries = m.entries[1:]
	}

	m.entries = append(m.entries, entry)
	return nil
}

// SaveBatch 批量保存记忆条目
func (m *BufferMemory) SaveBatch(ctx context.Context, entries []Entry) error {
	for _, entry := range entries {
		if err := m.Save(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

// Get 根据 ID 获取记忆条目
func (m *BufferMemory) Get(ctx context.Context, id string) (*Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for i := range m.entries {
		if m.entries[i].ID == id {
			entry := m.entries[i]
			return &entry, nil
		}
	}
	return nil, nil
}

// Search 搜索记忆条目
func (m *BufferMemory) Search(ctx context.Context, query SearchQuery) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 过滤
	var results []Entry
	for _, entry := range m.entries {
		if !matchQuery(entry, query) {
			continue
		}
		results = append(results, entry)
	}

	// 排序（默认按时间升序）
	if query.OrderDesc {
		// 降序：反转结果
		for i, j := 0, len(results)-1; i < j; i, j = i+1, j-1 {
			results[i], results[j] = results[j], results[i]
		}
	}

	// 分页
	start := query.Offset
	if start < 0 {
		start = 0
	}
	if start >= len(results) {
		return nil, nil
	}
	end := len(results)
	if query.Limit > 0 && start+query.Limit < end {
		end = start + query.Limit
	}

	return results[start:end], nil
}

func matchQuery(entry Entry, query SearchQuery) bool {
	// 角色过滤
	if len(query.Roles) > 0 {
		found := false
		for _, role := range query.Roles {
			if entry.Role == role {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// 时间范围过滤
	if query.Since != nil && entry.CreatedAt.Before(*query.Since) {
		return false
	}
	if query.Until != nil && entry.CreatedAt.After(*query.Until) {
		return false
	}

	// 元数据过滤
	if len(query.Metadata) > 0 {
		if entry.Metadata == nil {
			return false
		}
		for k, v := range query.Metadata {
			if mv, ok := entry.Metadata[k]; !ok || mv != v {
				return false
			}
		}
	}

	return true
}

// Delete 删除指定 ID 的记忆条目
func (m *BufferMemory) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i, entry := range m.entries {
		if entry.ID == id {
			m.entries = append(m.entries[:i], m.entries[i+1:]...)
			return nil
		}
	}
	return nil
}

// Clear 清空所有记忆
func (m *BufferMemory) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = m.entries[:0]
	return nil
}

// Stats 返回记忆统计信息
func (m *BufferMemory) Stats() MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := MemoryStats{
		EntryCount: len(m.entries),
	}

	if len(m.entries) > 0 {
		oldest := m.entries[0].CreatedAt
		newest := m.entries[len(m.entries)-1].CreatedAt
		stats.OldestEntry = &oldest
		stats.NewestEntry = &newest
	}

	return stats
}

// Entries 返回所有条目的副本
func (m *BufferMemory) Entries() []Entry {
	m.mu.RLock()
	defer m.mu.RUnlock()
	entries := make([]Entry, len(m.entries))
	copy(entries, m.entries)
	return entries
}

// Last 返回最后 n 条记忆
func (m *BufferMemory) Last(n int) []Entry {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if n >= len(m.entries) {
		entries := make([]Entry, len(m.entries))
		copy(entries, m.entries)
		return entries
	}

	start := len(m.entries) - n
	entries := make([]Entry, n)
	copy(entries, m.entries[start:])
	return entries
}

// ============== 便捷函数 ==============

// NewEntry 创建新的记忆条目
func NewEntry(role, content string) Entry {
	return Entry{
		Role:      role,
		Content:   content,
		CreatedAt: time.Now(),
	}
}

// NewUserEntry 创建用户消息条目
func NewUserEntry(content string) Entry {
	return NewEntry("user", content)
}

// NewAssistantEntry 创建助手消息条目
func NewAssistantEntry(content string) Entry {
	return NewEntry("assistant", content)
}

// NewSystemEntry 创建系统消息条目
func NewSystemEntry(content string) Entry {
	return NewEntry("system", content)
}

// NewToolEntry 创建工具消息条目
func NewToolEntry(content string) Entry {
	return NewEntry("tool", content)
}
