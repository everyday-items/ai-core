package memory

import (
	"context"
	"testing"
	"time"
)

func TestNewBuffer(t *testing.T) {
	mem := NewBuffer(10)

	stats := mem.Stats()
	if stats.EntryCount != 0 {
		t.Errorf("EntryCount = %d, want 0", stats.EntryCount)
	}
}

func TestNewBuffer_DefaultCapacity(t *testing.T) {
	// 负数容量应该使用默认值
	mem := NewBuffer(-1)
	if mem == nil {
		t.Fatal("NewBuffer(-1) returned nil")
	}

	// 零容量应该使用默认值
	mem = NewBuffer(0)
	if mem == nil {
		t.Fatal("NewBuffer(0) returned nil")
	}
}

func TestBufferMemory_Save(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	entry := Entry{
		Role:    "user",
		Content: "Hello",
	}

	if err := mem.Save(ctx, entry); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	stats := mem.Stats()
	if stats.EntryCount != 1 {
		t.Errorf("EntryCount = %d, want 1", stats.EntryCount)
	}
}

func TestBufferMemory_Save_AutoID(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	entry := Entry{
		Role:    "user",
		Content: "Hello",
	}

	mem.Save(ctx, entry)

	entries := mem.Entries()
	if len(entries) != 1 {
		t.Fatalf("len(Entries) = %d, want 1", len(entries))
	}
	if entries[0].ID == "" {
		t.Error("Entry.ID should be auto-generated")
	}
}

func TestBufferMemory_Save_AutoTimestamp(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	before := time.Now()
	entry := Entry{Role: "user", Content: "Hello"}
	mem.Save(ctx, entry)
	after := time.Now()

	entries := mem.Entries()
	createdAt := entries[0].CreatedAt

	if createdAt.Before(before) || createdAt.After(after) {
		t.Error("CreatedAt should be auto-set to current time")
	}
}

func TestBufferMemory_Save_Capacity(t *testing.T) {
	mem := NewBuffer(3)
	ctx := context.Background()

	// 保存 5 条，应该只保留最后 3 条
	for i := 0; i < 5; i++ {
		mem.Save(ctx, Entry{
			ID:      string(rune('A' + i)),
			Role:    "user",
			Content: string(rune('A' + i)),
		})
	}

	entries := mem.Entries()
	if len(entries) != 3 {
		t.Errorf("len(Entries) = %d, want 3", len(entries))
	}

	// 应该保留 C, D, E
	if entries[0].ID != "C" {
		t.Errorf("entries[0].ID = %q, want %q", entries[0].ID, "C")
	}
	if entries[2].ID != "E" {
		t.Errorf("entries[2].ID = %q, want %q", entries[2].ID, "E")
	}
}

func TestBufferMemory_SaveBatch(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	entries := []Entry{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "How are you?"},
	}

	if err := mem.SaveBatch(ctx, entries); err != nil {
		t.Fatalf("SaveBatch error: %v", err)
	}

	stats := mem.Stats()
	if stats.EntryCount != 3 {
		t.Errorf("EntryCount = %d, want 3", stats.EntryCount)
	}
}

func TestBufferMemory_Get(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	entry := Entry{
		ID:      "test-id",
		Role:    "user",
		Content: "Hello",
	}
	mem.Save(ctx, entry)

	// 获取存在的条目
	got, err := mem.Get(ctx, "test-id")
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	if got == nil {
		t.Fatal("Get returned nil")
	}
	if got.Content != "Hello" {
		t.Errorf("Content = %q, want %q", got.Content, "Hello")
	}

	// 获取不存在的条目
	got, err = mem.Get(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	if got != nil {
		t.Error("Get should return nil for nonexistent entry")
	}
}

func TestBufferMemory_Search(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	entries := []Entry{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "How are you?"},
	}
	mem.SaveBatch(ctx, entries)

	// 基本搜索
	results, err := mem.Search(ctx, SearchQuery{})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("len(results) = %d, want 3", len(results))
	}

	// 限制数量
	results, err = mem.Search(ctx, SearchQuery{Limit: 2})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("len(results) = %d, want 2", len(results))
	}

	// 偏移量
	results, err = mem.Search(ctx, SearchQuery{Offset: 1, Limit: 2})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("len(results) = %d, want 2", len(results))
	}
}

func TestBufferMemory_Search_RoleFilter(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	entries := []Entry{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "How are you?"},
	}
	mem.SaveBatch(ctx, entries)

	results, err := mem.Search(ctx, SearchQuery{Roles: []string{"user"}})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("len(results) = %d, want 2", len(results))
	}
	for _, r := range results {
		if r.Role != "user" {
			t.Errorf("Role = %q, want %q", r.Role, "user")
		}
	}
}

func TestBufferMemory_Search_TimeFilter(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	// 保存第一条
	mem.Save(ctx, Entry{Role: "user", Content: "First"})

	time.Sleep(10 * time.Millisecond)
	midTime := time.Now()
	time.Sleep(10 * time.Millisecond)

	// 保存第二条
	mem.Save(ctx, Entry{Role: "user", Content: "Second"})

	// 只获取 midTime 之后的
	results, err := mem.Search(ctx, SearchQuery{Since: &midTime})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("len(results) = %d, want 1", len(results))
	}
	if results[0].Content != "Second" {
		t.Errorf("Content = %q, want %q", results[0].Content, "Second")
	}
}

func TestBufferMemory_Search_OrderDesc(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	mem.Save(ctx, Entry{ID: "1", Role: "user", Content: "First"})
	time.Sleep(10 * time.Millisecond)
	mem.Save(ctx, Entry{ID: "2", Role: "user", Content: "Second"})

	// 降序
	results, err := mem.Search(ctx, SearchQuery{OrderDesc: true})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if results[0].ID != "2" {
		t.Errorf("results[0].ID = %q, want %q", results[0].ID, "2")
	}
	if results[1].ID != "1" {
		t.Errorf("results[1].ID = %q, want %q", results[1].ID, "1")
	}
}

func TestBufferMemory_Search_OffsetBeyondLength(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	mem.Save(ctx, Entry{Role: "user", Content: "Hello"})

	results, err := mem.Search(ctx, SearchQuery{Offset: 100})
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("len(results) = %d, want 0", len(results))
	}
}

func TestBufferMemory_Delete(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	mem.Save(ctx, Entry{ID: "to-delete", Role: "user", Content: "Hello"})
	mem.Save(ctx, Entry{ID: "keep", Role: "user", Content: "World"})

	if err := mem.Delete(ctx, "to-delete"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}

	stats := mem.Stats()
	if stats.EntryCount != 1 {
		t.Errorf("EntryCount = %d, want 1", stats.EntryCount)
	}

	got, _ := mem.Get(ctx, "to-delete")
	if got != nil {
		t.Error("Deleted entry should not exist")
	}

	got, _ = mem.Get(ctx, "keep")
	if got == nil {
		t.Error("Kept entry should exist")
	}
}

func TestBufferMemory_Delete_Nonexistent(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	// 删除不存在的条目不应该报错
	if err := mem.Delete(ctx, "nonexistent"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}
}

func TestBufferMemory_Clear(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	mem.Save(ctx, Entry{Role: "user", Content: "Hello"})
	mem.Save(ctx, Entry{Role: "assistant", Content: "Hi"})

	if err := mem.Clear(ctx); err != nil {
		t.Fatalf("Clear error: %v", err)
	}

	stats := mem.Stats()
	if stats.EntryCount != 0 {
		t.Errorf("EntryCount = %d, want 0", stats.EntryCount)
	}
}

func TestBufferMemory_Stats(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	// 空记忆
	stats := mem.Stats()
	if stats.EntryCount != 0 {
		t.Errorf("EntryCount = %d, want 0", stats.EntryCount)
	}
	if stats.OldestEntry != nil {
		t.Error("OldestEntry should be nil for empty memory")
	}
	if stats.NewestEntry != nil {
		t.Error("NewestEntry should be nil for empty memory")
	}

	// 添加条目
	mem.Save(ctx, Entry{Role: "user", Content: "First"})
	time.Sleep(10 * time.Millisecond)
	mem.Save(ctx, Entry{Role: "user", Content: "Second"})

	stats = mem.Stats()
	if stats.EntryCount != 2 {
		t.Errorf("EntryCount = %d, want 2", stats.EntryCount)
	}
	if stats.OldestEntry == nil {
		t.Error("OldestEntry should not be nil")
	}
	if stats.NewestEntry == nil {
		t.Error("NewestEntry should not be nil")
	}
	if !stats.OldestEntry.Before(*stats.NewestEntry) {
		t.Error("OldestEntry should be before NewestEntry")
	}
}

func TestBufferMemory_Entries(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	mem.Save(ctx, Entry{Role: "user", Content: "Hello"})
	mem.Save(ctx, Entry{Role: "assistant", Content: "Hi"})

	entries := mem.Entries()
	if len(entries) != 2 {
		t.Errorf("len(Entries) = %d, want 2", len(entries))
	}

	// 验证是副本
	entries[0].Content = "Modified"
	original := mem.Entries()
	if original[0].Content == "Modified" {
		t.Error("Entries() should return a copy")
	}
}

func TestBufferMemory_Last(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		mem.Save(ctx, Entry{ID: string(rune('A' + i)), Role: "user", Content: string(rune('A' + i))})
	}

	// 获取最后 3 条
	last := mem.Last(3)
	if len(last) != 3 {
		t.Errorf("len(Last(3)) = %d, want 3", len(last))
	}
	if last[0].ID != "C" {
		t.Errorf("last[0].ID = %q, want %q", last[0].ID, "C")
	}
	if last[2].ID != "E" {
		t.Errorf("last[2].ID = %q, want %q", last[2].ID, "E")
	}

	// 获取超过总数的
	last = mem.Last(100)
	if len(last) != 5 {
		t.Errorf("len(Last(100)) = %d, want 5", len(last))
	}
}

func TestBufferMemory_WithIDGenerator(t *testing.T) {
	counter := 0
	mem := NewBuffer(10, WithIDGenerator(func() string {
		counter++
		return "custom-" + string(rune('0'+counter))
	}))
	ctx := context.Background()

	mem.Save(ctx, Entry{Role: "user", Content: "Hello"})
	mem.Save(ctx, Entry{Role: "user", Content: "World"})

	entries := mem.Entries()
	if entries[0].ID != "custom-1" {
		t.Errorf("entries[0].ID = %q, want %q", entries[0].ID, "custom-1")
	}
	if entries[1].ID != "custom-2" {
		t.Errorf("entries[1].ID = %q, want %q", entries[1].ID, "custom-2")
	}
}

func TestConvenienceFunctions(t *testing.T) {
	tests := []struct {
		name    string
		entry   Entry
		role    string
		content string
	}{
		{"NewEntry", NewEntry("custom", "content"), "custom", "content"},
		{"NewUserEntry", NewUserEntry("user content"), "user", "user content"},
		{"NewAssistantEntry", NewAssistantEntry("assistant content"), "assistant", "assistant content"},
		{"NewSystemEntry", NewSystemEntry("system content"), "system", "system content"},
		{"NewToolEntry", NewToolEntry("tool content"), "tool", "tool content"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.entry.Role != tt.role {
				t.Errorf("Role = %q, want %q", tt.entry.Role, tt.role)
			}
			if tt.entry.Content != tt.content {
				t.Errorf("Content = %q, want %q", tt.entry.Content, tt.content)
			}
			if tt.entry.CreatedAt.IsZero() {
				t.Error("CreatedAt should be set")
			}
		})
	}
}

func TestBufferMemory_Concurrent(t *testing.T) {
	mem := NewBuffer(100)
	ctx := context.Background()

	// 并发写入
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 10; j++ {
				mem.Save(ctx, Entry{
					Role:    "user",
					Content: "msg",
				})
			}
			done <- true
		}(i)
	}

	// 等待完成
	for i := 0; i < 10; i++ {
		<-done
	}

	// 由于容量限制，应该有 100 条
	stats := mem.Stats()
	if stats.EntryCount != 100 {
		t.Errorf("EntryCount = %d, want 100", stats.EntryCount)
	}
}
