package vector

import (
	"context"
	"testing"
)

func TestNewMemoryStore(t *testing.T) {
	store := NewMemoryStore(128)

	if store == nil {
		t.Fatal("expected non-nil store")
	}

	if store.Dimension() != 128 {
		t.Errorf("expected dimension 128, got %d", store.Dimension())
	}
}

func TestMemoryStoreAdd(t *testing.T) {
	store := NewMemoryStore(3)
	ctx := context.Background()

	docs := []Document{
		{ID: "1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3}},
		{ID: "2", Content: "World", Embedding: []float32{0.4, 0.5, 0.6}},
	}

	err := store.Add(ctx, docs)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	count, err := store.Count(ctx)
	if err != nil {
		t.Fatalf("Count failed: %v", err)
	}

	if count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}
}

func TestMemoryStoreGet(t *testing.T) {
	store := NewMemoryStore(3)
	ctx := context.Background()

	docs := []Document{
		{ID: "1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3}},
	}
	store.Add(ctx, docs)

	// 获取存在的文档
	doc, err := store.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if doc == nil {
		t.Fatal("expected non-nil document")
	}

	if doc.Content != "Hello" {
		t.Errorf("expected content 'Hello', got '%s'", doc.Content)
	}

	// 获取不存在的文档
	doc, err = store.Get(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if doc != nil {
		t.Error("expected nil for nonexistent document")
	}
}

func TestMemoryStoreSearch(t *testing.T) {
	store := NewMemoryStore(3)
	ctx := context.Background()

	docs := []Document{
		{ID: "1", Content: "Hello", Embedding: []float32{1.0, 0.0, 0.0}},
		{ID: "2", Content: "World", Embedding: []float32{0.0, 1.0, 0.0}},
		{ID: "3", Content: "Test", Embedding: []float32{0.0, 0.0, 1.0}},
	}
	store.Add(ctx, docs)

	// 搜索与第一个文档相似的
	query := []float32{0.9, 0.1, 0.0}
	results, err := store.Search(ctx, query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	if results[0].ID != "1" {
		t.Errorf("expected first result ID '1', got '%s'", results[0].ID)
	}
}

func TestMemoryStoreSearchWithFilter(t *testing.T) {
	store := NewMemoryStore(3)
	ctx := context.Background()

	docs := []Document{
		{ID: "1", Content: "Hello", Embedding: []float32{1.0, 0.0, 0.0}, Metadata: map[string]any{"type": "greeting"}},
		{ID: "2", Content: "Bye", Embedding: []float32{0.9, 0.1, 0.0}, Metadata: map[string]any{"type": "farewell"}},
	}
	store.Add(ctx, docs)

	query := []float32{1.0, 0.0, 0.0}

	// 使用过滤条件
	results, err := store.Search(ctx, query, 10, WithFilter(map[string]any{"type": "farewell"}))
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("expected 1 result with filter, got %d", len(results))
	}

	if len(results) > 0 && results[0].ID != "2" {
		t.Errorf("expected result ID '2', got '%s'", results[0].ID)
	}
}

func TestMemoryStoreDelete(t *testing.T) {
	store := NewMemoryStore(3)
	ctx := context.Background()

	docs := []Document{
		{ID: "1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3}},
		{ID: "2", Content: "World", Embedding: []float32{0.4, 0.5, 0.6}},
	}
	store.Add(ctx, docs)

	// 删除一个文档
	err := store.Delete(ctx, []string{"1"})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	count, _ := store.Count(ctx)
	if count != 1 {
		t.Errorf("expected count 1 after delete, got %d", count)
	}

	// 验证删除的文档不存在
	doc, _ := store.Get(ctx, "1")
	if doc != nil {
		t.Error("expected deleted document to be nil")
	}
}

func TestMemoryStoreClear(t *testing.T) {
	store := NewMemoryStore(3)
	ctx := context.Background()

	docs := []Document{
		{ID: "1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3}},
		{ID: "2", Content: "World", Embedding: []float32{0.4, 0.5, 0.6}},
	}
	store.Add(ctx, docs)

	// 清空存储
	err := store.Clear(ctx)
	if err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	count, _ := store.Count(ctx)
	if count != 0 {
		t.Errorf("expected count 0 after clear, got %d", count)
	}
}

func TestMemoryStoreClose(t *testing.T) {
	store := NewMemoryStore(3)

	err := store.Close()
	if err != nil {
		t.Errorf("Close should not return error, got %v", err)
	}
}

func TestSearchOptions(t *testing.T) {
	cfg := &SearchConfig{}

	WithFilter(map[string]any{"key": "value"})(cfg)
	if cfg.Filter["key"] != "value" {
		t.Error("WithFilter failed")
	}

	WithMinScore(0.5)(cfg)
	if cfg.MinScore != 0.5 {
		t.Error("WithMinScore failed")
	}

	WithEmbedding(true)(cfg)
	if !cfg.IncludeEmbedding {
		t.Error("WithEmbedding failed")
	}

	WithMetadata(true)(cfg)
	if !cfg.IncludeMetadata {
		t.Error("WithMetadata failed")
	}
}

func TestCosineSimilarity(t *testing.T) {
	// 相同向量，相似度应为 1
	a := []float32{1.0, 0.0, 0.0}
	b := []float32{1.0, 0.0, 0.0}
	sim := cosineSimilarity(a, b)
	if sim < 0.99 || sim > 1.01 {
		t.Errorf("expected similarity ~1.0 for identical vectors, got %f", sim)
	}

	// 正交向量，相似度应为 0
	c := []float32{1.0, 0.0, 0.0}
	d := []float32{0.0, 1.0, 0.0}
	sim = cosineSimilarity(c, d)
	if sim < -0.01 || sim > 0.01 {
		t.Errorf("expected similarity ~0.0 for orthogonal vectors, got %f", sim)
	}

	// 不同长度的向量
	e := []float32{1.0, 0.0}
	f := []float32{1.0, 0.0, 0.0}
	sim = cosineSimilarity(e, f)
	if sim != 0 {
		t.Errorf("expected similarity 0 for different length vectors, got %f", sim)
	}
}

func TestMatchFilter(t *testing.T) {
	metadata := map[string]any{
		"type":   "greeting",
		"lang":   "en",
		"active": true,
	}

	// 空过滤条件
	if !matchFilter(metadata, nil) {
		t.Error("empty filter should match")
	}

	// 匹配的过滤条件
	if !matchFilter(metadata, map[string]any{"type": "greeting"}) {
		t.Error("matching filter should return true")
	}

	// 不匹配的过滤条件
	if matchFilter(metadata, map[string]any{"type": "farewell"}) {
		t.Error("non-matching filter should return false")
	}

	// 多个条件
	if !matchFilter(metadata, map[string]any{"type": "greeting", "lang": "en"}) {
		t.Error("multiple matching conditions should return true")
	}

	// nil metadata
	if matchFilter(nil, map[string]any{"type": "greeting"}) {
		t.Error("nil metadata should not match non-empty filter")
	}

	if !matchFilter(nil, map[string]any{}) {
		t.Error("nil metadata should match empty filter")
	}
}

func TestDocument(t *testing.T) {
	doc := Document{
		ID:      "test-1",
		Content: "Test content",
		Embedding: []float32{0.1, 0.2, 0.3},
		Metadata: map[string]any{
			"source": "test",
		},
		Score: 0.95,
	}

	if doc.ID != "test-1" {
		t.Errorf("expected ID 'test-1', got '%s'", doc.ID)
	}

	if doc.Content != "Test content" {
		t.Errorf("expected Content 'Test content', got '%s'", doc.Content)
	}

	if len(doc.Embedding) != 3 {
		t.Errorf("expected 3 embedding values, got %d", len(doc.Embedding))
	}

	if doc.Metadata["source"] != "test" {
		t.Error("expected metadata source 'test'")
	}

	if doc.Score != 0.95 {
		t.Errorf("expected Score 0.95, got %f", doc.Score)
	}
}
