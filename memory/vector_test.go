package memory

import (
	"context"
	"testing"
)

// mockEmbedder 测试用的嵌入器
type mockEmbedder struct {
	dimension int
}

func newMockEmbedder(dimension int) *mockEmbedder {
	return &mockEmbedder{dimension: dimension}
}

func (m *mockEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		// 简单的模拟：使用文本长度生成向量
		embedding := make([]float32, m.dimension)
		for j := 0; j < m.dimension; j++ {
			embedding[j] = float32(len(text)+i+j) / 100.0
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

func TestVectorMemory_Save(t *testing.T) {
	embedder := newMockEmbedder(8)
	mem := NewVectorMemory(embedder, WithDimension(8))

	ctx := context.Background()

	err := mem.Save(ctx, NewUserEntry("Hello world"))
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}

	stats := mem.Stats()
	if stats.EntryCount != 1 {
		t.Errorf("expected 1 entry, got %d", stats.EntryCount)
	}
}

func TestVectorMemory_Search(t *testing.T) {
	embedder := newMockEmbedder(8)
	mem := NewVectorMemory(embedder, WithDimension(8), WithMinScore(0.0))

	ctx := context.Background()

	// 保存多个条目
	mem.Save(ctx, NewUserEntry("Hello world"))
	mem.Save(ctx, NewAssistantEntry("Hi there"))
	mem.Save(ctx, NewUserEntry("How are you"))

	// 搜索
	entries, err := mem.Search(ctx, SearchQuery{
		Query: "Hello",
		Limit: 10,
	})

	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(entries) == 0 {
		t.Error("expected some results")
	}
}

func TestVectorMemory_SemanticSearch(t *testing.T) {
	embedder := newMockEmbedder(8)
	mem := NewVectorMemory(embedder, WithDimension(8), WithMinScore(0.0))

	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("The weather is nice today"))
	mem.Save(ctx, NewUserEntry("I love programming"))
	mem.Save(ctx, NewUserEntry("Machine learning is interesting"))

	entries, err := mem.SemanticSearch(ctx, "coding", 2)
	if err != nil {
		t.Fatalf("semantic search failed: %v", err)
	}

	if len(entries) == 0 {
		t.Error("expected some results")
	}
}

func TestVectorMemory_Delete(t *testing.T) {
	embedder := newMockEmbedder(8)
	mem := NewVectorMemory(embedder, WithDimension(8))

	ctx := context.Background()

	entry := NewUserEntry("Test entry")
	entry.ID = "test-id"
	mem.Save(ctx, entry)

	// 删除
	err := mem.Delete(ctx, "test-id")
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	// 确认删除
	found, _ := mem.Get(ctx, "test-id")
	if found != nil {
		t.Error("entry should be deleted")
	}
}

func TestMemoryVectorStore(t *testing.T) {
	store := NewMemoryVectorStore(8)
	ctx := context.Background()

	// 添加向量
	embedding := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	err := store.Add(ctx, "v1", embedding, map[string]any{"key": "value"})
	if err != nil {
		t.Fatalf("add failed: %v", err)
	}

	// 搜索
	results, err := store.Search(ctx, embedding, 5)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}

	// 相同向量应该有完美相似度
	if results[0].Score < 0.99 {
		t.Errorf("expected high similarity, got %f", results[0].Score)
	}

	// 计数
	count, _ := store.Count(ctx)
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}

	// 删除
	store.Delete(ctx, "v1")
	count, _ = store.Count(ctx)
	if count != 0 {
		t.Errorf("expected count 0 after delete, got %d", count)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// 相同向量
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	sim := cosineSimilarity(a, b)
	if sim < 0.99 {
		t.Errorf("expected ~1.0, got %f", sim)
	}

	// 正交向量
	c := []float32{1, 0, 0}
	d := []float32{0, 1, 0}
	sim = cosineSimilarity(c, d)
	if sim > 0.01 {
		t.Errorf("expected ~0.0, got %f", sim)
	}

	// 相反向量
	e := []float32{1, 0, 0}
	f := []float32{-1, 0, 0}
	sim = cosineSimilarity(e, f)
	if sim > -0.99 {
		t.Errorf("expected ~-1.0, got %f", sim)
	}
}
