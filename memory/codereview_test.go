package memory

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ============================================================================
// BUG-1: generateVectorID 使用 UnixNano — 高并发下 ID 碰撞
// ============================================================================

func TestGenerateVectorID_ConcurrentCollision(t *testing.T) {
	ids := make(map[string]bool)
	var mu sync.Mutex
	var collisions atomic.Int32

	var wg sync.WaitGroup
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := generateVectorID()
			mu.Lock()
			if ids[id] {
				collisions.Add(1)
			}
			ids[id] = true
			mu.Unlock()
		}()
	}
	wg.Wait()

	if collisions.Load() > 0 {
		t.Errorf("BUG: generateVectorID 产生了 %d 次 ID 碰撞！"+
			"使用 time.Now().UnixNano() 在高并发下不唯一。"+
			"应使用 atomic counter + random（如 BufferMemory.defaultIDGen）",
			collisions.Load())
	}
}

// 对比：BufferMemory 的 defaultIDGen 不会碰撞
func TestDefaultIDGen_NoConcurrentCollision(t *testing.T) {
	ids := make(map[string]bool)
	var mu sync.Mutex
	var collisions atomic.Int32

	var wg sync.WaitGroup
	for i := 0; i < 10000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := defaultIDGen()
			mu.Lock()
			if ids[id] {
				collisions.Add(1)
			}
			ids[id] = true
			mu.Unlock()
		}()
	}
	wg.Wait()

	if collisions.Load() > 0 {
		t.Errorf("defaultIDGen 产生了 %d 次碰撞", collisions.Load())
	}
}

// ============================================================================
// BUG-2: SummaryMemory.doSummarize 存在 TOCTOU 竞态 — 并发 Save 丢数据
// ============================================================================

func TestSummaryMemory_SaveDuringDoSummarize_DataLoss(t *testing.T) {
	// 模拟慢速 summarizer，在摘要期间有新数据写入
	summarizeCalled := make(chan struct{}, 1) // buffered to avoid blocking
	summarizeComplete := make(chan struct{})
	firstCall := true
	var scMu sync.Mutex

	slowSummarizer := NewSimpleSummarizer(func(ctx context.Context, content string) (string, error) {
		scMu.Lock()
		isFirst := firstCall
		firstCall = false
		scMu.Unlock()

		if isFirst {
			select {
			case summarizeCalled <- struct{}{}:
			default:
			}
			select {
			case <-summarizeComplete:
			case <-time.After(2 * time.Second):
				return "timeout summary", nil
			}
		}
		truncated := content
		if len(truncated) > 20 {
			truncated = truncated[:20]
		}
		return "summary of: " + truncated, nil
	})

	mem := NewSummaryMemory(slowSummarizer,
		WithSummaryConfig(SummaryConfig{
			MaxEntries:         5,
			KeepRecent:         2,
			SummaryPrompt:      "%s",
			ProgressiveSummary: false,
			BufferCapacity:     100,
		}),
	)

	ctx := context.Background()

	// 填入 6 条数据触发摘要
	for i := 0; i < 6; i++ {
		mem.Save(ctx, NewUserEntry(fmt.Sprintf("message-%d", i)))
	}

	// 等待摘要开始（或超时）
	select {
	case <-summarizeCalled:
		// doSummarize 已经快照了 entries 并开始摘要
	case <-time.After(500 * time.Millisecond):
		t.Skip("摘要未触发，跳过")
	}

	// 在摘要执行期间写入新数据
	mem.Save(ctx, NewUserEntry("important-new-message"))

	// 完成摘要
	close(summarizeComplete)
	time.Sleep(100 * time.Millisecond) // 等待 doSummarize 完成

	// 检查 "important-new-message" 是否还在
	entries := mem.Entries()
	found := false
	for _, e := range entries {
		if e.Content == "important-new-message" {
			found = true
			break
		}
	}

	if !found {
		t.Error("BUG: doSummarize 的 TOCTOU 竞态导致数据丢失！" +
			"doSummarize 先读取 entries，然后 Clear + 重存 recent entries，" +
			"期间写入的新数据被 Clear 删除了")
	}
}

// ============================================================================
// BUG-3: SummaryMemory.Stats() 读取 m.summary 时没有持锁 — 数据竞争
// ============================================================================

func TestSummaryMemory_Stats_DataRace(t *testing.T) {
	// 用 `go test -race` 运行才能检测到
	summarizer := NewSimpleSummarizer(func(ctx context.Context, content string) (string, error) {
		return "summary", nil
	})

	mem := NewSummaryMemory(summarizer, WithMaxEntries(3), WithKeepRecent(1))
	ctx := context.Background()

	var wg sync.WaitGroup

	// 并发写入触发摘要
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			mem.Save(ctx, NewUserEntry(fmt.Sprintf("msg-%d", i)))
		}(i)
	}

	// 并发读取 Stats
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = mem.Stats() // Stats() 读取 m.summary 没有加锁
		}()
	}

	wg.Wait()
	// 如果用 -race 运行，这里应该报告 data race on m.summary
}

// ============================================================================
// BUG-4: BufferMemory.Save 的 slice 截断导致内存泄漏
// ============================================================================

func TestBufferMemory_SliceLeak(t *testing.T) {
	// m.entries = m.entries[1:] 不释放底层数组中旧元素的引用
	// 随着不断 append + 截断，底层数组只会增长不会收缩
	mem := NewBuffer(5)
	ctx := context.Background()

	// 保存 100 条，每次保存都会触发截断
	for i := 0; i < 100; i++ {
		mem.Save(ctx, Entry{
			Content: fmt.Sprintf("large content %d with lots of data to simulate real usage", i),
		})
	}

	// 实际只有 5 条数据，但底层数组可能保存了所有 100 条的引用
	entries := mem.Entries()
	if len(entries) != 5 {
		t.Errorf("expected 5 entries, got %d", len(entries))
	}

	t.Log("WARNING: m.entries = m.entries[1:] 只移动 slice header，" +
		"底层数组中被截断的旧元素仍然被引用，无法 GC。" +
		"应该使用 copy + 重新分配来避免内存泄漏")
}

// ============================================================================
// BUG-5: MultiLayerMemory.Save 持锁期间调用外部服务 — 阻塞全局
// ============================================================================

func TestMultiLayerMemory_SaveBlocksDuringTransfer(t *testing.T) {
	// 当 transferShortTermToLongTerm 触发时，会调用 VectorMemory.SaveBatch
	// 而 SaveBatch 会调用 embedder.Embed (外部服务)
	// 整个过程都在 MultiLayerMemory.mu.Lock() 下
	// 导致所有其他操作（Get、Search、Save）被阻塞

	slowEmbedder := NewSimpleEmbedder(func(ctx context.Context, texts []string) ([][]float32, error) {
		time.Sleep(200 * time.Millisecond) // 模拟慢速嵌入服务
		result := make([][]float32, len(texts))
		for i := range texts {
			result[i] = []float32{0.1, 0.2, 0.3}
		}
		return result, nil
	})

	summarizer := NewSimpleSummarizer(func(ctx context.Context, content string) (string, error) {
		return "summary", nil
	})

	mem := NewMultiLayerMemory(
		WithSummarizer(summarizer),
		WithEmbedder(slowEmbedder),
		WithMultiLayerConfig(MultiLayerConfig{
			Working: WorkingConfig{Capacity: 5},
			ShortTerm: ShortTermConfig{
				MaxEntries: 10,
				KeepRecent: 2,
				Summarizer: summarizer,
			},
			LongTerm: LongTermConfig{
				Embedder: slowEmbedder,
				MinScore: 0.5,
				TopK:     5,
			},
			TransferPolicy: TransferPolicy{
				AutoTransfer:                 true,
				WorkingToShortTermThreshold:  4,
				ShortTermToLongTermThreshold: 8,
			},
		}),
	)

	ctx := context.Background()

	// 填充足够的数据触发转移
	for i := 0; i < 20; i++ {
		mem.Save(ctx, NewUserEntry(fmt.Sprintf("message-%d", i)))
	}

	// 在可能的转移期间尝试 Get
	done := make(chan bool, 1)
	go func() {
		start := time.Now()
		mem.Get(ctx, "any-id")
		elapsed := time.Since(start)
		if elapsed > 100*time.Millisecond {
			done <- false // 被阻塞了
		} else {
			done <- true
		}
	}()

	select {
	case fast := <-done:
		if !fast {
			t.Log("CONFIRMED: MultiLayerMemory.Save 持锁期间调用外部服务，" +
				"阻塞了其他操作。应在持锁前完成外部调用")
		}
	case <-time.After(2 * time.Second):
		t.Log("WARNING: Get 操作超时，可能被死锁或长时间阻塞")
	}
}

// ============================================================================
// BUG-6: MultiLayerMemory.Search 中 addLayerMeta 修改了共享 Metadata map
// ============================================================================

func TestMultiLayerMemory_SearchCorruptsOriginalMetadata(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	// 保存一条带 Metadata 的记录
	originalMeta := map[string]any{"key": "value"}
	mem.Save(ctx, Entry{
		ID:       "test-1",
		Role:     "user",
		Content:  "hello",
		Metadata: originalMeta,
	})

	// 执行搜索 — addLayerMeta 会修改 Metadata
	results, err := mem.Search(ctx, SearchQuery{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Fatal("expected results")
	}

	// 检查搜索结果是否添加了 _layer
	if results[0].Metadata["_layer"] == nil {
		t.Fatal("expected _layer in search results")
	}

	// 关键检查：原始 Metadata 是否被污染
	// 因为 Metadata 是 map（引用类型），修改 copy 的 Metadata 会影响原始数据
	entry, _ := mem.Get(ctx, "test-1")
	if entry != nil && entry.Metadata != nil {
		if _, hasLayer := entry.Metadata["_layer"]; hasLayer {
			t.Error("BUG: Search 的 addLayerMeta 污染了原始 Entry 的 Metadata！" +
				"map 是引用类型，BufferMemory.Search 返回值拷贝但 map 指向同一内存。" +
				"应该在 addLayerMeta 中深拷贝 map")
		}
	}
}

// ============================================================================
// BUG-7: VectorMemory.Search — 负 Offset 导致 panic
// ============================================================================

func TestVectorMemory_Search_NegativeOffset(t *testing.T) {
	embedder := NewSimpleEmbedder(func(ctx context.Context, texts []string) ([][]float32, error) {
		result := make([][]float32, len(texts))
		for i := range texts {
			result[i] = []float32{0.1, 0.2, 0.3}
		}
		return result, nil
	})

	mem := NewVectorMemory(embedder, WithDimension(3), WithMinScore(0))
	ctx := context.Background()

	mem.Save(ctx, Entry{Content: "test data"})

	// 普通搜索（不使用向量搜索，走时间排序路径）
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("BUG: 负 Offset 导致 panic: %v。应该 clamp 到 0", r)
		}
	}()

	// 负 Offset 在 non-vector 路径中没有被检查
	_, err := mem.Search(ctx, SearchQuery{Offset: -1, Limit: 10})
	if err != nil {
		t.Logf("search with negative offset returned error: %v", err)
	}
}

// ============================================================================
// BUG-8: EntityMemory.Clear — 在持有 mu 的同时获取 queueMu，可能与异步提取死锁
// ============================================================================

func TestEntityMemory_ConcurrentClearAndExtraction(t *testing.T) {
	// 异步提取正在运行时调用 Clear
	// extractEntities 获取 mu.Lock (line 535)
	// Clear 获取 mu.Lock (line 316) 然后 queueMu.Lock (line 318)
	// 如果时机不对，可能出现竞争（虽然不是死锁，因为锁顺序一致）

	extractCalled := make(chan struct{}, 100)
	extractor := NewSimpleEntityExtractor(func(ctx context.Context, text string) (*ExtractionResult, error) {
		extractCalled <- struct{}{}
		time.Sleep(50 * time.Millisecond) // 模拟慢速提取
		return &ExtractionResult{
			Entities: []ExtractedEntity{{Name: "Test", Type: EntityTypePerson}},
		}, nil
	})

	mem := NewEntityMemory(extractor,
		WithAsyncExtraction(true),
		WithBatchSize(2),
		WithEntityBufferCapacity(100),
	)
	ctx := context.Background()

	// 快速保存触发异步提取
	for i := 0; i < 10; i++ {
		mem.Save(ctx, NewUserEntry(fmt.Sprintf("Person%d works at Company%d", i, i)))
	}

	// 等待至少一个提取开始
	select {
	case <-extractCalled:
	case <-time.After(100 * time.Millisecond):
	}

	// 在提取进行中尝试 Clear
	done := make(chan struct{})
	go func() {
		mem.Clear(ctx)
		close(done)
	}()

	select {
	case <-done:
		// 成功完成
	case <-time.After(5 * time.Second):
		t.Fatal("Clear 超时，可能发生死锁")
	}
}

// ============================================================================
// BUG-9: VectorMemory.SaveBatch 会导致双重嵌入生成
// ============================================================================

func TestVectorMemory_SaveBatch_DoubleEmbedding(t *testing.T) {
	var embedCalls atomic.Int32

	embedder := NewSimpleEmbedder(func(ctx context.Context, texts []string) ([][]float32, error) {
		embedCalls.Add(1)
		result := make([][]float32, len(texts))
		for i := range texts {
			result[i] = []float32{float32(i) * 0.1, 0.2, 0.3}
		}
		return result, nil
	})

	mem := NewVectorMemory(embedder, WithDimension(3), WithMinScore(0))
	ctx := context.Background()

	entries := []Entry{
		{Content: "first message"},
		{Content: "second message"},
		{Content: "third message"},
	}

	mem.SaveBatch(ctx, entries)

	// SaveBatch 先批量生成嵌入（1次调用），然后逐个 Save
	// Save 中检查 len(entry.Embedding) == 0，如果已有则跳过
	// 但是否真的跳过了？
	calls := embedCalls.Load()
	if calls > 1 {
		t.Errorf("BUG: SaveBatch 产生了 %d 次嵌入调用（期望 1 次批量调用）。"+
			"Save() 中重复生成了嵌入", calls)
	} else {
		t.Logf("SaveBatch 正确复用了批量嵌入结果，共 %d 次嵌入调用", calls)
	}
}

// ============================================================================
// BUG-10: BufferMemory.Search 返回的 slice 引用了内部数组
// ============================================================================

func TestBufferMemory_SearchReturnSharedSlice(t *testing.T) {
	mem := NewBuffer(10)
	ctx := context.Background()

	mem.Save(ctx, Entry{ID: "1", Role: "user", Content: "hello"})
	mem.Save(ctx, Entry{ID: "2", Role: "user", Content: "world"})

	results1, _ := mem.Search(ctx, SearchQuery{Limit: 10})
	results2, _ := mem.Search(ctx, SearchQuery{Limit: 10})

	if len(results1) == 0 || len(results2) == 0 {
		t.Fatal("expected results")
	}

	// 修改 results1 的 Metadata 是否影响 results2 或原始数据
	if results1[0].Metadata == nil {
		results1[0].Metadata = make(map[string]any)
	}
	results1[0].Metadata["modified"] = true

	// 检查原始数据是否被污染
	entry, _ := mem.Get(ctx, "1")
	if entry != nil && entry.Metadata != nil {
		if _, has := entry.Metadata["modified"]; has {
			t.Log("NOTE: BufferMemory.Search 返回值的 Metadata map 与原始共享引用，" +
				"但由于 Entry 是值拷贝（struct copy），nil Metadata 不会共享")
		}
	}
}

// ============================================================================
// BENCHMARK: BufferMemory.Save 内存分配（验证 slice 泄漏影响）
// ============================================================================

func BenchmarkBufferMemory_SaveEviction(b *testing.B) {
	mem := NewBuffer(100)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem.Save(ctx, Entry{
			Content: fmt.Sprintf("message %d with some content", i),
		})
	}
}

// ============================================================================
// BENCHMARK: VectorMemory cosine similarity 性能
// ============================================================================

func BenchmarkCosineSimilarity(b *testing.B) {
	// 1536 维度（OpenAI embedding 维度）
	a := make([]float32, 1536)
	bv := make([]float32, 1536)
	for i := range a {
		a[i] = float32(i) * 0.001
		bv[i] = float32(i) * 0.002
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cosineSimilarity(a, bv)
	}
}

// ============================================================================
// BENCHMARK: generateVectorID vs defaultIDGen
// ============================================================================

func BenchmarkGenerateVectorID(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = generateVectorID()
	}
}

func BenchmarkDefaultIDGen(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = defaultIDGen()
	}
}
