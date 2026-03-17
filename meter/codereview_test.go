package meter

import (
	"sync"
	"testing"
	"time"
)

// ============================================================================
// BUG-1: Stats() 中 TotalRequests 与 EstimatedCost 不一致
// ============================================================================

func TestMeter_StatsInconsistency_AfterPruning(t *testing.T) {
	// 设置小的 maxRecords 使得记录被自动清理
	m := NewWithOptions(nil, 10)

	// 记录 20 条请求
	for i := 0; i < 20; i++ {
		m.Record("gpt-4", 1000, 500)
	}

	stats := m.Stats()

	// TotalRequests 使用 atomic 计数器：反映全部历史 = 20
	if stats.TotalRequests != 20 {
		t.Errorf("expected TotalRequests=20, got %d", stats.TotalRequests)
	}

	// EstimatedCost 遍历 records：只反映剩余记录（约 9-10 条）
	// 所以 TotalRequests 说有 20 个请求，但 EstimatedCost 只算了一半
	records := m.Records()
	t.Logf("BUG: TotalRequests=%d（历史累计），但 EstimatedCost 只基于 %d 条剩余记录。"+
		"成本被严重低估", stats.TotalRequests, len(records))

	// TotalTokens 也是 atomic：20 * (1000+500) = 30000
	if stats.TotalTokens != 30000 {
		t.Errorf("expected TotalTokens=30000, got %d", stats.TotalTokens)
	}
}

// ============================================================================
// BUG-2: Meter.Clear() 不是原子操作
// ============================================================================

func TestMeter_Clear_Concurrent(t *testing.T) {
	m := New()

	var wg sync.WaitGroup

	// 并发记录
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m.Record("gpt-4", 100, 50)
		}()
	}

	// 并发清零
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(time.Millisecond)
		m.Clear()
	}()

	wg.Wait()

	// Clear 先清 records（持锁），再清 atomic counters（无锁）
	// 在两步之间，其他 goroutine 可能已经写入新记录
	// 导致 records 和 counters 不同步
	stats := m.Stats()
	records := m.Records()

	if stats.TotalRequests > 0 && len(records) == 0 {
		t.Logf("WARNING: Clear 后 TotalRequests=%d 但 records=%d，"+
			"说明 Clear 不是原子操作", stats.TotalRequests, len(records))
	}
}

// ============================================================================
// BUG-3: Report() 两次获取锁，可能看到不同状态
// ============================================================================

func TestMeter_Report_Consistency(t *testing.T) {
	m := New()

	// Report() 调用 m.Stats() 和 m.AllModelStats()
	// 两次独立获取锁，中间可能有其他写入
	// 导致 overall stats 和 model stats 的数字不一致

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			m.Record("gpt-4", 100, 50)
		}
	}()

	// 在写入的同时生成报告
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			_ = m.Report()
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()

	t.Log("DESIGN: Report() 调用 Stats() 和 AllModelStats() 分别获取锁，" +
		"两个快照可能不一致")
}

// ============================================================================
// BUG-4: JSON() 同样有快照不一致问题
// ============================================================================

func TestMeter_JSON_Consistency(t *testing.T) {
	m := New()

	m.Record("gpt-4", 1000, 500)
	m.Record("claude-3-opus", 2000, 800)

	data, err := m.JSON()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("expected non-empty JSON")
	}

	// JSON() 内部 m.Stats() 和 m.AllModelStats() 分别获取锁
	t.Logf("JSON output length: %d bytes", len(data))
}

// ============================================================================
// BUG-5: RecordWithDetails 的 auto-cleanup 会删除 10%，但边界情况
// ============================================================================

func TestMeter_AutoCleanup_Boundary(t *testing.T) {
	// maxRecords=10, 10/10=1, 删除 1 条
	m := NewWithOptions(nil, 10)

	for i := 0; i < 11; i++ {
		m.Record("gpt-4", 100, 50)
	}

	records := m.Records()
	// 应该有 10 条（11 - 删除的 1 条）
	if len(records) != 10 {
		t.Errorf("expected 10 records after cleanup, got %d", len(records))
	}
}

func TestMeter_AutoCleanup_MaxRecordsOne(t *testing.T) {
	// maxRecords=1, 1/10=0, deleteCount 应该至少为 1
	m := NewWithOptions(nil, 1)

	m.Record("gpt-4", 100, 50)
	m.Record("gpt-4", 200, 100)

	records := m.Records()
	if len(records) > 1 {
		t.Errorf("maxRecords=1 后应最多 1 条记录，实际 %d 条", len(records))
	}
}

// ============================================================================
// BUG-6: Tracker 没有防止多次调用 Done/Error
// ============================================================================

func TestTracker_DoubleDone(t *testing.T) {
	m := New()
	tracker := m.NewTracker("gpt-4").SetInputTokens(100)

	tracker.Done(50)
	tracker.Done(50) // 第二次调用会重复记录

	stats := m.Stats()
	if stats.TotalRequests != 2 {
		t.Logf("Tracker.Done() 被调用两次，记录了 %d 条（期望用户不会这样做，但没有保护机制）",
			stats.TotalRequests)
	} else {
		t.Log("BUG: Tracker.Done() 没有防重复机制，多次调用会重复记录统计数据")
	}
}

// ============================================================================
// BENCHMARK: Meter.Record 在高并发下的性能
// ============================================================================

func BenchmarkMeter_Record_Parallel(b *testing.B) {
	m := New()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Record("gpt-4", 100, 50)
		}
	})
}

func BenchmarkMeter_Stats_Parallel(b *testing.B) {
	m := New()
	// 预填充
	for i := 0; i < 1000; i++ {
		m.Record("gpt-4", 100, 50)
	}

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = m.Stats()
		}
	})
}

func BenchmarkMeter_RecordAndStats_Mixed(b *testing.B) {
	m := New()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%10 == 0 {
				_ = m.Stats()
			} else {
				m.Record("gpt-4", 100, 50)
			}
			i++
		}
	})
}
