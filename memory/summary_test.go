package memory

import (
	"context"
	"strings"
	"testing"
)

// mockSummarizer 测试用的摘要器
type mockSummarizer struct {
	calls int
}

func (m *mockSummarizer) Summarize(ctx context.Context, content string) (string, error) {
	m.calls++
	// 简单的摘要：取前50个字符
	if len(content) > 50 {
		return content[:50] + "...", nil
	}
	return content, nil
}

func TestSummaryMemory_Save(t *testing.T) {
	summarizer := &mockSummarizer{}
	mem := NewSummaryMemory(summarizer, WithMaxEntries(5), WithKeepRecent(2))

	ctx := context.Background()

	// 保存条目
	for i := 0; i < 3; i++ {
		err := mem.Save(ctx, NewUserEntry("message "+string(rune('A'+i))))
		if err != nil {
			t.Fatalf("save failed: %v", err)
		}
	}

	stats := mem.Stats()
	if stats.EntryCount != 3 {
		t.Errorf("expected 3 entries, got %d", stats.EntryCount)
	}
}

func TestSummaryMemory_AutoSummarize(t *testing.T) {
	summarizer := &mockSummarizer{}
	mem := NewSummaryMemory(summarizer, WithMaxEntries(3), WithKeepRecent(1))

	ctx := context.Background()

	// 保存超过阈值的条目
	for i := 0; i < 5; i++ {
		mem.Save(ctx, NewUserEntry("message "+string(rune('A'+i))))
	}

	// 应该触发了摘要
	if summarizer.calls == 0 {
		t.Error("expected summarizer to be called")
	}

	// 检查摘要存在
	summary := mem.GetSummary()
	if summary == "" {
		t.Error("expected summary to be generated")
	}
}

func TestSummaryMemory_GetContext(t *testing.T) {
	summarizer := &mockSummarizer{}
	mem := NewSummaryMemory(summarizer)

	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("Hello"))
	mem.Save(ctx, NewAssistantEntry("Hi there"))

	context := mem.GetContext()
	if !strings.Contains(context, "Hello") {
		t.Error("context should contain 'Hello'")
	}
	if !strings.Contains(context, "Hi there") {
		t.Error("context should contain 'Hi there'")
	}
}

func TestSummaryMemory_Clear(t *testing.T) {
	summarizer := &mockSummarizer{}
	mem := NewSummaryMemory(summarizer)
	mem.SetSummary("test summary")

	ctx := context.Background()
	mem.Save(ctx, NewUserEntry("test"))

	// Clear
	mem.Clear(ctx)

	if mem.GetSummary() != "" {
		t.Error("summary should be cleared")
	}

	stats := mem.Stats()
	if stats.EntryCount != 0 {
		t.Errorf("expected 0 entries, got %d", stats.EntryCount)
	}
}
