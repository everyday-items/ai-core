package memory

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

// mockEntityExtractor 模拟实体提取器
type mockEntityExtractor struct {
	result *ExtractionResult
	err    error
}

func (m *mockEntityExtractor) Extract(ctx context.Context, text string) (*ExtractionResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.result, nil
}

func TestNewEntityMemory(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	if mem == nil {
		t.Fatal("NewEntityMemory returned nil")
	}

	if mem.buffer == nil {
		t.Error("buffer should not be nil")
	}

	if mem.entities == nil {
		t.Error("entities map should not be nil")
	}

	stats := mem.EntityStats()
	if stats.TotalEntities != 0 {
		t.Errorf("expected 0 entities, got %d", stats.TotalEntities)
	}
}

func TestEntityMemory_SaveAndGet(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor,
		WithAsyncExtraction(false),
		WithBatchSize(1),
	)

	ctx := context.Background()

	// 保存条目
	entry := NewUserEntry("Hello, I'm John from Google.")
	entry.ID = "test-1"

	if err := mem.Save(ctx, entry); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// 获取条目
	got, err := mem.Get(ctx, "test-1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if got == nil {
		t.Fatal("expected entry, got nil")
	}

	if got.Content != entry.Content {
		t.Errorf("expected content %q, got %q", entry.Content, got.Content)
	}
}

func TestEntityMemory_AddEntity(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	// 手动添加实体
	entity := &Entity{
		Name:        "张三",
		Type:        EntityTypePerson,
		Description: "软件工程师",
		Attributes: map[string]any{
			"公司": "Google",
			"职位": "高级工程师",
		},
	}

	mem.AddEntity(entity)

	// 获取实体
	got, ok := mem.GetEntity("张三")
	if !ok {
		t.Fatal("expected entity, got none")
	}

	if got.Name != "张三" {
		t.Errorf("expected name '张三', got %q", got.Name)
	}

	if got.Type != EntityTypePerson {
		t.Errorf("expected type person, got %v", got.Type)
	}

	// 测试不区分大小写
	got2, ok := mem.GetEntity("  张三  ") // 带空格
	if !ok || got2.Name != "张三" {
		t.Error("entity name normalization failed")
	}
}

func TestEntityMemory_AddRelation(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	// 添加实体
	mem.AddEntity(&Entity{Name: "Alice", Type: EntityTypePerson})
	mem.AddEntity(&Entity{Name: "Google", Type: EntityTypeOrganization})

	// 添加关系
	err := mem.AddRelation("Alice", "Google", "works_at", "Alice works at Google")
	if err != nil {
		t.Fatalf("AddRelation failed: %v", err)
	}

	// 验证关系
	alice, _ := mem.GetEntity("Alice")
	if len(alice.Relations) != 1 {
		t.Fatalf("expected 1 relation, got %d", len(alice.Relations))
	}

	rel := alice.Relations[0]
	if rel.Type != "works_at" {
		t.Errorf("expected relation type 'works_at', got %q", rel.Type)
	}
	if rel.TargetName != "Google" {
		t.Errorf("expected target 'Google', got %q", rel.TargetName)
	}
}

func TestEntityMemory_GetRelatedEntities(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	// 添加实体和关系
	mem.AddEntity(&Entity{Name: "Alice", Type: EntityTypePerson})
	mem.AddEntity(&Entity{Name: "Bob", Type: EntityTypePerson})
	mem.AddEntity(&Entity{Name: "Google", Type: EntityTypeOrganization})

	mem.AddRelation("Alice", "Google", "works_at", "")
	mem.AddRelation("Alice", "Bob", "knows", "")

	// 获取相关实体
	related := mem.GetRelatedEntities("Alice")
	if len(related) != 2 {
		t.Fatalf("expected 2 related entities, got %d", len(related))
	}

	// 检查相关实体
	names := make(map[string]bool)
	for _, e := range related {
		names[e.Name] = true
	}

	if !names["Bob"] || !names["Google"] {
		t.Error("expected Bob and Google in related entities")
	}
}

func TestEntityMemory_SearchEntities(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	// 添加实体
	mem.AddEntity(&Entity{Name: "张三", Type: EntityTypePerson, Description: "软件工程师"})
	mem.AddEntity(&Entity{Name: "李四", Type: EntityTypePerson, Description: "产品经理"})
	mem.AddEntity(&Entity{Name: "Google", Type: EntityTypeOrganization, Description: "科技公司"})

	// 按名称搜索
	results := mem.SearchEntities("张")
	if len(results) != 1 {
		t.Errorf("expected 1 result for '张', got %d", len(results))
	}

	// 按描述搜索
	results = mem.SearchEntities("工程师")
	if len(results) != 1 {
		t.Errorf("expected 1 result for '工程师', got %d", len(results))
	}

	// 搜索不存在的
	results = mem.SearchEntities("不存在")
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestEntityMemory_GetEntitiesByType(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	// 添加实体
	mem.AddEntity(&Entity{Name: "Alice", Type: EntityTypePerson})
	mem.AddEntity(&Entity{Name: "Bob", Type: EntityTypePerson})
	mem.AddEntity(&Entity{Name: "Google", Type: EntityTypeOrganization})

	// 按类型获取
	persons := mem.GetEntitiesByType(EntityTypePerson)
	if len(persons) != 2 {
		t.Errorf("expected 2 persons, got %d", len(persons))
	}

	orgs := mem.GetEntitiesByType(EntityTypeOrganization)
	if len(orgs) != 1 {
		t.Errorf("expected 1 organization, got %d", len(orgs))
	}
}

func TestEntityMemory_ExtractEntities(t *testing.T) {
	// 设置模拟提取结果
	extractor := &mockEntityExtractor{
		result: &ExtractionResult{
			Entities: []ExtractedEntity{
				{Name: "John", Type: EntityTypePerson, Description: "Engineer"},
				{Name: "Google", Type: EntityTypeOrganization, Description: "Tech company"},
			},
			Relations: []ExtractedRelation{
				{SourceName: "John", TargetName: "Google", Type: "works_at"},
			},
		},
	}

	mem := NewEntityMemory(extractor,
		WithAsyncExtraction(false),
		WithBatchSize(1),
	)

	ctx := context.Background()

	// 保存条目触发提取
	entry := NewUserEntry("John works at Google.")
	entry.ID = "test-entry-1"
	if err := mem.Save(ctx, entry); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// 验证实体被提取
	john, ok := mem.GetEntity("John")
	if !ok {
		t.Fatal("expected John entity")
	}
	if john.Description != "Engineer" {
		t.Errorf("expected description 'Engineer', got %q", john.Description)
	}

	google, ok := mem.GetEntity("Google")
	if !ok {
		t.Fatal("expected Google entity")
	}

	// 验证关系
	if len(john.Relations) != 1 {
		t.Fatalf("expected 1 relation, got %d", len(john.Relations))
	}
	if john.Relations[0].TargetName != "Google" {
		t.Errorf("expected target 'Google', got %q", john.Relations[0].TargetName)
	}

	// 验证来源
	if len(john.Sources) == 0 {
		t.Error("expected sources to be recorded")
	}

	// 验证统计
	stats := mem.EntityStats()
	if stats.TotalEntities != 2 {
		t.Errorf("expected 2 entities, got %d", stats.TotalEntities)
	}
	if stats.TotalRelations != 1 {
		t.Errorf("expected 1 relation, got %d", stats.TotalRelations)
	}

	_ = google // 使用变量避免警告
}

func TestEntityMemory_GetContextWithEntities(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	// 添加带属性和关系的实体
	mem.AddEntity(&Entity{
		Name:        "Alice",
		Type:        EntityTypePerson,
		Description: "高级软件工程师",
		Attributes: map[string]any{
			"部门": "研发部",
			"级别": "P7",
		},
	})
	mem.AddEntity(&Entity{Name: "Google", Type: EntityTypeOrganization})
	mem.AddRelation("Alice", "Google", "works_at", "")

	// 获取上下文
	context := mem.GetContextWithEntities("Alice")

	// 验证包含关键信息
	if context == "" {
		t.Fatal("expected non-empty context")
	}
	if !contains(context, "Alice") {
		t.Error("context should contain entity name")
	}
	if !contains(context, "person") {
		t.Error("context should contain entity type")
	}
	if !contains(context, "高级软件工程师") {
		t.Error("context should contain description")
	}
	if !contains(context, "works_at") {
		t.Error("context should contain relation")
	}
}

func TestEntityMemory_Clear(t *testing.T) {
	extractor := &mockEntityExtractor{}
	mem := NewEntityMemory(extractor)

	ctx := context.Background()

	// 添加数据
	mem.AddEntity(&Entity{Name: "Test", Type: EntityTypePerson})
	mem.Save(ctx, NewUserEntry("test"))

	// 清空
	if err := mem.Clear(ctx); err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	// 验证已清空
	if len(mem.GetEntities()) != 0 {
		t.Error("expected 0 entities after clear")
	}

	stats := mem.Stats()
	if stats.EntryCount != 0 {
		t.Errorf("expected 0 entries after clear, got %d", stats.EntryCount)
	}
}

func TestLLMEntityExtractor(t *testing.T) {
	// 模拟 LLM 响应
	mockResponse := `{
		"entities": [
			{"name": "张三", "type": "person", "description": "项目经理"}
		],
		"relations": [
			{"source_name": "张三", "target_name": "项目A", "type": "manages"}
		]
	}`

	extractor := NewLLMEntityExtractor(func(ctx context.Context, prompt string) (string, error) {
		return mockResponse, nil
	})

	result, err := extractor.Extract(context.Background(), "张三是项目A的项目经理")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if len(result.Entities) != 1 {
		t.Errorf("expected 1 entity, got %d", len(result.Entities))
	}

	if result.Entities[0].Name != "张三" {
		t.Errorf("expected name '张三', got %q", result.Entities[0].Name)
	}

	if len(result.Relations) != 1 {
		t.Errorf("expected 1 relation, got %d", len(result.Relations))
	}
}

func TestLLMEntityExtractor_WithExtraText(t *testing.T) {
	// 模拟 LLM 响应（包含额外文本）
	mockResponse := `好的，我来提取实体：

{
	"entities": [{"name": "Test", "type": "person", "description": "test"}],
	"relations": []
}

以上是提取结果。`

	extractor := NewLLMEntityExtractor(func(ctx context.Context, prompt string) (string, error) {
		return mockResponse, nil
	})

	result, err := extractor.Extract(context.Background(), "test")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if len(result.Entities) != 1 {
		t.Errorf("expected 1 entity, got %d", len(result.Entities))
	}
}

func TestEntity_JSON(t *testing.T) {
	entity := &Entity{
		Name:        "Test",
		Type:        EntityTypePerson,
		Description: "Test entity",
		Attributes: map[string]any{
			"key": "value",
		},
		Relations: []EntityRelation{
			{Type: "knows", TargetName: "Other"},
		},
		FirstMentioned: time.Now(),
		LastMentioned:  time.Now(),
		MentionCount:   1,
	}

	// 序列化
	data, err := json.Marshal(entity)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	// 反序列化
	var decoded Entity
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if decoded.Name != entity.Name {
		t.Errorf("expected name %q, got %q", entity.Name, decoded.Name)
	}
	if decoded.Type != entity.Type {
		t.Errorf("expected type %v, got %v", entity.Type, decoded.Type)
	}
}

// 辅助函数
func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsInner(s, substr)))
}

func containsInner(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
