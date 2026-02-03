// Package qdrant 提供 Qdrant 向量数据库适配器
//
// Qdrant 是一个高性能的开源向量数据库，支持：
//   - 向量相似度搜索
//   - 元数据过滤
//   - 分布式部署
//
// 使用示例:
//
//	store, err := qdrant.New(qdrant.Config{
//	    Host:       "localhost",
//	    Port:       6333,
//	    Collection: "documents",
//	    Dimension:  1536,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer store.Close()
package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/everyday-items/ai-core/store/vector"
)

// Config Qdrant 配置
type Config struct {
	// Host Qdrant 服务器地址
	Host string

	// Port Qdrant 服务器端口
	Port int

	// Collection 集合名称
	Collection string

	// Dimension 向量维度
	Dimension int

	// APIKey API 密钥（可选，用于 Qdrant Cloud）
	APIKey string

	// HTTPS 是否使用 HTTPS
	HTTPS bool

	// Timeout 请求超时时间
	Timeout time.Duration

	// Distance 距离度量方式：Cosine, Euclid, Dot
	Distance Distance

	// OnDisk 是否将向量存储在磁盘上
	OnDisk bool

	// CreateCollection 如果集合不存在是否自动创建
	CreateCollection bool
}

// Distance 距离度量方式
type Distance string

const (
	// DistanceCosine 余弦距离
	DistanceCosine Distance = "Cosine"
	// DistanceEuclid 欧几里得距离
	DistanceEuclid Distance = "Euclid"
	// DistanceDot 点积
	DistanceDot Distance = "Dot"
)

// Store Qdrant 向量存储
type Store struct {
	config  Config
	client  *http.Client
	baseURL string
}

// New 创建 Qdrant 存储
//
// 参数:
//   - cfg: Qdrant 配置
//
// 返回:
//   - *Store: Qdrant 存储实例
//   - error: 连接失败时返回错误
func New(cfg Config) (*Store, error) {
	// 设置默认值
	if cfg.Host == "" {
		cfg.Host = "localhost"
	}
	if cfg.Port == 0 {
		cfg.Port = 6333
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.Distance == "" {
		cfg.Distance = DistanceCosine
	}

	scheme := "http"
	if cfg.HTTPS {
		scheme = "https"
	}

	s := &Store{
		config:  cfg,
		client:  &http.Client{Timeout: cfg.Timeout},
		baseURL: fmt.Sprintf("%s://%s:%d", scheme, cfg.Host, cfg.Port),
	}

	// 检查连接
	if err := s.healthCheck(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to connect to Qdrant: %w", err)
	}

	// 自动创建集合
	if cfg.CreateCollection {
		if err := s.ensureCollection(context.Background()); err != nil {
			return nil, err
		}
	}

	return s, nil
}

// healthCheck 健康检查
func (s *Store) healthCheck(ctx context.Context) error {
	_, err := s.doRequest(ctx, "GET", "/", nil)
	return err
}

// ensureCollection 确保集合存在
func (s *Store) ensureCollection(ctx context.Context) error {
	// 检查集合是否存在
	_, err := s.doRequest(ctx, "GET", "/collections/"+s.config.Collection, nil)
	if err == nil {
		return nil // 集合已存在
	}

	// 创建集合
	createReq := map[string]any{
		"vectors": map[string]any{
			"size":     s.config.Dimension,
			"distance": string(s.config.Distance),
			"on_disk":  s.config.OnDisk,
		},
	}

	_, err = s.doRequest(ctx, "PUT", "/collections/"+s.config.Collection, createReq)
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	return nil
}

// Add 添加文档
func (s *Store) Add(ctx context.Context, docs []vector.Document) error {
	if len(docs) == 0 {
		return nil
	}

	points := make([]map[string]any, len(docs))
	for i, doc := range docs {
		payload := map[string]any{
			"content":     doc.Content,
			"created_at":  doc.CreatedAt.Format(time.RFC3339),
			"_original_id": doc.ID, // 存储原始 ID 以便还原
		}
		// 合并元数据
		for k, v := range doc.Metadata {
			payload[k] = v
		}

		points[i] = map[string]any{
			"id":      s.toPointID(doc.ID),
			"vector":  doc.Embedding,
			"payload": payload,
		}
	}

	req := map[string]any{
		"points": points,
	}

	_, err := s.doRequest(ctx, "PUT", "/collections/"+s.config.Collection+"/points", req)
	if err != nil {
		return fmt.Errorf("failed to add documents: %w", err)
	}

	return nil
}

// Search 搜索相似文档
func (s *Store) Search(ctx context.Context, query []float32, k int, opts ...vector.SearchOption) ([]vector.Document, error) {
	cfg := &vector.SearchConfig{
		IncludeMetadata: true,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	req := map[string]any{
		"vector":       query,
		"limit":        k,
		"with_payload": true,
		"with_vector":  cfg.IncludeEmbedding,
	}

	// 添加分数阈值
	if cfg.MinScore > 0 {
		req["score_threshold"] = cfg.MinScore
	}

	// 添加过滤条件
	if cfg.Filter != nil && len(cfg.Filter) > 0 {
		req["filter"] = s.buildFilter(cfg.Filter)
	}

	resp, err := s.doRequest(ctx, "POST", "/collections/"+s.config.Collection+"/points/search", req)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// 解析响应
	var searchResp searchResponse
	if err := json.Unmarshal(resp, &searchResp); err != nil {
		return nil, fmt.Errorf("failed to parse search response: %w", err)
	}

	docs := make([]vector.Document, len(searchResp.Result))
	for i, point := range searchResp.Result {
		doc := vector.Document{
			ID:    s.fromPointID(point.ID, point.Payload),
			Score: point.Score,
		}

		if point.Payload != nil {
			if content, ok := point.Payload["content"].(string); ok {
				doc.Content = content
			}
			if createdAt, ok := point.Payload["created_at"].(string); ok {
				if t, err := time.Parse(time.RFC3339, createdAt); err == nil {
					doc.CreatedAt = t
				}
			}

			// 提取元数据（排除内部字段）
			if cfg.IncludeMetadata {
				doc.Metadata = make(map[string]any)
				for k, v := range point.Payload {
					if k != "content" && k != "created_at" && k != "_original_id" {
						doc.Metadata[k] = v
					}
				}
			}
		}

		if cfg.IncludeEmbedding && point.Vector != nil {
			doc.Embedding = point.Vector
		}

		docs[i] = doc
	}

	return docs, nil
}

// Get 根据 ID 获取文档
func (s *Store) Get(ctx context.Context, id string) (*vector.Document, error) {
	req := map[string]any{
		"ids":          []any{s.toPointID(id)},
		"with_payload": true,
		"with_vector":  true,
	}

	resp, err := s.doRequest(ctx, "POST", "/collections/"+s.config.Collection+"/points", req)
	if err != nil {
		return nil, fmt.Errorf("get failed: %w", err)
	}

	var getResp getResponse
	if err := json.Unmarshal(resp, &getResp); err != nil {
		return nil, fmt.Errorf("failed to parse get response: %w", err)
	}

	if len(getResp.Result) == 0 {
		return nil, nil
	}

	point := getResp.Result[0]
	doc := &vector.Document{
		ID:        id,
		Embedding: point.Vector,
	}

	if point.Payload != nil {
		if content, ok := point.Payload["content"].(string); ok {
			doc.Content = content
		}
		if createdAt, ok := point.Payload["created_at"].(string); ok {
			if t, err := time.Parse(time.RFC3339, createdAt); err == nil {
				doc.CreatedAt = t
			}
		}

		doc.Metadata = make(map[string]any)
		for k, v := range point.Payload {
			if k != "content" && k != "created_at" && k != "_original_id" {
				doc.Metadata[k] = v
			}
		}
	}

	return doc, nil
}

// Delete 删除文档
func (s *Store) Delete(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	pointIDs := make([]any, len(ids))
	for i, id := range ids {
		pointIDs[i] = s.toPointID(id)
	}

	req := map[string]any{
		"points": pointIDs,
	}

	_, err := s.doRequest(ctx, "POST", "/collections/"+s.config.Collection+"/points/delete", req)
	if err != nil {
		return fmt.Errorf("delete failed: %w", err)
	}

	return nil
}

// Clear 清空存储
func (s *Store) Clear(ctx context.Context) error {
	// 删除并重新创建集合
	_, _ = s.doRequest(ctx, "DELETE", "/collections/"+s.config.Collection, nil)

	return s.ensureCollection(ctx)
}

// Count 返回文档数量
func (s *Store) Count(ctx context.Context) (int, error) {
	resp, err := s.doRequest(ctx, "GET", "/collections/"+s.config.Collection, nil)
	if err != nil {
		return 0, fmt.Errorf("count failed: %w", err)
	}

	var collResp collectionResponse
	if err := json.Unmarshal(resp, &collResp); err != nil {
		return 0, fmt.Errorf("failed to parse collection response: %w", err)
	}

	return collResp.Result.PointsCount, nil
}

// Close 关闭存储
func (s *Store) Close() error {
	s.client.CloseIdleConnections()
	return nil
}

// 确保实现了 Store 接口
var _ vector.Store = (*Store)(nil)

// ============== 内部方法 ==============

// doRequest 执行 HTTP 请求
func (s *Store) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if s.config.APIKey != "" {
		req.Header.Set("api-key", s.config.APIKey)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// toPointID 将字符串 ID 转换为 Qdrant 支持的格式
func (s *Store) toPointID(id string) any {
	// 使用 ID 的 hash 作为整数 ID
	var hash uint64
	for i := 0; i < len(id); i++ {
		hash = hash*31 + uint64(id[i])
	}
	return hash
}

// fromPointID 从 Qdrant point ID 还原字符串 ID
// 优先从 payload 中的 _original_id 字段获取原始 ID
func (s *Store) fromPointID(id any, payload map[string]any) string {
	// 优先使用 payload 中存储的原始 ID
	if payload != nil {
		if originalID, ok := payload["_original_id"].(string); ok && originalID != "" {
			return originalID
		}
	}
	// 回退到 point ID
	switch v := id.(type) {
	case float64:
		return fmt.Sprintf("%d", int64(v))
	case int64:
		return fmt.Sprintf("%d", v)
	case string:
		return v
	default:
		return fmt.Sprintf("%v", id)
	}
}

// buildFilter 构建过滤条件
func (s *Store) buildFilter(filter map[string]any) map[string]any {
	conditions := make([]map[string]any, 0, len(filter))

	for k, v := range filter {
		conditions = append(conditions, map[string]any{
			"key":   k,
			"match": map[string]any{"value": v},
		})
	}

	return map[string]any{
		"must": conditions,
	}
}

// ============== 响应类型 ==============

type searchResponse struct {
	Result []searchPoint `json:"result"`
}

type searchPoint struct {
	ID      any            `json:"id"`
	Score   float32        `json:"score"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector"`
}

type getResponse struct {
	Result []getPoint `json:"result"`
}

type getPoint struct {
	ID      any            `json:"id"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector"`
}

type collectionResponse struct {
	Result collectionInfo `json:"result"`
}

type collectionInfo struct {
	PointsCount int `json:"points_count"`
}
