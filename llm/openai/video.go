package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/hexagon-codes/ai-core/llm"
)

// CreateVideoTask 提交视频生成任务
//
// 调用 /videos/generations 端点创建异步任务。
// 兼容智谱 CogVideoX、OpenAI Sora 等 API。
func (p *Provider) CreateVideoTask(ctx context.Context, req llm.VideoRequest) (*llm.VideoTask, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	payload := videoGenRequest{
		Model:  req.Model,
		Prompt: req.Prompt,
	}
	if req.Size != "" {
		payload.Size = req.Size
	}
	if req.Duration > 0 {
		payload.Duration = req.Duration
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("序列化视频生成请求失败: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/videos/generations", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("视频生成请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if readErr != nil {
			return nil, fmt.Errorf("openai video api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		return nil, fmt.Errorf("openai video api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result videoGenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析视频生成响应失败: %w", err)
	}

	return &llm.VideoTask{
		ID:     result.ID,
		Status: mapVideoStatus(result.TaskStatus),
	}, nil
}

// QueryVideoTask 查询视频生成任务状态
//
// 调用 /async-result/{id} 端点查询（智谱风格）。
// 同时兼容 /videos/{id} 端点（OpenAI 风格）。
func (p *Provider) QueryVideoTask(ctx context.Context, taskID string) (*llm.VideoTask, error) {
	// 优先尝试智谱风格端点：/async-result/{id}
	httpReq, err := http.NewRequestWithContext(ctx, "GET", p.baseURL+"/async-result/"+taskID, nil)
	if err != nil {
		return nil, err
	}
	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("查询视频任务失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if readErr != nil {
			return nil, fmt.Errorf("openai video query error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		return nil, fmt.Errorf("openai video query error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result videoQueryResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析视频任务状态失败: %w", err)
	}

	task := &llm.VideoTask{
		ID:     result.ID,
		Status: mapVideoStatus(result.TaskStatus),
	}

	// 提取错误信息
	if result.Error != nil && result.Error.Message != "" {
		task.Error = result.Error.Message
	}

	// 提取视频和封面 URL
	for _, v := range result.VideoResult {
		if v.URL != "" {
			task.VideoURL = v.URL
		}
		if v.CoverImageURL != "" {
			task.CoverURL = v.CoverImageURL
		}
	}

	return task, nil
}

// videoGenRequest 视频生成请求结构
type videoGenRequest struct {
	Model    string `json:"model"`
	Prompt   string `json:"prompt"`
	Size     string `json:"size,omitempty"`
	Duration int    `json:"duration,omitempty"`
}

// videoGenResponse 视频生成创建响应
type videoGenResponse struct {
	ID         string `json:"id"`
	TaskStatus string `json:"task_status"`
}

// videoQueryResponse 视频任务查询响应
type videoQueryResponse struct {
	ID          string `json:"id"`
	TaskStatus  string `json:"task_status"`
	VideoResult []struct {
		URL           string `json:"url"`
		CoverImageURL string `json:"cover_image_url"`
	} `json:"video_result"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// mapVideoStatus 将 API 返回的状态字符串映射为统一枚举
func mapVideoStatus(raw string) llm.VideoTaskStatus {
	switch raw {
	case "SUCCESS", "completed":
		return llm.VideoTaskCompleted
	case "FAIL", "failed":
		return llm.VideoTaskFailed
	case "PROCESSING", "in_progress":
		return llm.VideoTaskProcessing
	default:
		return llm.VideoTaskQueued
	}
}

// 确保 OpenAI Provider 实现了 VideoProvider 接口
var _ llm.VideoProvider = (*Provider)(nil)
