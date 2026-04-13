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

// GenerateImage 调用 OpenAI 兼容的 /images/generations 端点生成图片
//
// 支持 OpenAI DALL-E、智谱 CogView 等兼容此协议的模型。
// 图片生成为非流式请求，超时由调用方 context 控制。
func (p *Provider) GenerateImage(ctx context.Context, req llm.ImageRequest) (*llm.ImageResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	payload := imageGenRequest{
		Model:  req.Model,
		Prompt: req.Prompt,
	}
	if req.Size != "" {
		payload.Size = req.Size
	}
	if req.N > 0 {
		payload.N = req.N
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("序列化图片生成请求失败: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/images/generations", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("图片生成请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) // 限制 1MB
		if readErr != nil {
			return nil, fmt.Errorf("openai image api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		return nil, fmt.Errorf("openai image api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result imageGenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析图片生成响应失败: %w", err)
	}

	images := make([]llm.ImageData, len(result.Data))
	for i, img := range result.Data {
		images[i] = llm.ImageData{
			URL:           img.URL,
			RevisedPrompt: img.RevisedPrompt,
		}
	}

	return &llm.ImageResponse{Data: images}, nil
}

// imageGenRequest OpenAI Images API 请求结构
type imageGenRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Size   string `json:"size,omitempty"`
	N      int    `json:"n,omitempty"`
}

// imageGenResponse OpenAI Images API 响应结构
type imageGenResponse struct {
	Data []struct {
		URL           string `json:"url"`
		RevisedPrompt string `json:"revised_prompt,omitempty"`
	} `json:"data"`
}

// 确保 OpenAI Provider 实现了 ImageProvider 接口
var _ llm.ImageProvider = (*Provider)(nil)
