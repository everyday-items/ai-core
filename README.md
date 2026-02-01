# ai-core

Go 语言的 AI 基础能力库，为 [Hexagon](https://github.com/everyday-items/hexagon) AI Agent 框架提供核心支持。

## 特性

- **统一的 LLM 接口** - 一套代码，多家 Provider（OpenAI、Anthropic、DeepSeek、Gemini、通义千问、豆包、Ollama）
- **流式响应** - 统一的 SSE 流式处理，支持回调和 channel 两种模式
- **工具系统** - 类型安全的工具定义，从 Go 结构体自动生成 JSON Schema
- **记忆系统** - 多种记忆策略（缓冲、摘要、向量检索、多层组合）
- **智能路由** - 多 Provider 路由器，支持轮询、加权、最低延迟、降级等策略
- **用量追踪** - Token 消耗统计和成本估算

## 安装

```bash
go get github.com/everyday-items/ai-core
```

## 快速开始

### 基本对话

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/everyday-items/ai-core/llm"
    "github.com/everyday-items/ai-core/llm/openai"
)

func main() {
    provider := openai.New(os.Getenv("OPENAI_API_KEY"))

    resp, err := provider.Complete(context.Background(), llm.CompletionRequest{
        Model: "gpt-4o",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: "你好！"},
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Content)
}
```

### 流式响应

```go
stream, err := provider.Stream(ctx, llm.CompletionRequest{
    Model: "gpt-4o",
    Messages: llm.NewMessages("你是一个助手", "讲个笑话"),
})
if err != nil {
    panic(err)
}
defer stream.Close()

for chunk := range stream.Chunks() {
    fmt.Print(chunk.Content)
}
```

### 工具调用

```go
import "github.com/everyday-items/ai-core/tool"

type WeatherInput struct {
    City string `json:"city" desc:"城市名称" required:"true"`
}

weatherTool := tool.NewFunc("get_weather", "获取城市天气",
    func(ctx context.Context, input WeatherInput) (string, error) {
        return fmt.Sprintf("%s: 晴，25°C", input.City), nil
    },
)

// 转换为 LLM 工具定义
toolDef := llm.NewToolDefinition(
    weatherTool.Name(),
    weatherTool.Description(),
    weatherTool.Schema(),
)

resp, _ := provider.Complete(ctx, llm.CompletionRequest{
    Model:    "gpt-4o",
    Messages: []llm.Message{{Role: llm.RoleUser, Content: "北京天气怎么样？"}},
    Tools:    []llm.ToolDefinition{toolDef},
})

if resp.HasToolCalls() {
    for _, tc := range resp.ToolCalls {
        args, _ := tool.ParseArgs(tc.Arguments)
        result, _ := weatherTool.Execute(ctx, args)
        fmt.Println(result)
    }
}
```

### 多 Provider 路由

```go
import (
    "github.com/everyday-items/ai-core/llm/router"
    "github.com/everyday-items/ai-core/llm/openai"
    "github.com/everyday-items/ai-core/llm/deepseek"
)

r := router.NewBuilder().
    Add("openai", openai.New(openaiKey)).
    Add("deepseek", deepseek.New(deepseekKey)).
    Strategy(router.StrategyLeastLatency).
    Fallback("deepseek").
    Build()

// 使用路由器（自动选择最优 Provider）
resp, _ := r.Complete(ctx, req)
```

### 记忆系统

```go
import "github.com/everyday-items/ai-core/memory"

mem := memory.NewBuffer(100) // 保留最近 100 条消息

mem.Save(ctx, memory.NewUserEntry("你好"))
mem.Save(ctx, memory.NewAssistantEntry("你好！有什么可以帮助你的？"))

// 检索最近的消息
entries, _ := mem.Search(ctx, memory.SearchQuery{Limit: 10})
```

## 包结构

| 包 | 说明 |
|---|------|
| `llm` | LLM Provider 抽象接口 |
| `llm/openai` | OpenAI 实现（GPT-4o、o1、o3-mini 等） |
| `llm/anthropic` | Anthropic Claude 实现 |
| `llm/deepseek` | DeepSeek 实现 |
| `llm/gemini` | Google Gemini 实现 |
| `llm/qwen` | 通义千问实现 |
| `llm/ark` | 豆包（字节跳动）实现 |
| `llm/ollama` | Ollama 本地模型实现 |
| `llm/router` | 多 Provider 智能路由 |
| `memory` | Agent 记忆系统 |
| `tool` | 工具定义和注册 |
| `schema` | JSON Schema 生成 |
| `streamx` | 流式响应统一抽象 |
| `template` | Prompt 模板引擎 |
| `tokenizer` | Token 计数估算 |
| `meter` | 用量统计和成本追踪 |

## 支持的 LLM Provider

| Provider | 模型示例 | 特性 |
|----------|---------|------|
| OpenAI | gpt-4o, gpt-4o-mini, o1, o3-mini | 流式、函数调用、视觉 |
| Anthropic | claude-3-opus, claude-3-sonnet | 流式、函数调用 |
| DeepSeek | deepseek-chat, deepseek-coder | 流式、函数调用 |
| Gemini | gemini-pro, gemini-1.5-pro | 流式、函数调用 |
| 通义千问 | qwen-turbo, qwen-plus, qwen-max | 流式、函数调用 |
| 豆包 | doubao-pro-* | 流式、函数调用 |
| Ollama | llama3, mistral, codellama | 本地部署 |

## 路由策略

| 策略 | 说明 |
|------|------|
| `StrategyRoundRobin` | 轮询，均匀分发请求 |
| `StrategyRandom` | 随机选择 |
| `StrategyLeastLatency` | 选择延迟最低的 Provider |
| `StrategyLeastCost` | 选择成本最低的 Provider |
| `StrategyWeighted` | 按权重分发 |
| `StrategyFallback` | 按顺序尝试，失败后降级 |
| `StrategyModelMatch` | 根据请求的模型自动匹配 Provider |

## 许可证

MIT License
