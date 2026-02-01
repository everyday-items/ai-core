// Package memory 提供 AI Agent 记忆系统的核心接口和实现
//
// 记忆系统让 Agent 能够存储和检索历史信息，实现上下文感知和长期记忆。
// 本包提供了统一的 Memory 接口和多种实现。
//
// # 主要功能
//
//   - Memory 接口: 定义记忆系统的核心方法
//   - Buffer 记忆: 简单的内存缓冲区，适合短期对话
//   - 向量记忆: 基于向量相似度的语义检索（需要向量存储）
//
// # 基本用法
//
// 使用缓冲记忆：
//
//	mem := memory.NewBuffer(10) // 保留最近 10 条消息
//
//	// 保存消息
//	mem.Save(ctx, memory.Entry{
//	    Role:    "user",
//	    Content: "你好",
//	})
//
//	// 检索最近的消息
//	entries, err := mem.Search(ctx, memory.SearchQuery{Limit: 5})
//
// # 记忆类型
//
//   - BufferMemory: 简单的 FIFO 缓冲区
//   - SummaryMemory: 自动摘要的记忆（需要 LLM）
//   - VectorMemory: 向量检索记忆（需要向量存储）
//   - EntityMemory: 实体提取记忆
//   - GraphMemory: 知识图谱记忆
package memory
