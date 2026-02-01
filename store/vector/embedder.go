package vector

import "context"

// Embedder 向量生成器接口
//
// Embedder 负责将文本转换为向量嵌入。
// 典型的实现包括 OpenAI Embeddings、Sentence Transformers 等。
type Embedder interface {
	// Embed 将多个文本转换为向量
	//
	// 参数:
	//   - ctx: 上下文
	//   - texts: 要转换的文本列表
	//
	// 返回:
	//   - [][]float32: 向量列表，与输入文本一一对应
	//   - error: 转换失败时返回错误
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// EmbedOne 将单个文本转换为向量
	//
	// 参数:
	//   - ctx: 上下文
	//   - text: 要转换的文本
	//
	// 返回:
	//   - []float32: 向量
	//   - error: 转换失败时返回错误
	EmbedOne(ctx context.Context, text string) ([]float32, error)

	// Dimension 返回向量维度
	Dimension() int
}

// EmbedderFunc 函数式 Embedder
//
// EmbedderFunc 允许使用函数来创建 Embedder，
// 适用于简单的自定义实现场景。
type EmbedderFunc struct {
	embedFn   func(ctx context.Context, texts []string) ([][]float32, error)
	dimension int
}

// NewEmbedderFunc 创建函数式 Embedder
//
// 参数:
//   - dimension: 向量维度
//   - fn: 嵌入函数
//
// 返回:
//   - *EmbedderFunc: 函数式 Embedder 实例
func NewEmbedderFunc(dimension int, fn func(ctx context.Context, texts []string) ([][]float32, error)) *EmbedderFunc {
	return &EmbedderFunc{
		embedFn:   fn,
		dimension: dimension,
	}
}

// Embed 将多个文本转换为向量
func (e *EmbedderFunc) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return e.embedFn(ctx, texts)
}

// EmbedOne 将单个文本转换为向量
func (e *EmbedderFunc) EmbedOne(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := e.embedFn(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, nil
	}
	return embeddings[0], nil
}

// Dimension 返回向量维度
func (e *EmbedderFunc) Dimension() int {
	return e.dimension
}

var _ Embedder = (*EmbedderFunc)(nil)
