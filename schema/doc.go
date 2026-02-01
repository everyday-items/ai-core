// Package schema 提供 JSON Schema 生成和验证功能
//
// 主要用于 AI Agent 框架中的工具参数定义和验证。
// 支持从 Go 结构体自动生成 JSON Schema，以便与 LLM 的 Function Calling 功能配合使用。
//
// # 主要功能
//
//   - 从 Go 类型自动生成 JSON Schema
//   - 使用构建器模式手动构建 Schema
//   - 支持常见的 Schema 约束（required、minimum、maximum 等）
//
// # 基本用法
//
// 从结构体生成 Schema：
//
//	type CalculatorInput struct {
//	    A float64 `json:"a" desc:"第一个数字" required:"true"`
//	    B float64 `json:"b" desc:"第二个数字" required:"true"`
//	    Op string `json:"op" desc:"运算符: add, sub, mul, div" required:"true"`
//	}
//
//	schema := schema.Of[CalculatorInput]()
//
// 使用构建器：
//
//	schema := schema.NewBuilder().
//	    Type("object").
//	    Property("name", schema.String("用户名称"), true).
//	    Property("age", schema.Integer("用户年龄"), false).
//	    Build()
package schema
