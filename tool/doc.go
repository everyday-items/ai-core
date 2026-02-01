// Package tool 提供 AI Agent 工具系统的核心接口和实现
//
// 工具（Tool）是 Agent 与外部世界交互的能力，如搜索、计算、API 调用等。
// 本包提供了统一的 Tool 接口和便捷的工具创建方式。
//
// # 主要功能
//
//   - Tool 接口: 定义工具的核心方法
//   - 函数式工具: 从普通函数快速创建工具
//   - 工具注册中心: 管理和查找工具
//
// # 基本用法
//
// 使用函数创建工具：
//
//	type CalcInput struct {
//	    A  float64 `json:"a" desc:"第一个数字" required:"true"`
//	    B  float64 `json:"b" desc:"第二个数字" required:"true"`
//	    Op string  `json:"op" desc:"运算符" enum:"add,sub,mul,div" required:"true"`
//	}
//
//	calculatorTool := tool.NewFunc("calculator", "执行数学计算",
//	    func(ctx context.Context, input CalcInput) (float64, error) {
//	        switch input.Op {
//	        case "add":
//	            return input.A + input.B, nil
//	        case "sub":
//	            return input.A - input.B, nil
//	        // ...
//	        }
//	    },
//	)
//
// 注册和使用工具：
//
//	registry := tool.NewRegistry()
//	registry.Register(calculatorTool)
//
//	t, err := registry.Get("calculator")
//	result, err := t.Execute(ctx, map[string]any{"a": 1, "b": 2, "op": "add"})
package tool
