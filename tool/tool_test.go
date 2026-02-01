package tool

import (
	"context"
	"errors"
	"testing"
)

// 测试输入类型
type CalculatorInput struct {
	A  float64 `json:"a" desc:"第一个数字" required:"true"`
	B  float64 `json:"b" desc:"第二个数字" required:"true"`
	Op string  `json:"op" desc:"运算符" required:"true"`
}

func TestNewFunc(t *testing.T) {
	tool := NewFunc("calculator", "执行计算",
		func(ctx context.Context, input CalculatorInput) (float64, error) {
			switch input.Op {
			case "add":
				return input.A + input.B, nil
			case "sub":
				return input.A - input.B, nil
			default:
				return 0, errors.New("unknown operator")
			}
		},
	)

	if tool.Name() != "calculator" {
		t.Errorf("Name() = %q, want %q", tool.Name(), "calculator")
	}
	if tool.Description() != "执行计算" {
		t.Errorf("Description() = %q, want %q", tool.Description(), "执行计算")
	}

	schema := tool.Schema()
	if schema == nil {
		t.Fatal("Schema() is nil")
	}
	if schema.Type != "object" {
		t.Errorf("Schema().Type = %q, want %q", schema.Type, "object")
	}
}

func TestFuncTool_Execute(t *testing.T) {
	tool := NewFunc("add", "加法",
		func(ctx context.Context, input CalculatorInput) (float64, error) {
			return input.A + input.B, nil
		},
	)

	ctx := context.Background()
	result, err := tool.Execute(ctx, map[string]any{
		"a":  10.0,
		"b":  20.0,
		"op": "add",
	})

	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if !result.Success {
		t.Fatalf("Execute failed: %s", result.Error)
	}
	if result.Output != 30.0 {
		t.Errorf("Execute result = %v, want 30.0", result.Output)
	}
}

func TestFuncTool_Execute_Error(t *testing.T) {
	tool := NewFunc("fail", "总是失败",
		func(ctx context.Context, input CalculatorInput) (float64, error) {
			return 0, errors.New("intentional error")
		},
	)

	ctx := context.Background()
	result, err := tool.Execute(ctx, map[string]any{
		"a":  1.0,
		"b":  2.0,
		"op": "fail",
	})

	if err != nil {
		t.Fatalf("Execute should not return error: %v", err)
	}
	if result.Success {
		t.Error("Execute should fail")
	}
	if result.Error != "intentional error" {
		t.Errorf("Error = %q, want %q", result.Error, "intentional error")
	}
}

func TestFuncTool_Validate(t *testing.T) {
	tool := NewFunc("test", "测试",
		func(ctx context.Context, input CalculatorInput) (float64, error) {
			return 0, nil
		},
	)

	// 缺少必填字段
	err := tool.Validate(map[string]any{"a": 1.0})
	if err == nil {
		t.Error("Validate should fail when missing required fields")
	}

	// 所有必填字段都有
	err = tool.Validate(map[string]any{"a": 1.0, "b": 2.0, "op": "add"})
	if err != nil {
		t.Errorf("Validate should pass: %v", err)
	}
}

func TestSimpleTool(t *testing.T) {
	tool := New("echo", "回显输入",
		func(ctx context.Context, args map[string]any) (any, error) {
			return args["message"], nil
		},
	)

	if tool.Name() != "echo" {
		t.Errorf("Name() = %q, want %q", tool.Name(), "echo")
	}

	ctx := context.Background()
	result, err := tool.Execute(ctx, map[string]any{"message": "hello"})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if result.Output != "hello" {
		t.Errorf("Output = %v, want %q", result.Output, "hello")
	}
}

func TestSimpleTool_WithValidator(t *testing.T) {
	tool := New("validate", "带验证",
		func(ctx context.Context, args map[string]any) (any, error) {
			return "ok", nil
		},
		WithValidator(func(args map[string]any) error {
			if _, ok := args["required"]; !ok {
				return errors.New("missing required field")
			}
			return nil
		}),
	)

	err := tool.Validate(map[string]any{})
	if err == nil {
		t.Error("Validate should fail")
	}

	err = tool.Validate(map[string]any{"required": true})
	if err != nil {
		t.Errorf("Validate should pass: %v", err)
	}
}

func TestRegistry(t *testing.T) {
	reg := NewRegistry()

	tool1 := New("tool1", "工具1", func(ctx context.Context, args map[string]any) (any, error) {
		return "tool1", nil
	})
	tool2 := New("tool2", "工具2", func(ctx context.Context, args map[string]any) (any, error) {
		return "tool2", nil
	})

	// 注册
	if err := reg.Register(tool1); err != nil {
		t.Fatalf("Register tool1 error: %v", err)
	}
	if err := reg.Register(tool2); err != nil {
		t.Fatalf("Register tool2 error: %v", err)
	}

	// 获取
	got, ok := reg.Get("tool1")
	if !ok {
		t.Error("Get(tool1) failed")
	}
	if got.Name() != "tool1" {
		t.Errorf("Get(tool1).Name() = %q, want %q", got.Name(), "tool1")
	}

	// 不存在的工具
	_, ok = reg.Get("nonexistent")
	if ok {
		t.Error("Get(nonexistent) should fail")
	}

	// 列表
	list := reg.List()
	if len(list) != 2 {
		t.Errorf("len(List()) = %d, want 2", len(list))
	}

	// 所有工具
	all := reg.All()
	if len(all) != 2 {
		t.Errorf("len(All()) = %d, want 2", len(all))
	}

	// 删除
	reg.Remove("tool1")
	_, ok = reg.Get("tool1")
	if ok {
		t.Error("tool1 should be removed")
	}

	// 清空
	reg.Clear()
	list = reg.List()
	if len(list) != 0 {
		t.Errorf("len(List()) after Clear = %d, want 0", len(list))
	}
}

func TestRegistry_RegisterAll(t *testing.T) {
	reg := NewRegistry()

	tool1 := New("t1", "工具1", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil })
	tool2 := New("t2", "工具2", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil })

	if err := reg.RegisterAll(tool1, tool2); err != nil {
		t.Fatalf("RegisterAll error: %v", err)
	}

	if len(reg.List()) != 2 {
		t.Errorf("len(List()) = %d, want 2", len(reg.List()))
	}
}

func TestRegistry_EmptyName(t *testing.T) {
	reg := NewRegistry()

	tool := New("", "空名称", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil })

	err := reg.Register(tool)
	if err == nil {
		t.Error("Register with empty name should fail")
	}
}

func TestRegistry_MustGet(t *testing.T) {
	reg := NewRegistry()

	tool := New("test", "测试", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil })
	reg.Register(tool)

	// 正常获取
	got := reg.MustGet("test")
	if got.Name() != "test" {
		t.Errorf("MustGet(test).Name() = %q, want %q", got.Name(), "test")
	}

	// panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("MustGet(nonexistent) should panic")
		}
	}()
	reg.MustGet("nonexistent")
}

func TestGlobalRegistry(t *testing.T) {
	// 清理可能存在的全局状态
	for _, name := range List() {
		globalRegistry.Remove(name)
	}

	tool := New("global", "全局工具", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil })

	if err := Register(tool); err != nil {
		t.Fatalf("Register error: %v", err)
	}

	got, ok := Get("global")
	if !ok {
		t.Error("Get(global) failed")
	}
	if got.Name() != "global" {
		t.Errorf("Get(global).Name() = %q, want %q", got.Name(), "global")
	}

	list := List()
	if len(list) == 0 {
		t.Error("List() is empty")
	}

	all := All()
	if len(all) == 0 {
		t.Error("All() is empty")
	}

	// 清理
	globalRegistry.Remove("global")
}

func TestToLLMFormat(t *testing.T) {
	tool := NewFunc("calc", "计算器",
		func(ctx context.Context, input CalculatorInput) (float64, error) {
			return 0, nil
		},
	)

	def := ToLLMFormat(tool)

	if def.Type != "function" {
		t.Errorf("Type = %q, want %q", def.Type, "function")
	}
	if def.Function.Name != "calc" {
		t.Errorf("Function.Name = %q, want %q", def.Function.Name, "calc")
	}
	if def.Function.Description != "计算器" {
		t.Errorf("Function.Description = %q, want %q", def.Function.Description, "计算器")
	}
	if def.Function.Parameters == nil {
		t.Error("Function.Parameters is nil")
	}
}

func TestToLLMFormatBatch(t *testing.T) {
	tools := []Tool{
		New("t1", "工具1", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil }),
		New("t2", "工具2", func(ctx context.Context, args map[string]any) (any, error) { return nil, nil }),
	}

	defs := ToLLMFormatBatch(tools)

	if len(defs) != 2 {
		t.Errorf("len(defs) = %d, want 2", len(defs))
	}
}

func TestParseArgs(t *testing.T) {
	// 正常 JSON
	args, err := ParseArgs(`{"a": 1, "b": "hello"}`)
	if err != nil {
		t.Fatalf("ParseArgs error: %v", err)
	}
	if args["a"] != 1.0 {
		t.Errorf("args[a] = %v, want 1", args["a"])
	}
	if args["b"] != "hello" {
		t.Errorf("args[b] = %v, want hello", args["b"])
	}

	// 空字符串
	args, err = ParseArgs("")
	if err != nil {
		t.Fatalf("ParseArgs('') error: %v", err)
	}
	if len(args) != 0 {
		t.Errorf("ParseArgs('') should return empty map")
	}

	// 无效 JSON
	_, err = ParseArgs("invalid")
	if err == nil {
		t.Error("ParseArgs(invalid) should fail")
	}
}

func TestStructToArgs(t *testing.T) {
	input := CalculatorInput{A: 10, B: 20, Op: "add"}
	args, err := StructToArgs(input)

	if err != nil {
		t.Fatalf("StructToArgs error: %v", err)
	}
	if args["a"] != 10.0 {
		t.Errorf("args[a] = %v, want 10", args["a"])
	}
	if args["b"] != 20.0 {
		t.Errorf("args[b] = %v, want 20", args["b"])
	}
	if args["op"] != "add" {
		t.Errorf("args[op] = %v, want add", args["op"])
	}
}

func TestArgsToStruct(t *testing.T) {
	args := map[string]any{"a": 10.0, "b": 20.0, "op": "add"}
	input, err := ArgsToStruct[CalculatorInput](args)

	if err != nil {
		t.Fatalf("ArgsToStruct error: %v", err)
	}
	if input.A != 10 {
		t.Errorf("input.A = %v, want 10", input.A)
	}
	if input.B != 20 {
		t.Errorf("input.B = %v, want 20", input.B)
	}
	if input.Op != "add" {
		t.Errorf("input.Op = %v, want add", input.Op)
	}
}

func TestResult(t *testing.T) {
	// 成功结果
	success := NewResult("hello")
	if !success.Success {
		t.Error("NewResult should be successful")
	}
	if success.Output != "hello" {
		t.Errorf("Output = %v, want hello", success.Output)
	}

	// 错误结果
	err := NewErrorResult(errors.New("failed"))
	if err.Success {
		t.Error("NewErrorResult should not be successful")
	}
	if err.Error != "failed" {
		t.Errorf("Error = %q, want %q", err.Error, "failed")
	}
}

func TestResult_String(t *testing.T) {
	success := NewResult(map[string]int{"count": 42})
	str := success.String()
	if str != `{"count":42}` {
		t.Errorf("String() = %q, want %q", str, `{"count":42}`)
	}

	fail := NewErrorResult(errors.New("oops"))
	str = fail.String()
	if str != "Error: oops" {
		t.Errorf("String() = %q, want %q", str, "Error: oops")
	}
}

func TestFuncTool_Context(t *testing.T) {
	tool := NewFunc("ctx", "上下文测试",
		func(ctx context.Context, input CalculatorInput) (string, error) {
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			default:
				return "ok", nil
			}
		},
	)

	// 正常上下文
	ctx := context.Background()
	result, _ := tool.Execute(ctx, map[string]any{"a": 1.0, "b": 2.0, "op": "add"})
	if !result.Success {
		t.Error("Execute with normal context should succeed")
	}

	// 已取消的上下文
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, _ = tool.Execute(ctx, map[string]any{"a": 1.0, "b": 2.0, "op": "add"})
	if result.Success {
		t.Error("Execute with cancelled context should fail")
	}
}
