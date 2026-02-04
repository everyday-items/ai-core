package schema

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
)

// Schema 表示 JSON Schema 定义
// 用于描述 LLM 工具参数的结构和约束
type Schema struct {
	Type                 string             `json:"type"`
	Title                string             `json:"title,omitempty"`
	Description          string             `json:"description,omitempty"`
	Properties           map[string]*Schema `json:"properties,omitempty"`
	Required             []string           `json:"required,omitempty"`
	Items                *Schema            `json:"items,omitempty"`
	AdditionalProperties *Schema            `json:"additionalProperties,omitempty"`
	Enum                 []any              `json:"enum,omitempty"`
	Default              any                `json:"default,omitempty"`
	Minimum              *float64           `json:"minimum,omitempty"`
	Maximum              *float64           `json:"maximum,omitempty"`
	MinLength            *int               `json:"minLength,omitempty"`
	MaxLength            *int               `json:"maxLength,omitempty"`
	Pattern              string             `json:"pattern,omitempty"`
	Format               string             `json:"format,omitempty"`
}

// String 返回 Schema 的 JSON 字符串表示
func (s *Schema) String() string {
	b, _ := json.Marshal(s)
	return string(b)
}

// MarshalJSON 实现 json.Marshaler 接口
func (s *Schema) MarshalJSON() ([]byte, error) {
	type Alias Schema
	return json.Marshal((*Alias)(s))
}

// Of 从 Go 类型生成 Schema
// 支持的 struct tag：
//   - json: 字段名（同 encoding/json）
//   - desc: 字段描述
//   - required: 是否必填 ("true" 或 "false")
//   - enum: 枚举值，逗号分隔
//   - default: 默认值
//   - min: 最小值（数字）或最小长度（字符串）
//   - max: 最大值（数字）或最大长度（字符串）
//   - pattern: 正则表达式（字符串）
//   - format: 格式（如 "email"、"uri"、"date-time"）
//
// 示例：
//
//	type Input struct {
//	    Name  string  `json:"name" desc:"用户名" required:"true" min:"1" max:"100"`
//	    Age   int     `json:"age" desc:"年龄" min:"0" max:"150"`
//	    Email string  `json:"email" format:"email"`
//	}
//	schema := schema.Of[Input]()
func Of[T any]() *Schema {
	var zero T
	return fromType(reflect.TypeOf(zero))
}

// FromType 从 reflect.Type 生成 Schema
func FromType(t reflect.Type) *Schema {
	return fromType(t)
}

// FromValue 从任意值生成 Schema
func FromValue(v any) *Schema {
	if v == nil {
		return &Schema{Type: "null"}
	}
	return fromType(reflect.TypeOf(v))
}

func fromType(t reflect.Type) *Schema {
	if t == nil {
		return &Schema{Type: "null"}
	}

	// 处理指针
	if t.Kind() == reflect.Pointer {
		return fromType(t.Elem())
	}

	switch t.Kind() {
	case reflect.String:
		return &Schema{Type: "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &Schema{Type: "integer"}
	case reflect.Float32, reflect.Float64:
		return &Schema{Type: "number"}
	case reflect.Bool:
		return &Schema{Type: "boolean"}
	case reflect.Slice, reflect.Array:
		return &Schema{
			Type:  "array",
			Items: fromType(t.Elem()),
		}
	case reflect.Map:
		return &Schema{Type: "object"}
	case reflect.Struct:
		return fromStruct(t)
	case reflect.Interface:
		return &Schema{Type: "object"}
	default:
		return &Schema{Type: "object"}
	}
}

func fromStruct(t reflect.Type) *Schema {
	schema := &Schema{
		Type:       "object",
		Properties: make(map[string]*Schema),
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		// 获取 JSON tag 或使用字段名
		name := field.Name
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}
		if jsonTag != "" {
			// 解析 json tag（处理 omitempty 等）
			if idx := strings.Index(jsonTag, ","); idx != -1 {
				jsonTag = jsonTag[:idx]
			}
			if jsonTag != "" {
				name = jsonTag
			}
		}

		propSchema := fromType(field.Type)

		// 解析 desc tag
		if desc := field.Tag.Get("desc"); desc != "" {
			propSchema.Description = desc
		}

		// 解析 required tag
		if field.Tag.Get("required") == "true" {
			schema.Required = append(schema.Required, name)
		}

		// 解析 enum tag
		if enumStr := field.Tag.Get("enum"); enumStr != "" {
			parts := strings.Split(enumStr, ",")
			enums := make([]any, len(parts))
			for j, p := range parts {
				enums[j] = strings.TrimSpace(p)
			}
			propSchema.Enum = enums
		}

		// 解析 default tag
		if defStr := field.Tag.Get("default"); defStr != "" {
			propSchema.Default = defStr
		}

		// 解析 format tag
		if format := field.Tag.Get("format"); format != "" {
			propSchema.Format = format
		}

		// 解析 pattern tag
		if pattern := field.Tag.Get("pattern"); pattern != "" {
			propSchema.Pattern = pattern
		}

		// 解析 min/max tag
		if minStr := field.Tag.Get("min"); minStr != "" {
			if propSchema.Type == "string" {
				// 字符串最小长度
				minLen := parseInt(minStr)
				propSchema.MinLength = &minLen
			} else {
				// 数字最小值
				minVal := parseFloat(minStr)
				propSchema.Minimum = &minVal
			}
		}
		if maxStr := field.Tag.Get("max"); maxStr != "" {
			if propSchema.Type == "string" {
				// 字符串最大长度
				maxLen := parseInt(maxStr)
				propSchema.MaxLength = &maxLen
			} else {
				// 数字最大值
				maxVal := parseFloat(maxStr)
				propSchema.Maximum = &maxVal
			}
		}

		schema.Properties[name] = propSchema
	}

	return schema
}

func parseInt(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		// 尝试 JSON 解析作为回退（支持带引号的数字）
		var jv int
		if json.Unmarshal([]byte(s), &jv) == nil {
			return jv
		}
		return 0 // 解析失败返回 0
	}
	return v
}

func parseFloat(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		// 尝试 JSON 解析作为回退（支持带引号的数字）
		var jv float64
		if json.Unmarshal([]byte(s), &jv) == nil {
			return jv
		}
		return 0 // 解析失败返回 0
	}
	return v
}

// ============== 构建器模式 ==============

// Builder 提供链式构建 Schema 的能力
type Builder struct {
	schema *Schema
}

// NewBuilder 创建新的 Schema 构建器
func NewBuilder() *Builder {
	return &Builder{
		schema: &Schema{
			Properties: make(map[string]*Schema),
		},
	}
}

// Type 设置 Schema 类型
func (b *Builder) Type(t string) *Builder {
	b.schema.Type = t
	return b
}

// Title 设置标题
func (b *Builder) Title(title string) *Builder {
	b.schema.Title = title
	return b
}

// Description 设置描述
func (b *Builder) Description(desc string) *Builder {
	b.schema.Description = desc
	return b
}

// Property 添加一个属性
func (b *Builder) Property(name string, prop *Schema, required bool) *Builder {
	b.schema.Properties[name] = prop
	if required {
		b.schema.Required = append(b.schema.Required, name)
	}
	return b
}

// Items 设置数组元素的 Schema
func (b *Builder) Items(items *Schema) *Builder {
	b.schema.Items = items
	return b
}

// Enum 设置枚举值
func (b *Builder) Enum(values ...any) *Builder {
	b.schema.Enum = values
	return b
}

// Default 设置默认值
func (b *Builder) Default(v any) *Builder {
	b.schema.Default = v
	return b
}

// Min 设置最小值
func (b *Builder) Min(v float64) *Builder {
	b.schema.Minimum = &v
	return b
}

// Max 设置最大值
func (b *Builder) Max(v float64) *Builder {
	b.schema.Maximum = &v
	return b
}

// MinLength 设置最小长度
func (b *Builder) MinLength(v int) *Builder {
	b.schema.MinLength = &v
	return b
}

// MaxLength 设置最大长度
func (b *Builder) MaxLength(v int) *Builder {
	b.schema.MaxLength = &v
	return b
}

// Pattern 设置正则模式
func (b *Builder) Pattern(p string) *Builder {
	b.schema.Pattern = p
	return b
}

// Format 设置格式
func (b *Builder) Format(f string) *Builder {
	b.schema.Format = f
	return b
}

// Build 返回构建的 Schema
func (b *Builder) Build() *Schema {
	return b.schema
}

// ============== 便捷构造函数 ==============

// String 创建字符串类型的 Schema
func String(desc string) *Schema {
	return &Schema{Type: "string", Description: desc}
}

// Integer 创建整数类型的 Schema
func Integer(desc string) *Schema {
	return &Schema{Type: "integer", Description: desc}
}

// Number 创建数字类型的 Schema
func Number(desc string) *Schema {
	return &Schema{Type: "number", Description: desc}
}

// Boolean 创建布尔类型的 Schema
func Boolean(desc string) *Schema {
	return &Schema{Type: "boolean", Description: desc}
}

// Array 创建数组类型的 Schema
func Array(desc string, items *Schema) *Schema {
	return &Schema{Type: "array", Description: desc, Items: items}
}

// Object 创建对象类型的 Schema
func Object(desc string) *Schema {
	return &Schema{Type: "object", Description: desc, Properties: make(map[string]*Schema)}
}

// StringEnum 创建字符串枚举类型的 Schema
func StringEnum(desc string, values ...string) *Schema {
	enums := make([]any, len(values))
	for i, v := range values {
		enums[i] = v
	}
	return &Schema{Type: "string", Description: desc, Enum: enums}
}
