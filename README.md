# SkillFlow

基于 [deepagents](https://github.com/langchain-ai/deepagents) 的 AI Skill 创建、评估与进化框架。

## 概述

SkillFlow 提供了一套完整的工作流，用于自动化地创建、测试和优化 AI Skill：

1. **Create** - 从 YAML 规格定义自动生成 Skill
2. **Eval** - 使用 LLM-as-Judge 自动生成测试用例并进行多轮评估
3. **Evolve** - 通过审计-反思-重写的迭代循环持续优化 Skill

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd skillflow

# 创建虚拟环境并安装
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

配置 `.env` 文件：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=gpt-4o
```

## 使用方法

### 创建 Skill

通过 YAML 规格文件生成 Skill：

```bash
skillflow create --spec specs/example_spec.yaml --output skills/my-skill --lang zh
```

### 评估 Skill

自动生成测试用例并运行评估：

```bash
skillflow eval --skill skills/my-skill --trials 3
```

参数说明：
- `--skill`: Skill 目录路径（需包含 SKILL.md）
- `--trials`: 每个测试用例的运行次数，用于计算 pass@k 稳定性
- `--output`: 评估结果输出路径（默认 `results/`）

### 进化 Skill

多轮迭代优化 Skill：

```bash
skillflow evolve --skill skills/my-skill --iterations 5
```

参数说明：
- `--skill`: Skill 目录路径
- `--iterations`: 进化迭代轮数
- `--output`: 结果输出路径（默认 `results/`）

## Skill 规格定义

使用 YAML 定义 Skill 规格，参考 `specs/example_spec.yaml`：

```yaml
description: "Skill 的简要描述"

scene:
  - "使用场景 1"
  - "使用场景 2"

input_output:
  input: "输入描述"
  output: "输出描述"

rule:
  - "规则 1"
  - "规则 2"

example:
  - input: |
      示例输入
    output: |
      示例输出

reference: []
```

## 项目结构

```
skillflow/
├── core/                  # 核心模块
│   ├── config.py          # 配置管理（环境变量加载）
│   ├── llm.py             # LLM 客户端（含限流）
│   ├── agent.py           # Agent 构建与执行
│   ├── prompts.py         # Prompt 模板
│   ├── utils.py           # 工具函数（文件 I/O、哈希等）
│   └── debug_middleware.py # 调试中间件（执行追踪）
├── create/                # Skill 创建模块
│   ├── spec_parser.py     # YAML 规格解析
│   ├── creator.py         # Skill 生成逻辑
│   └── cli.py             # CLI 命令
├── eval/                  # 评估模块
│   ├── test_generator.py  # 测试用例自动生成
│   ├── runner.py          # 评估执行器（多 trial 并行）
│   ├── metrics.py         # 指标计算（pass@k、reward）
│   └── cli.py             # CLI 命令
├── evolve/                # 进化模块
│   ├── mutator.py         # Skill 变异（审计-反思-重写）
│   ├── orchestrator.py    # 多轮进化编排（含早停）
│   └── cli.py             # CLI 命令
└── __main__.py            # 入口点
```

## 技术栈

- **Python 3.12+**
- **deepagents** - Agent 框架
- **langchain** - LLM 调用与编排
- **rich** - 终端美化输出
- **numpy** - 评估指标计算

## License

Private
