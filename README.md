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

### Web UI

启动 Web 服务器：

```bash
skillflow serve [--host 127.0.0.1] [--port 8765]
```

Web UI 包含以下页面：

| 页面 | 功能 |
|------|------|
| **Manage** | 上传技能（ZIP 包）、查看技能元信息、删除技能、下载技能 |
| **Create** | 从 YAML Spec 生成新技能 |
| **Eval** | 选择技能并运行评估，支持测试用例生成/保存 |
| **Evolve** | 运行多轮演化优化技能 |
| **指南** | 使用说明和参数文档 |

Manage 页面 API：

| API | 功能 |
|-----|------|
| `POST /api/upload-skill` | 上传 ZIP 格式技能包 |
| `GET /api/skills/{name}/meta` | 获取技能元信息（name、description） |
| `DELETE /api/skills/{name}` | 删除技能目录 |
| `GET /api/skills/{name}/download` | 下载技能 ZIP 包 |

### CLI 命令

#### 创建 Skill

通过 YAML 规格文件生成 Skill：

```bash
skillflow create --spec specs/example_spec.yaml --output skills/my-skill --lang zh
```

### 评估 Skill

自动生成测试用例并运行评估：

```bash
skillflow eval skills/my-skill --trials 3
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `skill` | Skill 目录路径（需包含 SKILL.md） | 必填 |
| `--spec` | SPEC 文件路径（用于生成测试用例） | 无 |
| `--trials` | 每个测试用例的运行次数 | 5 |
| `-j, --parallel` | 并行评估线程数 | 1（串行） |
| `--save-trace` | 将执行轨迹落盘到 `trace/` 目录 | 关闭 |
| `--test-cases` | 指定测试用例 JSON 文件，跳过生成 | 无 |
| `--init` | 只生成测试用例，不运行评估 | 关闭 |
| `--debug` | 启用 debug 中间件，输出详细执行日志 | 关闭 |
| `--ignore-cache` | 忽略缓存，重新生成测试用例 | 关闭 |
| `-o, --output` | 结果输出目录 | `results/<skill名>` |

示例：

```bash
# 使用已有测试用例，5 并发评估，保存轨迹
skillflow eval skills/my-skill \
  --test-cases results/my-skill/test_cases_202604081430.json \
  -j 5 --save-trace

# 只生成测试用例
skillflow eval skills/my-skill --init
```

### 进化 Skill

多轮迭代优化 Skill：

```bash
skillflow evolve skills/my-skill --test-cases results/my-skill/test_cases.json -j 5 -i 20 -p 10
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `skill` | Skill 目录路径 | 必填 |
| `--spec` | SPEC 文件路径 | 无 |
| `--trials` | 每个测试用例的运行次数 | 5 |
| `-j, --parallel` | 并行评估线程数 | 1 |
| `-i, --iterations` | 最大演化迭代轮数 | 100 |
| `-p, --patience` | 连续无提升早停轮数 | 10 |
| `-M, --mode` | 演化模式：`steady`（每轮从 baseline 出发）或 `greedy`（从当前最优出发） | steady |
| `-s, --speed` | 演化速度：`low`（保守，1 ~ 3 个修改点）、`medium`（均衡，3 ~ 6 个）、`high`（激进，不限制） | low |
| `--save-trace` | 将每轮执行轨迹落盘 | 关闭 |
| `--test-cases` | 指定测试用例 JSON 文件，跳过生成 | 无 |
| `--threshold` | 单轮 reward 提升验收阈值 | 0.01 |
| `--debug` | 启用 debug 中间件 | 关闭 |
| `--ignore-cache` | 忽略缓存 | 关闭 |
| `-o, --output` | 结果输出目录 | `results/<skill名>/evolve` |

示例：

```bash
# steady 模式，greedy 模式
skillflow evolve skills/my-skill -M steady -i 20 -p 10
skillflow evolve skills/my-skill -M greedy -i 30 -p 15 --save-trace

# 使用指定测试用例
skillflow evolve skills/my-skill \
  --test-cases results/my-skill/test_cases_202604081430.json \
  -j 5 -i 10 -p 5
```

## 输出目录结构

```
results/<skill-name>/
├── test_cases_202604081430.json          # 生成的测试用例
│
├── eval/                                 # 评估结果
│   └── 202604081707/                     # 每次评估独立时间戳目录
│       ├── eval.json                     # 评估结果（含各 case 得分）
│       └── trace/                        # 执行轨迹（--save-trace 时）
│           ├── case_0_trial_1.json
│           ├── case_0_trial_2.json
│           └── ...
│
└── evolve/                               # 演化结果
    └── 202604081800/                     # 每次演化独立时间戳目录
        ├── baseline.json                 # Baseline 评估结果
        ├── evolve_log.json               # 演化日志（含全部迭代历史）
        ├── baseline_skill/               # 原始技能备份
        │   └── <skill-name>/
        ├── iter-1/                       # 第 1 轮演化
        │   ├── analysis.md              # 本轮分析报告
        │   ├── eval.json                # 评估结果
        │   ├── <skill-name>/            # 演化后的技能副本
        │   └── trace/                   # 执行轨迹（--save-trace 时）
        ├── iter-2/
        └── ...
```

## 评估体系

### LLM-as-Judge 七维评分

每个测试用例的响应从 7 个维度评估，基础分 1.0，按问题扣分，下限 0.0：

| 维度 | 评估内容 |
|------|---------|
| 目标达成 | 是否完成测试点要求 |
| 内容逻辑 | 结构清晰、层次分明、逻辑连贯 |
| 事实准确 | 数据、概念、引用是否正确 |
| 场景适配 | 是否符合技能使用场景 |
| 格式规范 | 输出格式是否符合要求 |
| 边界处理 | 异常和边界条件是否处理 |
| 效率 | 步骤和 token 消耗是否合理 |

### Reward 计算

#### Case Reward

每个测试用例根据 n 次 trial 得分计算 case reward：

```
pass_rate = count(score >= 0.8) / n
pass@n    = 1 - (1 - pass_rate)^n     # 可达性：至少通过一次的概率
pass^n    = pass_rate^n               # 可靠性：全部通过的概率

case_reward = 0.5 × mean_score + 0.2 × pass@n + 0.3 × pass^n
```

- 通过阈值：`score >= 0.8` 视为通过
- `n` 为实际 trial 数，非硬编码

#### Overall Reward

所有用例的 reward 汇总为全局分数（归一化综合法）：

```
overall_reward = 0.8 × mean(case_rewards) + 0.2 × (1 - std(case_rewards))
```

- 均值权重 0.8：整体质量水平
- 标准差权重 0.2：各 case 表现稳定性（std 越小越好）

## 演化机制

### 三分支策略

每轮演化包含三个独立分支：

1. **Audit（审视）**：必执行，基于质量标准审视所有技能文件，就地修改
2. **Reflect（反思）**：有失败用例时执行，分析失败原因并改写 SKILL.md
3. **Exploit（范例提取）**：有近阈值失败响应时执行，提取优质模式注入技能

### 演化模式

| 模式 | 说明 | 轨迹来源 |
|------|------|---------|
| `steady` | 每轮从 baseline 技能出发 | 仅 baseline 执行轨迹 |
| `greedy` | 每轮从当前最优技能出发 | 每次接受后更新轨迹 |

### 演化防护

- **改动筛选**：Diff 比例 + LLM 语义判断双重过滤微小改动
- **Case 回归检测**：拒绝导致已通过 case 失败的演化
- **自适应阈值**：根据当前 reward 水平动态调整验收门槛
- **历史策略参考**：每轮迭代携带已尝试演化记录，避免重复方向
- **收敛检测**：连续 3 轮无文件修改提前终止

### 终止原因

| 原因 | 说明 |
|------|------|
| `converged` | 连续 3 轮判定无改进空间 |
| `plateaued` | 连续 patience 轮无 reward 提升 |
| `max_iterations` | 达到最大迭代次数 |

## 执行轨迹

开启 `--save-trace` 后，每个 trial 的完整执行轨迹保存为独立 JSON 文件，包含：

- **步骤记录**：LLM 调用、工具调用（名称、参数、结果、错误）
- **Token 统计**：每次 LLM 调用的 input/output tokens
- **技能触发检测**：是否使用了技能目录中的文件
- **评估结果**：最终响应、得分、是否通过

轨迹数据用于：
1. 演化时注入诊断信息，指导技能改进方向
2. 事后分析 agent 行为模式（路径稳定性、工具误用、pass/fail 分化点）

## 崩溃恢复

- **eval**：JSONL 增量保存，每个 trial 完成后追加写入，正常运行结束后自动清理
- **evolve**：每轮迭代后更新 `evolve_log.json`，崩溃后可查看已完成迭代

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
│   ├── runner.py          # 评估执行器（多 trial 并行、JSONL 增量保存）
│   ├── metrics.py         # 指标计算（pass@k、reward）
│   ├── trace.py           # 执行轨迹（提取、聚合、落盘）
│   └── cli.py             # CLI 命令
├── evolve/                # 进化模块
│   ├── mutator.py         # Skill 变异（审视-反思-重写）
│   ├── orchestrator.py    # 多轮进化编排（早停、历史策略）
│   ├── guards.py          # 演化防护（回归检测、改动筛选）
│   ├── exemplar.py        # 范例提取（高方差 trial 对比启发）
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
