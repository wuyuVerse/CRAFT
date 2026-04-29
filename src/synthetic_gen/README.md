# Synthetic Data Generation Module

集成 tau-bench-gen 的合成数据生成功能，用于扩展训练数据的多样性和规模。

## 模块结构

```
synthetic_gen/
├── core/                     # 核心逻辑（tau-bench-gen原始文件）
│   ├── playground.py         # 多轮任务执行模拟器
│   ├── agents.py             # Agent模拟器
│   ├── multi_turn_task_gen.py # 多轮任务生成器
│   ├── self_reflection.py    # 自我反思
│   ├── error_correction_pruner.py # 错误剪枝
│   ├── synthetic_reward.py   # 合成奖励
│   └── sampler.py            # 采样器
│
├── prompts/                  # Prompt模板
│   ├── prompt.py             # 通用Prompt
│   └── tau_prompt.py         # Tau-bench prompts
│
├── utils/                    # 工具函数
│   ├── logger.py             # 日志
│   ├── json_utils.py         # JSON工具
│   ├── cost_calculator.py    # 成本计算
│   └── format_qwen25.py      # 格式转换
│
├── runners/                  # 运行器（新增功能）
│   ├── synthetic_runner.py   # 合成数据生成运行器
│   ├── data_merger.py        # 数据合并器
│   └── quality_filter.py     # 质量过滤器
│
├── api/                      # API定义
│   ├── airline_tool.json
│   ├── retail_tool.json
│   └── telecom_tool.json
│
└── examples/                 # 示例代码
    ├── example_usage.py
    └── main.py
```

## 快速开始

### 1. 生成合成数据

```bash
# 生成所有领域的合成数据
bash scripts/generate/run_synthetic.sh

# 或单独生成某个领域
python -m src.synthetic_gen.runners.synthetic_runner \
    --config scripts/generate/config/synthetic_v1.yaml \
    --domain airline \
    --num-samples 1000
```

### 2. 质量过滤（可选）

```bash
python -m src.synthetic_gen.runners.quality_filter \
    --config scripts/generate/config/synthetic_v1.yaml \
    --input output/synthetic_v1 \
    --output output/synthetic_v1_filtered \
    --mode directory
```

### 3. 合并数据

```bash
bash scripts/generate/merge_data.sh
```

## 模块说明

### runners/synthetic_runner.py
- 封装 tau-bench-gen 的 PlayGround
- 生成合成任务数据
- 支持多API负载均衡和并发控制

### runners/data_merger.py
- 合并真实任务数据和合成任务数据
- 按比例采样
- 格式统一转换

### runners/quality_filter.py
- 过滤低质量的合成数据
- 多维度质量检查
- 生成过滤统计报告

## 核心文件（从 tau-bench-gen 复制）

### core/
- `playground.py` - 多轮任务执行模拟器
- `agents.py` - Agent模拟器（User, ToolAgent, ToolSimulator）
- `multi_turn_task_gen.py` - 多轮任务生成器
- `self_reflection.py` - 自我反思机制
- `error_correction_pruner.py` - 错误纠正剪枝器
- `synthetic_reward.py` - 合成奖励计算器
- `sampler.py` - 采样器

### prompts/
- `prompt.py` - 通用Prompt模板
- `tau_prompt.py` - Tau-bench system prompts

### utils/
- `logger.py` - 日志工具
- `json_utils.py` - JSON工具
- `cost_calculator.py` - 成本计算器
- `format_qwen25.py` - 格式转换工具

## 配置文件

- `scripts/generate/config/synthetic_v1.yaml` - 合成数据生成配置
- `scripts/generate/config/merge_config.yaml` - 数据合并配置

## 注意事项

1. **完全保留原始逻辑** - 所有核心文件从 tau-bench-gen 复制，只修改导入路径
2. **计算成本** - 合成数据生成需要大量API调用，注意控制并发数
3. **质量控制** - 建议启用质量过滤，定期抽查数据质量
4. **混合比例** - 建议真实数据占比≥50%

## 详细文档

参见 `docs/iterative_learning_v5.md` 和 `docs/restructure_verification.md`
