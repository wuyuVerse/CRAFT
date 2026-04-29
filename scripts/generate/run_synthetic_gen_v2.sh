#!/bin/bash
# Synthetic Task Generation - 使用多个LLM Agent生成tau2格式任务
#
# 架构:
#   - Agent1 (TaskDesigner): 设计任务目标和reason_for_call
#   - Agent2 (ScenarioWriter): 编写详细的user_scenario
#   - Agent3 (CriteriaWriter): 编写evaluation_criteria
#
# 使用方法:
#   ./run_synthetic_gen_v2.sh config.yaml              # 运行所有领域（从配置读取）
#   ./run_synthetic_gen_v2.sh config.yaml airline      # 只运行指定领域
#   ./run_synthetic_gen_v2.sh config.yaml airline 100  # 指定领域和任务数

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON="$VENV_PATH/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="${PYTHON:-python3}"
fi

# 第一个参数是配置文件
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/generate/synthetic_tasks.example.yaml}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "使用方法:"
    echo "  $0 <config.yaml>              # 运行所有领域"
    echo "  $0 <config.yaml> <domain>     # 运行指定领域"
    echo ""
    echo "示例:"
    echo "  $0 configs/generate/synthetic_tasks.example.yaml"
    echo "  $0 configs/generate/synthetic_tasks.example.yaml airline"
    exit 1
fi

# 从配置文件读取默认值
OUTPUT_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output']['dir'])" 2>/dev/null || echo "output/synthetic_tasks")

# 获取所有配置的领域
ALL_DOMAINS=$($PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(cfg['generation']['num_tasks_per_domain'].keys()))" 2>/dev/null || echo "airline retail telecom")

# 如果指定了领域，只运行该领域；否则运行所有领域
if [ -n "$2" ]; then
    DOMAINS="$2"
    CUSTOM_NUM_TASKS="$3"
else
    DOMAINS="$ALL_DOMAINS"
    CUSTOM_NUM_TASKS=""
fi

echo "=========================================="
echo "Synthetic Task Generation (Multi-Agent)"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "运行领域: $DOMAINS"
echo "=========================================="
echo ""
echo "Agents:"
echo "  [1] TaskDesigner    - 设计任务目标"
echo "  [2] ScenarioWriter  - 编写用户场景"
echo "  [3] CriteriaWriter  - 编写评估标准"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/tau2-bench/src:$PYTHONPATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# 遍历所有领域
for DOMAIN in $DOMAINS; do
    # 获取该领域的任务数
    if [ -n "$CUSTOM_NUM_TASKS" ]; then
        NUM_TASKS="$CUSTOM_NUM_TASKS"
    else
        NUM_TASKS=$($PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg['generation']['num_tasks_per_domain'].get('$DOMAIN', 100))" 2>/dev/null || echo "100")
    fi

    echo "=========================================="
    echo "开始生成 $DOMAIN 领域 ($NUM_TASKS 个任务)"
    echo "=========================================="
    echo "增量数据: $OUTPUT_DIR/$DOMAIN/synthetic_data.jsonl"
    echo ""

    # 构建命令
    CMD="$PYTHON -m src.synthetic_gen.runners.synthetic_task_runner \
        --domain $DOMAIN \
        --num_tasks $NUM_TASKS \
        --output_dir $OUTPUT_DIR \
        --config $CONFIG_FILE"

    # 运行
    $CMD

    echo ""
    echo "[OK] $DOMAIN 领域生成完成!"
    echo ""
done

echo ""
echo "=========================================="
echo "全部生成完成！"
echo "=========================================="
echo ""
echo "输出文件:"
for DOMAIN in $DOMAINS; do
    echo "  [$DOMAIN]"
    echo "    - 增量数据: $OUTPUT_DIR/$DOMAIN/synthetic_data.jsonl"
    echo "    - 完整任务: $OUTPUT_DIR/$DOMAIN/synthetic_tasks_$DOMAIN.json"
done
echo ""
echo "日志目录: $OUTPUT_DIR/logs/"
echo ""
