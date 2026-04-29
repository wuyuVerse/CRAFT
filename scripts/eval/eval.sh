#!/bin/bash
# tau2-bench 并行评测脚本
#
# 用法: 
#   bash scripts/eval/eval.sh <config.yaml>
#   bash scripts/eval/eval.sh <config.yaml> --num-runs 4
#   bash scripts/eval/eval.sh <config.yaml> --domains airline retail
#
# 示例:
#   bash scripts/eval/eval.sh configs/eval/example.yaml
#   bash scripts/eval/eval.sh configs/eval/example.yaml --num-runs 4

set -e

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <config.yaml> [options]"
    echo ""
    echo "选项:"
    echo "  --num-runs N       每个榜测几次取平均（覆盖配置文件）"
    echo "  --domains D1 D2    指定评测领域（覆盖配置文件）"
    echo "  --num-tasks N      每个领域测试多少任务（覆盖配置文件）"
    echo "  --max-concurrency N 并发数（覆盖配置文件）"
    echo "  -v, --verbose      显示详细日志"
    echo ""
    echo "示例:"
    echo "  $0 configs/eval/example.yaml"
    echo "  $0 configs/eval/example.yaml --num-runs 4"
    echo "  $0 configs/eval/example.yaml --domains airline retail --num-runs 2"
    exit 1
fi

CONFIG_FILE="$1"
shift  # 移除第一个参数，剩余的传给Python

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON="$VENV_PATH/bin/python"

if [ ! -f "$PYTHON" ]; then
    PYTHON="${PYTHON:-python3}"
else
    source "$VENV_PATH/bin/activate"
fi

# 设置环境变量
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LITELLM_LOG="ERROR"
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/tau2-bench/src:$PYTHONPATH"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "=============================================="
echo "tau2-bench 并行评测"
echo "=============================================="
echo "配置文件: $CONFIG_FILE"
echo "项目目录: $PROJECT_ROOT"
echo "=============================================="
echo ""

# 运行评测
$PYTHON -m src.eval.main "$CONFIG_FILE" "$@"

echo ""
echo "=============================================="
echo "评测完成!"
echo "=============================================="
