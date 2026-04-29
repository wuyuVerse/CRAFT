#!/bin/bash
# Data Merge 运行脚本
# 用法: bash scripts/generate/run_merge.sh [config.yaml]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON="$VENV_PATH/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="${PYTHON:-python3}"
fi

# 配置文件路径（默认使用 merge_config.yaml）
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/generate/merge.example.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 添加 src 到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/tau2-bench/src:$PYTHONPATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# 禁用 Pydantic 序列化警告
export PYTHONWARNINGS="ignore::pydantic.warnings.PydanticSerializationUnexpectedValue"

echo "=============================================="
echo "Data Merge"
echo "=============================================="
echo "配置文件: $CONFIG_FILE"
echo "=============================================="

# 运行
$PYTHON -m src.synthetic_gen.runners.data_merger \
    --config "$CONFIG_FILE"

echo ""
echo "=============================================="
echo "完成!"
echo "=============================================="
