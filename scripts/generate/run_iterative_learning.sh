#!/bin/bash
# Iterative Learning 运行脚本
# 用法: bash scripts/generate/run_iterative_learning.sh [config.yaml]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON="$VENV_PATH/bin/python"

# 配置文件路径（默认使用 v4_ds_v2.yaml）
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/generate/iterative_learning.example.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ -f "$PYTHON" ]; then
    source "$VENV_PATH/bin/activate"
else
    PYTHON="${PYTHON:-python3}"
fi

# 添加 src 到 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/tau2-bench/src:$PYTHONPATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# 禁用 Pydantic 序列化警告
export PYTHONWARNINGS="ignore::pydantic.warnings.PydanticSerializationUnexpectedValue"

echo "=============================================="
echo "Iterative Learning 数据生成"
echo "=============================================="
echo "配置文件: $CONFIG_FILE"
echo "=============================================="

# 运行
$PYTHON -m src.iterative_learning.cli --config "$CONFIG_FILE"

echo "=============================================="
echo "完成!"
echo "=============================================="
