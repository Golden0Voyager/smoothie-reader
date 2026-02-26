#!/bin/bash
# Reader3 启动脚本
# 用法: bash start.sh

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "❌ 请先运行安装: bash setup.sh"
    exit 1
fi

echo "📚 Reader3 启动中..."
echo "   浏览器打开: http://localhost:8000"
echo "   按 Ctrl+C 停止"
echo ""

.venv/bin/python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
