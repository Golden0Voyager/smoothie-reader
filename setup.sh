#!/bin/bash
# Reader3 一键安装 & 启动脚本 (macOS)
# 用法: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "==============================="
echo "  Reader3 📚 安装向导"
echo "==============================="
echo ""

# 1. 检查 Python3
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 python3，请先安装："
    echo "   打开终端输入: xcode-select --install"
    echo "   或从 https://www.python.org/downloads/ 下载安装"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PY_VER"

# 2. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv .venv
fi

# 确保 pip 可用
.venv/bin/python3 -m ensurepip --upgrade 2>/dev/null || true

# 3. 安装依赖
echo "📦 安装依赖包（首次可能需要几分钟）..."
.venv/bin/python3 -m pip install -q --upgrade pip
.venv/bin/python3 -m pip install -q -r requirements.txt

# 4. 检查 .env 配置
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  未找到 .env 文件，需要配置 API Key"
    echo "   请输入 Gemini API Key（回车跳过）:"
    read -r gemini_key
    echo "GEMINI_API_KEY=\"${gemini_key}\"" > .env
    echo "✅ 已创建 .env"
else
    echo "✅ .env 已存在"
fi

# 5. 检查词典
if [ ! -f "dict/stardict.db" ] && [ -f "dict/ecdict.zip" ]; then
    echo "📖 解压英文词典..."
    cd dict && unzip -o ecdict.zip && cd ..
fi

if [ ! -f "dict/cn_dict.db" ] && [ -f "dict/build_cn_dict.py" ]; then
    echo "📖 构建中文词典..."
    .venv/bin/python3 dict/build_cn_dict.py
fi

echo ""
echo "==============================="
echo "  ✅ 安装完成！"
echo "==============================="
echo ""
echo "启动方式: bash start.sh"
echo "然后在浏览器打开: http://localhost:8000"
echo ""
