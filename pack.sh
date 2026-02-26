#!/bin/bash
# 打包 Reader3 给其他人使用
# 用法: bash pack.sh
# 产出: reader3-release.zip

set -e
cd "$(dirname "$0")"

OUT="reader3-release"
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/dict" "$OUT/templates"

# 核心文件
cp server.py reader3.py requirements.txt setup.sh start.sh "$OUT/"
cp .env "$OUT/"
cp templates/reader.html templates/library.html "$OUT/templates/"

# 词典数据
[ -f dict/stardict.db ] && cp dict/stardict.db "$OUT/dict/"
[ -f dict/cn_dict.db ] && cp dict/cn_dict.db "$OUT/dict/"
[ -f dict/ecdict.zip ] && cp dict/ecdict.zip "$OUT/dict/"

# 中文词典构建用的源数据和脚本（以防需要重建）
cp dict/build_cn_dict.py "$OUT/dict/"
[ -f dict/ci_xinhua.json ] && cp dict/ci_xinhua.json "$OUT/dict/"
[ -f dict/idiom_xinhua.json ] && cp dict/idiom_xinhua.json "$OUT/dict/"
[ -f dict/moedict.json ] && cp dict/moedict.json "$OUT/dict/"

# 打包（不含 .venv、book data、__pycache__）
echo "📦 打包中..."
zip -r "$OUT.zip" "$OUT" -x "*.pyc" "*__pycache__*"

SIZE=$(du -sh "$OUT.zip" | cut -f1)
rm -rf "$OUT"

echo ""
echo "✅ 打包完成: $OUT.zip ($SIZE)"
echo ""
echo "使用说明："
echo "  1. 解压 reader3-release.zip"
echo "  2. 打开终端，进入解压后的文件夹"
echo "  3. 运行: bash setup.sh"
echo "  4. 运行: bash start.sh"
echo "  5. 浏览器打开: http://localhost:8000"
