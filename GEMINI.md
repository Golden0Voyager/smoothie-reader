# Project Intelligence: Reader3

一个现代化的、支持 AI 交互的 EPUB 阅读器。

## 🚀 运行环境 (Runtime)
- **后端**: FastAPI
- **音频引擎**: Edge-TTS
- **环境管理**: `uv` (强制)

## 🧠 AI 协作规范 (AI Patterns)
- **模型选型**: 已固定为 **gemini-3-flash-preview** (追求极速响应)。
- **核心功能**: 负责全文翻译、内容摘要及 TTS 文本预处理。

## 🧹 缓存规范 (Cache)
- **数据缓存**: `books/` 目录下的解析结果和 `cache/` 音频严禁提交。
