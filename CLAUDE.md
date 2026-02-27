# Reader3 — AI 智能电子书阅读器

## 项目简介

本地部署的 AI EPUB 阅读器，集成划词查词、AI 翻译/对话、TTS 朗读、高亮笔记系统。灵感源自 Karpathy 的同名项目，在其基础上深度重构。

## 架构

| 文件 | 职责 |
|------|------|
| `server.py` | FastAPI 后端 — 图书加载、AI 路由（20+ 提供商）、TTS (edge-tts)、Google Translate、ECDICT 词典、EPUB 上传 |
| `reader3.py` | EPUB 解析模块 |
| `templates/reader.html` | 阅读器页面（CSS + HTML + JS 单文件） |
| `templates/library.html` | 图书馆页面（封面墙、上传、Apple Books 扫描） |
| `tools/_md2pdf.py` | 文档转 PDF（Playwright/Chromium），源文件在 `docs/`，输出到 `package/` |
| `package/pack.sh` | 发行版打包脚本，自动生成 PDF 并打包为 `reader3-release.zip` |

## 常用命令

```bash
# 启动开发服务
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 打包发行版
cd package && bash pack.sh

# 生成文档 PDF
python tools/_md2pdf.py
```

## 开发规范

- **包管理**：始终使用 `uv pip install`，不用 pip
- **语言**：始终用中文回复
- **前端**：reader.html 是单文件架构（CSS + HTML + JS），不拆分
- **数据存储**：图书数据 `{name}_data/book.pkl`（pickle），高亮笔记在浏览器 localStorage
- **AI 配置**：`ai_config.json` 存储提供商设置，`.env` 存储 API Key

## 目录说明

```
reader3/
├── server.py / reader3.py    # 后端
├── templates/                 # 前端页面
├── docs/                      # 源文档 (md)
├── package/                   # 打包相关（pack.sh, setup.sh, start.sh, .env.example, 生成的 PDF）
├── tools/                     # 开发工具脚本
├── dict/                      # 词典文件（应用内下载）
└── books/                     # 导入的电子书数据
```

## 隐私约束

- `books/` 下的电子书数据、`.env`、`ai_config.json` 禁止提交
- 词典文件 (`dict/*.db`) 不提交
- 服务器日志 `server.log` 不提交
