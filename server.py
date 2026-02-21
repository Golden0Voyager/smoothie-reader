import os
import json
import hashlib
import pickle
import asyncio
import sqlite3
import tempfile
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import edge_tts
import httpx

# AI Imports
import google.generativeai as genai
from dotenv import load_dotenv

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry, process_epub, save_to_pickle

# Load .env file automatically
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Applying vercel-best-practice: Using gemini-3-flash-preview for speed/cost balance,
    # but the prompt engineering ensures Pro-level depth.
    model = genai.GenerativeModel('gemini-3-flash-preview')
else:
    print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment.")
    model = None

# Configure ZhipuAI (GLM-4.7-Flash, free model for translation)
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
_zhipu_client = None
if ZHIPUAI_API_KEY:
    _zhipu_client = httpx.AsyncClient(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        headers={"Authorization": f"Bearer {ZHIPUAI_API_KEY}", "Content-Type": "application/json"},
        timeout=30,
        trust_env=False,
    )
    _zhipu_headers = {"Authorization": f"Bearer {ZHIPUAI_API_KEY}", "Content-Type": "application/json"}
    print("ZhipuAI configured (GLM-4.7-Flash) for translation.")
else:
    print("Warning: ZHIPUAI_API_KEY not found, translation will use Gemini.")

# Google Translate direct API (fast, connection-pooled)
_gt_client = httpx.AsyncClient(timeout=5, http2=False)


async def _zhipu_chat(prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Non-streaming ZhipuAI call, returns full text response."""
    resp = await _zhipu_client.post("chat/completions", json={
        "model": "glm-4.7-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    })
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _zhipu_stream(prompt: str, temperature: float = 0.7, max_tokens: int = 4096):
    """Async generator yielding ZhipuAI streaming text chunks. Uses a fresh client to avoid connection pool issues."""
    try:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            async with client.stream("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers=_zhipu_headers,
                json={
                    "model": "glm-4.7-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except asyncio.CancelledError:
        return
    except Exception as e:
        print(f"[ZhipuAI stream error] {e}")
        return

def _detect_cjk_ratio(text):
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    return cjk / max(len(text), 1)

async def _google_translate(text, dest='zh-CN'):
    """Direct Google Translate API call, ~100ms with connection reuse."""
    resp = await _gt_client.get('https://translate.googleapis.com/translate_a/single', params={
        'client': 'gtx', 'sl': 'auto', 'tl': dest, 'dt': 't', 'q': text
    })
    data = resp.json()
    return ''.join(s[0] for s in data[0] if s[0])

# ECDICT offline dictionary (~3.4M entries, <1ms lookup)
_dict_db_path = os.path.join(os.path.dirname(__file__), 'dict', 'stardict.db')
_dict_conn = None
if os.path.exists(_dict_db_path):
    _dict_conn = sqlite3.connect(_dict_db_path, check_same_thread=False)
    _dict_conn.row_factory = sqlite3.Row

def _dict_lookup(word: str) -> dict | None:
    """Look up a word in the local ECDICT dictionary. Returns dict or None."""
    if not _dict_conn:
        return None
    row = _dict_conn.execute(
        'SELECT word, phonetic, translation, definition FROM stardict WHERE word = ? COLLATE NOCASE',
        (word.strip(),)
    ).fetchone()
    if not row or not (row['translation'] or row['definition']):
        return None
    return {
        'word': row['word'],
        'phonetic': row['phonetic'] or '',
        'translation': (row['translation'] or '').strip(),
        'definition': (row['definition'] or '').strip(),
    }

async def _wiki_summary(term: str) -> dict:
    """Fetch Wikipedia summary for a term. Auto-detect language."""
    import urllib.request, urllib.parse
    is_cjk = _detect_cjk_ratio(term) > 0.3
    lang = 'zh' if is_cjk else 'en'
    try:
        encoded = urllib.parse.quote(term)
        url = f'https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Reader3/1.0'})
        def _fetch():
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        data = await asyncio.to_thread(_fetch)
        return {
            'title': data.get('title', ''),
            'extract': data.get('extract', ''),
            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
        }
    except Exception:
        return {}

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."

# TTS audio cache directory
TTS_CACHE_DIR = os.path.join(BOOKS_DIR, ".tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# server-cache-lru: Multi-level cache for AI Analysis
# We use both in-memory and could easily extend to disk.
_analysis_cache = {}

def _get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

@lru_cache(maxsize=20)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """Load a Book object from its pickle file, with LRU caching."""
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    books = []
    if os.path.exists(BOOKS_DIR):
        for item in sorted(os.listdir(BOOKS_DIR)):
            if item.endswith("_data") and os.path.isdir(item):
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine),
                        "language": book.metadata.language or "en",
                    })
    return templates.TemplateResponse("library.html", {"request": request, "books": books})


def _find_cover_image(book_id: str) -> str | None:
    """Find cover image path for a book. Returns absolute path or None."""
    import re
    images_dir = os.path.join(BOOKS_DIR, book_id, "images")
    if not os.path.isdir(images_dir):
        return None

    img_exts = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(img_exts)]

    # 1. Marker file left by process_epub (most reliable)
    marker = os.path.join(BOOKS_DIR, book_id, "cover_image.txt")
    if os.path.exists(marker):
        fname = open(marker).read().strip()
        path = os.path.join(images_dir, fname)
        if os.path.exists(path):
            return path

    # 2. File named cover* (most explicit naming convention)
    for f in images:
        if re.match(r'cover', f, re.I):
            return os.path.join(images_dir, f)
    # 3. File containing *cover* anywhere in name
    for f in images:
        if 'cover' in f.lower():
            return os.path.join(images_dir, f)

    # 4. Parse first chapter HTML for image reference (often the cover page)
    book = load_book_cached(book_id)
    if book and book.spine:
        content = book.spine[0].content[:2000]
        m = re.search(r'(?:src|href)=["\']([^"\']+\.(?:jpe?g|png|gif|webp))', content, re.I)
        if m:
            src = m.group(1)
            fname = os.path.basename(src)
            path = os.path.join(images_dir, fname)
            if os.path.exists(path):
                return path

    # 5. Fallback: pick the largest image file (covers are usually the biggest)
    if images:
        largest = max(images, key=lambda f: os.path.getsize(os.path.join(images_dir, f)))
        return os.path.join(images_dir, largest)

    return None


@app.get("/api/book-cover/{book_id}")
async def serve_book_cover(book_id: str):
    """Serve cover image for an imported book."""
    safe_id = os.path.basename(book_id)
    cover = _find_cover_image(safe_id)
    if cover and os.path.exists(cover):
        return FileResponse(cover)
    raise HTTPException(status_code=404, detail="No cover found")

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: str):
    """Render a single chapter, or serve an image if chapter_index is a filename."""
    # Handle ../images/ or ../Images/ relative paths (book_id would be "images" or "Images")
    if book_id.lower() == 'images':
        referer = request.headers.get("referer", "")
        import re as _re
        from urllib.parse import unquote
        m = _re.search(r'/read/([^/]+)/', referer)
        if m:
            real_book_id = unquote(m.group(1))
            safe_name = os.path.basename(chapter_index)
            image_path = os.path.join(BOOKS_DIR, real_book_id, "images", safe_name)
            if os.path.exists(image_path):
                return FileResponse(image_path)
        raise HTTPException(status_code=404, detail="Image not found")

    # If it looks like a file (has extension), serve as image fallback
    if '.' in chapter_index:
        safe_name = os.path.basename(chapter_index)
        image_path = os.path.join(BOOKS_DIR, book_id, "images", safe_name)
        if os.path.exists(image_path):
            return FileResponse(image_path)
        raise HTTPException(status_code=404, detail="Not found")

    try:
        idx = int(chapter_index)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")

    book = load_book_cached(book_id)
    if not book or idx < 0 or idx >= len(book.spine):
        raise HTTPException(status_code=404, detail="Not found")

    current_chapter = book.spine[idx]
    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx + 1 if idx < len(book.spine) - 1 else None

    # Fix SVG cover distortion: replace preserveAspectRatio="none" and width/height="100%"
    import re as _re
    content = current_chapter.content
    if '<svg' in content:
        content = _re.sub(r'preserveaspectratio="none"', 'preserveAspectRatio="xMidYMid meet"', content, flags=_re.I)
        content = _re.sub(r'(<svg[^>]*)\s+width="100%"', r'\1', content, flags=_re.I)
        content = _re.sub(r'(<svg[^>]*)\s+height="100%"', r'\1', content, flags=_re.I)

    return templates.TemplateResponse("reader.html", {
        "request": request, "book": book, "current_chapter": current_chapter,
        "chapter_index": idx, "book_id": book_id,
        "prev_idx": prev_idx, "next_idx": next_idx,
        "chapter_content": content
    })


@app.get("/read/{book_id}/images/{image_name}")
async def serve_book_image(book_id: str, image_name: str):
    """Serve an image file from a book's extracted images directory."""
    safe_name = os.path.basename(image_name)
    image_path = os.path.join(BOOKS_DIR, book_id, "images", safe_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


# --- Delete books ---
@app.post("/api/delete-books")
async def delete_books(req: dict):
    """Delete one or more books by their IDs."""
    import shutil
    book_ids = req.get("book_ids", [])
    if not book_ids:
        raise HTTPException(status_code=400, detail="No books specified")
    deleted = []
    for bid in book_ids:
        safe_id = os.path.basename(bid)
        book_dir = os.path.join(BOOKS_DIR, safe_id)
        if os.path.isdir(book_dir) and os.path.exists(os.path.join(book_dir, "book.pkl")):
            await asyncio.to_thread(shutil.rmtree, book_dir)
            deleted.append(safe_id)
    load_book_cached.cache_clear()
    return {"deleted": deleted, "count": len(deleted)}

# --- AI MODULE REFACTORED (Gemini 3 Pro standards) ---

class AIAnalyzeRequest(BaseModel):
    book_id: str
    chapter_index: int

@app.post("/api/ai/analyze")
async def analyze_chapter(req: AIAnalyzeRequest):
    """Analyze a chapter and return structured insights."""
    if not _zhipu_client and not model:
        raise HTTPException(status_code=500, detail="AI not configured")

    book = load_book_cached(req.book_id)
    if not book or req.chapter_index < 0 or req.chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")
    chapter = book.spine[req.chapter_index]

    # Use text field, fallback to stripping HTML from content
    chapter_text = chapter.text.strip()
    if not chapter_text and chapter.content:
        from html.parser import HTMLParser
        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
            def handle_data(self, d):
                self.parts.append(d)
        s = _Strip()
        s.feed(chapter.content)
        chapter_text = ' '.join(s.parts).strip()

    if len(chapter_text) < 20:
        return {
            "summary": "本章内容过短，无法进行有效分析。",
            "key_points": [],
            "difficulties": "",
            "insight": ""
        }

    # server-cache-lru: Fingerprint based on text content hash
    content_hash = _get_text_hash(chapter_text)
    cache_key = f"{req.book_id}:{req.chapter_index}:{content_hash}"

    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    prompt = f"""你是一位经验丰富的读书会领读者，同时具备深厚的文学素养和跨学科知识。
你正在带领一群认真的成年读者讨论这本书。你的风格：有见地但不卖弄，善于发现文字背后的深意。

书名：{book.metadata.title}
作者：{', '.join(book.metadata.authors) if book.metadata.authors else '未知'}
章节：{chapter.title}

【本章内容】：
{chapter_text[:15000]}

请阅读后，以领读者的身份进行分析。严格按以下 JSON 返回（不要包含 ```json 标记）：

{{
    "summary": "用 150-250 字概述本章核心内容。不要罗列事件，而是说清楚：这一章到底在讲什么、推进了什么、改变了什么。如果是非虚构类，提炼核心论点和关键论据。",
    "key_points": [
        "提炼 3-5 个本章最值得关注的要点。每个要点用一句话点明是什么，再用一句话说明为什么重要。不要泛泛而谈。"
    ],
    "difficulties": "找出本章中读者可能卡住的地方：专业术语、文化背景、隐晦的表达、复杂的逻辑链等，用大白话解释清楚。如果没有难点就坦诚说明。",
    "insight": "分享一个有启发性的深层解读：可以是与其他作品的对比、一个反直觉的发现、当下社会的映射、或者作者没有明说但暗含的立场。要言之有物，避免空洞的感悟。"
}}"""

    try:
        if model:
            response = await asyncio.to_thread(model.generate_content, prompt)
            text = response.text.strip()
        elif _zhipu_client:
            text = await _zhipu_chat(prompt, temperature=0.3, max_tokens=8192)
            text = text.strip()
        else:
            raise HTTPException(status_code=500, detail="AI not configured")

        # Robust JSON cleaning
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]

        result = json.loads(text)
        _analysis_cache[cache_key] = result # Cache the processed object
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/ai/translate")
async def translate_text(req: dict):
    """Translate text using ZhipuAI with Gemini fallback."""
    if not _zhipu_client and not model:
        raise HTTPException(status_code=500, detail="AI not configured")
    text = req.get("text", "")
    if not text:
        return {"translation": ""}

    # Auto-detect: if mostly CJK → translate to English, otherwise → translate to Chinese
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    target = "英文" if cjk_count > len(text) * 0.3 else "中文"

    prompt = f"""请将以下文本翻译成{target}。

翻译要求：
- 准确传达原意，语句通顺自然，符合{target}的表达习惯
- 专有名词（人名、地名、术语）首次出现时附注原文
- 保留原文的语气和风格（正式/口语/文学/技术）
- 只返回译文，不要解释

原文：
{text}"""

    try:
        if _zhipu_client:
            result = await _zhipu_chat(prompt, temperature=0.1, max_tokens=4096)
            return {"translation": result.strip()}
        response = await asyncio.to_thread(model.generate_content, prompt)
        return {"translation": response.text.strip()}
    except Exception as e:
        return {"translation": f"[Translation Error: {str(e)}]"}


@app.post("/api/ai/translate-stream")
async def translate_text_stream(req: dict):
    """Translate text using ZhipuAI GLM-4.7-Flash (free) with Gemini fallback, streaming output."""
    if not _zhipu_client and not model:
        raise HTTPException(status_code=500, detail="AI not configured")
    text = req.get("text", "")
    if not text:
        return StreamingResponse(iter([""]), media_type="text/plain")

    # Auto-detect: if mostly CJK → translate to English, otherwise → translate to Chinese
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    target = "英文" if cjk_count > len(text) * 0.3 else "中文"

    # Build context from book metadata
    context = ""
    book_id = req.get("book_id")
    chapter_index = req.get("chapter_index")
    if book_id:
        book = load_book_cached(book_id)
        if book:
            meta = book.metadata
            parts = [f"书名《{meta.title}》"]
            if meta.authors:
                parts.append(f"作者{', '.join(meta.authors)}")
            if chapter_index is not None and 0 <= chapter_index < len(book.spine):
                ch_title = book.spine[chapter_index].title
                if ch_title:
                    parts.append(f"当前章节「{ch_title}」")
            context = f"[{', '.join(parts)}] "

    prompt = f"{context}将以下内容完整翻译成{target}，所有词汇都必须翻译，不得保留原文（专有名词首次出现时括号附注原文除外），只返回译文：\n{text}"

    # Prefer ZhipuAI (free), fall back to Gemini
    if _zhipu_client:
        async def generate_zhipu():
            try:
                async for chunk in _zhipu_stream(prompt, temperature=0.1, max_tokens=4096):
                    yield chunk
            except Exception as e:
                yield f"[Translation Error: {str(e)}]"

        return StreamingResponse(generate_zhipu(), media_type="text/plain; charset=utf-8")

    # Gemini fallback
    def generate_gemini():
        try:
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                ),
            )
            for chunk in response:
                try:
                    if chunk.text:
                        yield chunk.text
                except (ValueError, AttributeError):
                    continue
        except Exception as e:
            yield f"[Translation Error: {str(e)}]"

    return StreamingResponse(generate_gemini(), media_type="text/plain; charset=utf-8")


@app.post("/api/quick-translate")
async def quick_translate(req: dict):
    """Dict lookup (instant) → Google Translate fallback (~100ms)."""
    text = (req.get("text") or "").strip()
    if not text:
        return {"translation": ""}

    # Step 1: Try local dictionary for single words/short phrases
    dict_result = _dict_lookup(text)
    if dict_result:
        return {
            "source": "dict",
            "word": dict_result['word'],
            "phonetic": dict_result['phonetic'],
            "translation": dict_result['translation'],
            "definition": dict_result['definition'],
        }

    # Step 2: Fallback to Google Translate
    is_cjk = _detect_cjk_ratio(text) > 0.3
    dest = 'en' if is_cjk else 'zh-CN'
    try:
        translation = await _google_translate(text, dest)
        return {"source": "google", "translation": translation}
    except Exception as e:
        return {"source": "error", "translation": "", "error": str(e)}


@app.post("/api/wiki-lookup")
async def wiki_lookup(req: dict):
    """Wikipedia summary for a term."""
    text = (req.get("text") or "").strip()
    if not text:
        return {"extract": ""}
    result = await _wiki_summary(text)
    return result


@app.post("/api/search")
async def search_book(req: dict):
    """Full-text search across all chapters of a book."""
    book_id = (req.get("book_id") or "").strip()
    query = (req.get("query") or "").strip()
    if not book_id or not query:
        return {"results": []}
    book = load_book_cached(book_id)
    if not book:
        return {"results": []}
    lower_q = query.lower()
    results = []
    for ch in book.spine:
        text = ch.text or ""
        lower_text = text.lower()
        pos = 0
        while (pos := lower_text.find(lower_q, pos)) != -1:
            start = max(0, pos - 40)
            end = min(len(text), pos + len(query) + 40)
            snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
            results.append({
                "chapterIndex": ch.order,
                "chapterTitle": ch.title,
                "snippet": snippet,
                "matchStart": pos,
            })
            pos += len(query)
            if len(results) >= 200:
                break
        if len(results) >= 200:
            break
    return {"results": results}


class AIChatRequest(BaseModel):
    book_id: str
    chapter_index: int
    question: str


@app.post("/api/ai/chat")
async def chat_about_chapter(req: AIChatRequest):
    """Answer a free-form question about the current chapter."""
    if not _zhipu_client and not model:
        raise HTTPException(status_code=500, detail="AI not configured")

    book = load_book_cached(req.book_id)
    if not book or req.chapter_index < 0 or req.chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")
    chapter = book.spine[req.chapter_index]

    # Use text field, fallback to stripping HTML from content
    chapter_text = chapter.text.strip()
    if not chapter_text and chapter.content:
        from html.parser import HTMLParser
        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
            def handle_data(self, d):
                self.parts.append(d)
        s = _Strip()
        s.feed(chapter.content)
        chapter_text = ' '.join(s.parts).strip()

    prompt = f"""你是这本书的深度阅读伙伴。基于章节内容回答问题时：
- 尽量引用原文中的具体段落或细节来支撑回答
- 如果问题涉及更广的背景知识，可以拓展，但要标注哪些是原文内容、哪些是补充
- 用与提问者相同的语言回答
- 回答要有条理，必要时使用小标题分段

书名：{book.metadata.title}
作者：{', '.join(book.metadata.authors) if book.metadata.authors else '未知'}
章节：{chapter.title}

【章节内容】：
{chapter_text[:12000]}

【读者提问】：
{req.question}"""

    try:
        if model:
            def generate_gemini():
                try:
                    response = model.generate_content(
                        prompt,
                        stream=True,
                        generation_config=genai.types.GenerationConfig(temperature=0.7),
                    )
                    for chunk in response:
                        try:
                            if chunk.text:
                                yield chunk.text
                        except (ValueError, AttributeError):
                            continue
                except Exception as e:
                    yield f"\n[Error: {str(e)}]"

            return StreamingResponse(generate_gemini(), media_type="text/plain; charset=utf-8")

        if _zhipu_client:
            async def generate_zhipu():
                try:
                    async for chunk in _zhipu_stream(prompt, temperature=0.7, max_tokens=4096):
                        yield chunk
                except Exception as e:
                    yield f"\n[Error: {str(e)}]"

            return StreamingResponse(generate_zhipu(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# --- TTS MODULE: edge-tts (local, free, low latency) ---

@app.get("/api/tts")
async def stream_tts(text: str, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%"):
    """Stream TTS audio via edge-tts."""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    async def generate():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    return StreamingResponse(generate(), media_type="audio/mpeg")

# --- Apple Books Integration ---

APPLE_BOOKS_DB = os.path.expanduser(
    "~/Library/Containers/com.apple.iBooksX/Data/Documents/BKLibrary/BKLibrary-1-091020131601.sqlite"
)
APPLE_BOOKS_COVER_DIR = os.path.expanduser(
    "~/Library/Containers/com.apple.iBooksX/Data/Library/Caches/BCCoverCache-1/BICDiskDataStore"
)

@app.get("/api/apple-books")
async def list_apple_books():
    """List books from Apple Books library."""
    if not os.path.exists(APPLE_BOOKS_DB):
        return {"books": [], "error": "Apple Books database not found"}

    def _query():
        conn = sqlite3.connect(APPLE_BOOKS_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ZTITLE, ZAUTHOR, ZPATH, ZFILESIZE, ZASSETID FROM ZBKLIBRARYASSET "
            "WHERE ZTITLE IS NOT NULL AND ZPATH IS NOT NULL AND ZCONTENTTYPE = 1 "
            "ORDER BY ZTITLE"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    rows = await asyncio.to_thread(_query)

    books = []
    for r in rows:
        path = r['ZPATH']
        if not path or not os.path.exists(path):
            continue
        base_name = os.path.splitext(os.path.basename(path))[0]
        book_id = base_name + "_data"
        already_imported = os.path.exists(os.path.join(BOOKS_DIR, book_id, "book.pkl"))
        books.append({
            "title": r['ZTITLE'],
            "author": r['ZAUTHOR'] or '',
            "path": path,
            "size_mb": round((r['ZFILESIZE'] or 0) / 1048576, 1),
            "imported": already_imported,
            "asset_id": r['ZASSETID'] or '',
        })
    return {"books": books}


@app.get("/api/apple-books/cover/{asset_id}")
async def serve_apple_books_cover(asset_id: str):
    """Serve a cover image from Apple Books cache."""
    import subprocess
    safe_id = os.path.basename(asset_id)
    cover_dir = os.path.join(APPLE_BOOKS_COVER_DIR, safe_id)
    if not os.path.isdir(cover_dir):
        raise HTTPException(status_code=404, detail="Cover not found")
    # Check for cached JPEG first
    jpeg_path = os.path.join(cover_dir, f"{safe_id}.jpg")
    if os.path.exists(jpeg_path) and os.path.getsize(jpeg_path) > 0:
        return FileResponse(jpeg_path, media_type="image/jpeg")
    # Find largest HEIC file (best quality)
    import glob as _glob
    heics = sorted(_glob.glob(os.path.join(cover_dir, "*.heic")), key=os.path.getsize, reverse=True)
    if not heics:
        raise HTTPException(status_code=404, detail="No cover image")
    # Convert HEIC to JPEG synchronously (awaited)
    result = await asyncio.to_thread(
        subprocess.run,
        ["sips", "-s", "format", "jpeg", heics[0], "--out", jpeg_path],
        capture_output=True, timeout=10
    )
    if os.path.exists(jpeg_path) and os.path.getsize(jpeg_path) > 0:
        return FileResponse(jpeg_path, media_type="image/jpeg")
    raise HTTPException(status_code=500, detail="Cover conversion failed")


@app.post("/api/import-local")
async def import_local_epub(req: dict):
    """Import an EPUB from a local file path (e.g. Apple Books)."""
    path = req.get("path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=400, detail="File not found")
    if not path.lower().endswith('.epub'):
        raise HTTPException(status_code=400, detail="Only .epub files are supported")

    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(BOOKS_DIR, base_name + "_data")

    book_obj = await asyncio.to_thread(process_epub, path, out_dir)
    await asyncio.to_thread(save_to_pickle, book_obj, out_dir)

    # Keep a copy of the epub for future reprocessing
    import shutil
    epub_copy = os.path.join(out_dir, "source.epub")
    if not os.path.exists(epub_copy):
        shutil.copy2(path, epub_copy)

    load_book_cached.cache_clear()

    return {
        "success": True,
        "book_id": base_name + "_data",
        "title": book_obj.metadata.title,
        "chapters": len(book_obj.spine),
        "has_cover": _find_cover_image(base_name + "_data") is not None,
    }


@app.post("/api/upload")
async def upload_epub(file: UploadFile = File(...)):
    """Upload and process an EPUB file."""
    if not file.filename or not file.filename.lower().endswith('.epub'):
        raise HTTPException(status_code=400, detail="Only .epub files are supported")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Determine output dir name
        base_name = os.path.splitext(file.filename)[0]
        out_dir = os.path.join(BOOKS_DIR, base_name + "_data")

        # Process in thread to avoid blocking
        book_obj = await asyncio.to_thread(process_epub, tmp_path, out_dir)
        await asyncio.to_thread(save_to_pickle, book_obj, out_dir)

        # Keep a copy of the epub for future reprocessing
        import shutil
        shutil.copy2(tmp_path, os.path.join(out_dir, "source.epub"))

        # Clear LRU cache so new book appears
        load_book_cached.cache_clear()

        return {
            "success": True,
            "book_id": base_name + "_data",
            "title": book_obj.metadata.title,
            "chapters": len(book_obj.spine),
            "has_cover": _find_cover_image(base_name + "_data") is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process EPUB: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/reprocess/{book_id}")
async def reprocess_book(book_id: str):
    """Reprocess a book from its saved source.epub."""
    safe_id = os.path.basename(book_id)
    book_dir = os.path.join(BOOKS_DIR, safe_id)
    source_epub = os.path.join(book_dir, "source.epub")
    if not os.path.exists(source_epub):
        raise HTTPException(status_code=400, detail="No source epub found. Please re-upload the book.")

    # Copy source.epub to temp so process_epub can rmtree the output dir
    import shutil
    tmp_epub = source_epub + ".tmp"
    shutil.copy2(source_epub, tmp_epub)

    try:
        book_obj = await asyncio.to_thread(process_epub, tmp_epub, book_dir)
        await asyncio.to_thread(save_to_pickle, book_obj, book_dir)
        # Restore source.epub into the fresh output dir
        shutil.copy2(tmp_epub, os.path.join(book_dir, "source.epub"))
        load_book_cached.cache_clear()
        # Clear AI analysis cache for this book so stale results aren't served
        stale_keys = [k for k in _analysis_cache if k.startswith(f"{safe_id}:")]
        for k in stale_keys:
            del _analysis_cache[k]
        return {"success": True, "title": book_obj.metadata.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocess failed: {str(e)}")
    finally:
        if os.path.exists(tmp_epub):
            os.unlink(tmp_epub)


@app.post("/api/search-cover/{book_id}")
async def search_cover_online(book_id: str):
    """Search for book cover online using Google Books API (free, no key needed)."""
    safe_id = os.path.basename(book_id)
    book = load_book_cached(safe_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    title = book.metadata.title
    authors = ', '.join(book.metadata.authors) if book.metadata.authors else ''
    query = f"{title} {authors}".strip()

    import urllib.parse, urllib.request
    url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(query)}&maxResults=12"

    try:
        def _fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "Reader3/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read())
        data = await asyncio.to_thread(_fetch)

        covers = []
        for item in data.get("items", []):
            info = item.get("volumeInfo", {})
            images = info.get("imageLinks", {})
            # Prefer largest available
            img_url = images.get("extraLarge") or images.get("large") or images.get("medium") or images.get("thumbnail")
            if img_url:
                # Google Books returns http, upgrade to https; remove edge=curl for higher res
                img_url = img_url.replace("http://", "https://").replace("&edge=curl", "").replace("zoom=1", "zoom=2")
                covers.append({
                    "title": info.get("title", ""),
                    "authors": info.get("authors", []),
                    "image_url": img_url,
                })
        return {"covers": covers, "query": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cover search failed: {str(e)}")


@app.post("/api/set-cover/{book_id}")
async def set_cover_from_url(book_id: str, req: dict):
    """Download an image URL and set it as the book cover."""
    safe_id = os.path.basename(book_id)
    images_dir = os.path.join(BOOKS_DIR, safe_id, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_url = req.get("image_url", "")
    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL provided")

    import urllib.request
    try:
        def _download():
            req_obj = urllib.request.Request(image_url, headers={"User-Agent": "Reader3/1.0"})
            with urllib.request.urlopen(req_obj, timeout=10) as resp:
                return resp.read()
        img_data = await asyncio.to_thread(_download)

        # Save as cover.jpg
        cover_path = os.path.join(images_dir, "cover.jpg")
        with open(cover_path, "wb") as f:
            f.write(img_data)

        # Also write marker
        marker_path = os.path.join(BOOKS_DIR, safe_id, "cover_image.txt")
        with open(marker_path, "w") as f:
            f.write("cover.jpg")

        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download cover: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)
