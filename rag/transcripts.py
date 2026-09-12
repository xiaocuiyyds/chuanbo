"""页面转录的持久化存储。

转录是整个入库流程里最贵最慢的一步——每一页都要调一次视觉模型，几百页的手册要跑很久。
但在这一版之前，转录结果（PageText）在切块之后就被丢掉了，只有切好的 chunk 进了索引。
后果是切块参数、上下文 prompt、嵌入模型任何一项要调整，都得把整本手册重新过一遍视觉模型。

把转录单独落盘之后，流程变成三层，每一层都能从上一层纯函数地重建：

    page_images/ + transcripts/        贵，每份文档只做一次
        -> chunks.pkl                  切块 + 生成上下文，可反复重跑
        -> index.faiss                 嵌入，可反复重跑

目录结构（每份文档一个目录，每页一个 Markdown 文件）：

    data/transcripts/
        <目录名>/
            manifest.json    原始文档名、页数、每页对应的页图文件名
            p1.md
            p2.md

目录名是把文档名里的路径分隔符替换掉得到的，这个变换不可逆，所以原始文档名单独记在
manifest 里，不从目录名反推。
"""

import json
import re
import shutil
from pathlib import Path

from rag.pdf_loader import PageText

TRANSCRIPTS_DIRNAME = "transcripts"
MANIFEST_FILENAME = "manifest.json"


def _safe_dirname(source: str) -> str:
    return re.sub(r"[/\\]", "_", source)


def transcripts_dir(data_dir: Path) -> Path:
    return data_dir / TRANSCRIPTS_DIRNAME


def save_pages(data_dir: Path, source: str, pages: list[PageText]) -> Path:
    """把一份文档的所有页转录写入磁盘，返回该文档的目录。

    整个目录先删后写：重新上传同名文档时页数可能变少，残留的旧页会在重建索引时
    被当成真实内容读回来。
    """
    doc_dir = transcripts_dir(data_dir) / _safe_dirname(source)
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    doc_dir.mkdir(parents=True)

    for page in pages:
        (doc_dir / f"p{page.page}.md").write_text(page.text, encoding="utf-8")

    manifest = {
        "source": source,
        "pages": [{"page": p.page, "image_path": p.image_path} for p in sorted(pages, key=lambda p: p.page)],
    }
    (doc_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return doc_dir


def load_pages(data_dir: Path, source: str) -> list[PageText]:
    """读回一份文档的转录；没有存过则返回空列表。"""
    doc_dir = transcripts_dir(data_dir) / _safe_dirname(source)
    manifest_path = doc_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = []
    for entry in manifest["pages"]:
        page_path = doc_dir / f"p{entry['page']}.md"
        if not page_path.exists():
            continue  # 单页文件缺失不影响其余页面重建
        pages.append(
            PageText(
                source=manifest["source"],
                page=entry["page"],
                text=page_path.read_text(encoding="utf-8"),
                image_path=entry.get("image_path"),
            )
        )
    return pages


def list_sources(data_dir: Path) -> list[str]:
    """列出已存转录的所有文档名（按 manifest 里的原始名，不是目录名）。"""
    root = transcripts_dir(data_dir)
    if not root.exists():
        return []
    sources = []
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_FILENAME}")):
        try:
            sources.append(json.loads(manifest_path.read_text(encoding="utf-8"))["source"])
        except (json.JSONDecodeError, KeyError):
            continue  # 损坏的 manifest 跳过，不影响其他文档
    return sources


def load_all_pages(data_dir: Path) -> list[PageText]:
    pages = []
    for source in list_sources(data_dir):
        pages.extend(load_pages(data_dir, source))
    return pages


def remove_source(data_dir: Path, source: str) -> bool:
    doc_dir = transcripts_dir(data_dir) / _safe_dirname(source)
    if not doc_dir.exists():
        return False
    shutil.rmtree(doc_dir)
    return True
