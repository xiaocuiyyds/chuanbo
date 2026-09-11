import re
from dataclasses import dataclass

from rag.pdf_loader import PageText

TARGET_CHUNK_SIZE = 800  # 贪心打包的软目标：多个小块会被合并到这个大小左右
MAX_CHUNK_SIZE = 1600  # 硬上限：超过这个大小的单个块（通常是大表格）会被再拆开
CHAR_SPLIT_OVERLAP = 100

HEADER_RE = re.compile(r"^#{1,6}\s")
TABLE_ROW_RE = re.compile(r"^\s*\|")


@dataclass
class Chunk:
    source: str
    page: int
    text: str
    image_path: str | None = None
    context: str | None = None  # Contextual Retrieval：定位性说明，只用于检索，不用于展示/生成

    @property
    def retrieval_text(self) -> str:
        """用于 embedding 和 BM25 索引的文本。拼接上下文能帮助区分内容相似的片段
        （比如同一张表格拆出来的多个碎片），但不改变 chunk.text 本身，
        展示给用户和喂给回答模型时用的还是原文。"""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


def _split_into_blocks(text: str) -> list[str]:
    """按标题和表格边界把一页 Markdown 切成结构化的块，表格不会被从中间断开。"""
    blocks: list[str] = []
    current: list[str] = []
    current_type: str | None = None

    def flush() -> None:
        joined = "\n".join(current).strip("\n")
        if joined.strip():
            blocks.append(joined)
        current.clear()

    for line in text.split("\n"):
        if not line.strip():
            flush()
            current_type = None
            continue

        is_header = bool(HEADER_RE.match(line))
        line_type = "table" if TABLE_ROW_RE.match(line) else "text"

        if current_type is not None and line_type != current_type:
            flush()
        elif is_header and current:
            flush()

        current_type = line_type
        current.append(line)

    flush()
    return blocks


def _split_table(lines: list[str], max_size: int) -> list[str]:
    """按行拆分过大的表格，每一段都重复表头，保证独立可读。"""
    header_lines = lines[:2] if len(lines) >= 2 else lines
    header_text = "\n".join(header_lines)
    body_lines = lines[2:] if len(lines) >= 2 else []

    parts = []
    current_rows: list[str] = []
    current_len = len(header_text)
    for row in body_lines:
        if current_rows and current_len + len(row) + 1 > max_size:
            parts.append(header_text + "\n" + "\n".join(current_rows))
            current_rows = []
            current_len = len(header_text)
        current_rows.append(row)
        current_len += len(row) + 1

    if current_rows:
        parts.append(header_text + "\n" + "\n".join(current_rows))
    return parts or [header_text]


def _split_by_chars(text: str, max_size: int, overlap: int = CHAR_SPLIT_OVERLAP) -> list[str]:
    parts = []
    start = 0
    while start < len(text):
        end = start + max_size
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return parts


def _split_oversized_block(block: str, max_size: int) -> list[str]:
    lines = block.split("\n")
    if lines and TABLE_ROW_RE.match(lines[0]):
        return _split_table(lines, max_size)
    return _split_by_chars(block, max_size)


def _pack_blocks(blocks: list[str], target_size: int, max_size: int) -> list[str]:
    normalized: list[str] = []
    for block in blocks:
        if len(block) > max_size:
            normalized.extend(_split_oversized_block(block, max_size))
        else:
            normalized.append(block)

    chunks: list[str] = []
    current = ""
    for block in normalized:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= target_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = block
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(
    pages: list[PageText],
    target_size: int = TARGET_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[Chunk]:
    chunks = []
    for page in pages:
        blocks = _split_into_blocks(page.text)
        for piece in _pack_blocks(blocks, target_size, max_size):
            if piece.strip():
                chunks.append(
                    Chunk(source=page.source, page=page.page, text=piece, image_path=page.image_path)
                )
    return chunks
