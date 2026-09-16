"""用 MinerU 解析 PDF，替代逐页调用视觉模型转录。

原来的做法是把每一页渲染成图片、交给视觉模型重新读一遍。但全库 837 页里有 782 页
（93%）本来就带可提取的文字层——等于放着原文不用，让模型去重新生成它本可以直接读到的
内容。而模型在读不清时不会说"我做不到"，只会一直编下去，这正是转录退化的根源。

MinerU 优先用文字层、只在必要时 OCR，并做版面分析把标题、正文、表格分开。
五页样本上的实测对比（基准是 PDF 文字层里的原文）：

    设备位号     视觉模型 3/5 (60%)   MinerU 5/5 (100%)
    带单位数值   视觉模型 16/18       MinerU 18/18
    曾退化的两页  223/244、233/252 行套话   完全正常

视觉模型改错的那些，MinerU 都是对的：Shinko 不会变成 Shiniko、T21-PFM 不会变成
T21-PPFM、1,310kW 不会变成 1.310kW；连原文自带的排印错误 0.6.8MPa 都原样保留——
这正是"读取"和"重新生成"的区别。

图纸页上的设备位号仍然只能靠视觉识别（它们印在嵌入位图里，文字层中一个都没有），
那部分由 rag/diagram_tags.py 单独处理，跟这里互补。
"""

import html
import io
import json
import re
import time
import zipfile
from collections.abc import Callable

import requests

from rag.pdf_loader import PageText

API_BASE = "https://mineru.net/api/v4"
MODEL_VERSION = "vlm"
POLL_INTERVAL = 6  # 秒
POLL_TIMEOUT = 1800  # 单批最长等待，超过就当失败

# 正文里这些块不是内容，是每页都有的页眉页脚和页码
SKIP_TYPES = {"header", "footer", "page_number"}


def _latex_to_plain(text: str) -> str:
    """把行内公式还原成普通写法。

    MinerU 会把带单位的数值写成 LaTeX，例如 235m³/hour 输出为 $235m^{3}/hour$。
    检索时用户不会输入 LaTeX，BM25 也匹配不上，所以统一还原。
    """
    def unwrap(match: re.Match) -> str:
        body = match.group(1)
        body = re.sub(r"\^\{?3\}?", "³", body)
        body = re.sub(r"\^\{?2\}?", "²", body)
        body = re.sub(r"\\times", "×", body)
        body = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", body)
        body = re.sub(r"[{}\\]", "", body)
        return body.strip()

    return re.sub(r"\$([^$]{1,120})\$", unwrap, text)


def _table_to_markdown(table_html: str) -> str:
    """把 MinerU 输出的 HTML 表格转成 Markdown 表格。

    必须转：chunker 是按行首的 | 来识别表格边界的，靠它保证表格不被从中间切断、
    以及超大表格拆分时每段重复表头。留着 HTML 的话这套逻辑会整个失效。

    colspan 用重复单元格来补齐，宁可冗余也不要让各行列数对不上。
    """
    rows: list[list[str]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S | re.I):
        cells: list[str] = []
        for attrs, cell in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row_html, re.S | re.I):
            text = re.sub(r"<[^>]+>", " ", cell)
            text = html.unescape(text)
            text = _latex_to_plain(text)
            text = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
            span = re.search(r'colspan\s*=\s*"?(\d+)', attrs, re.I)
            cells.extend([text] * (int(span.group(1)) if span else 1))
        if cells:
            rows.append(cells)

    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(out)


def blocks_to_pages(blocks: list[dict], source: str, image_name: Callable[[int], str | None]) -> list[PageText]:
    """把 MinerU 的 content_list 还原成按页的转录。

    content_list 里每个块都带 page_idx，所以能精确还原到页——这一点很重要，
    转录层是按页存的，下游的检索结果、引用页码、页面原图全都依赖页号对得上。
    """
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        btype = block.get("type")
        if btype in SKIP_TYPES:
            continue
        try:
            page = int(block.get("page_idx", 0)) + 1
        except (TypeError, ValueError):
            continue

        if btype == "table":
            piece = _table_to_markdown(block.get("table_body") or "")
            caption = " ".join(block.get("table_caption") or [])
            if caption:
                piece = f"{caption}\n{piece}"
        elif btype == "image":
            piece = " ".join(block.get("image_caption") or [])
        else:
            text = _latex_to_plain(block.get("text") or "")
            level = block.get("text_level")
            piece = f"{'#' * min(int(level), 6)} {text}" if level and text else text

        if piece and piece.strip():
            by_page.setdefault(page, []).append(piece.strip())

    return [
        PageText(source=source, page=page, text="\n\n".join(parts), image_path=image_name(page))
        for page, parts in sorted(by_page.items())
    ]


def parse_pdf(
    api_key: str,
    pdf_bytes: bytes,
    file_name: str,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """把一份 PDF 交给 MinerU 解析，返回 content_list（带 page_idx 的块列表）。

    走三步：申请带签名的上传地址 -> PUT 上传 -> 轮询批次结果。
    上传那一步不能带任何 Content-Type 请求头——签名是按"无 Content-Type"算的，
    加了就签名不匹配、返回 403（urllib 会自动加，所以这里必须用 requests）。
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        f"{API_BASE}/file-urls/batch",
        headers=headers,
        timeout=90,
        json={
            "files": [{"name": file_name}],
            "model_version": MODEL_VERSION,
            "enable_formula": False,
            "enable_table": True,
            "language": "en",
        },
    )
    response.raise_for_status()
    data = response.json().get("data") or {}
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls") or []
    if not (batch_id and file_urls):
        raise RuntimeError(f"MinerU 未返回上传地址：{response.text[:300]}")

    requests.put(file_urls[0], data=pdf_bytes, timeout=1800).raise_for_status()
    if on_progress:
        on_progress(f"已上传 {len(pdf_bytes) / 1048576:.1f}MB，批次 {batch_id}")

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        poll = requests.get(
            f"{API_BASE}/extract-results/batch/{batch_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=90,
        ).json()
        result = ((poll.get("data") or {}).get("extract_result") or [{}])[0]
        state = result.get("state")
        if state == "done":
            archive = requests.get(result["full_zip_url"], timeout=600).content
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                name = next(n for n in bundle.namelist() if n.endswith("content_list.json"))
                return json.loads(bundle.read(name).decode("utf-8"))
        if state in ("failed", "error"):
            raise RuntimeError(f"MinerU 解析失败：{json.dumps(result, ensure_ascii=False)[:300]}")
        if on_progress:
            progress = result.get("extract_progress") or {}
            if progress:
                on_progress(f"{progress.get('extracted_pages')}/{progress.get('total_pages')} 页")

    raise TimeoutError(f"MinerU 解析超时（批次 {batch_id}）")
