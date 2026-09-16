import base64
import concurrent.futures
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VL_MODEL = "qwen-vl-plus"
VL_ZOOM = 2.5  # ~180 DPI，兼顾清晰度和图片体积

VL_PROMPT = (
    "请将这页船舶设备手册的内容完整转录为 Markdown。"
    "表格请转成 Markdown 表格；如果是接线图/流程图/示意图，"
    "请用文字描述图中的关键组件、标注和连接关系；"
    "保持原文语言，不要翻译，不要编造图中没有的内容。"
)

# 只在检测到退化时用来重试。对着位号密集的管路图，模型偶尔会放弃真正读图，改成逐个列举
# 位号、给每个都套同一句废话（索引里 Machinery p112 就有 206 句"with specific notes on
# its function"），或者干脆按编号顺序臆造出一长串连续位号。这段额外约束是针对这种情况的，
# 平时不用——实测正常情况下它相比原 prompt 没有可测量的收益。
STRICT_VL_PROMPT = VL_PROMPT + (
    "\n注意：图中位号标签很多时，只转录你能确实读清的关键位号，并说明主要设备之间的连接关系；"
    "不要试图逐个列举所有位号，更不要按编号顺序臆造连续的位号列表。"
    "严禁把同一句描述重复套用到多个元件上。"
)

# 雷同行至少要有这么多。定在 20 是对着真实语料量出来的：退化页面的重复行动辄上百
# （Machinery p184 有 313 行），而正常页面上重复最多的是每页页脚的
# "Draft Manual for Review & Comment" 水印，在内容稀疏的章节首页上也只有十几行。
# 取 8 会把 6 个这类页面误判成退化（白白多一次转录调用），取 20 则一个不误报，
# 同时已知的退化页仍然全部命中。
DEGENERATE_MIN_LINES = 20
DEGENERATE_LINE_RATIO = 0.35  # 且要占全部内容行的三成半以上
DEGENERATE_MIN_REPEAT = 3  # 同一个句式至少重复这么多次才算进雷同行
DEGENERATE_MIN_SUBSTANCE = 15  # 句式里至少要有这么多字母/数字/汉字，滤掉分隔线之类
STRICT_RETRIES = 3  # 检测到退化后，用严格 prompt 最多重试几次

# 归一化时把位号、编号替换掉，这样"同一句话换个位号"能被识别成同一个句式
_TAG_RE = re.compile(r"[A-Za-z]{0,6}[-\s]?\d[\w\-./]*")
_SUBSTANCE_RE = re.compile(r"[0-9A-Za-z一-鿿]")


@dataclass
class PageText:
    source: str
    page: int
    text: str
    image_path: str | None = None  # images_dir 下的文件名，回答阶段用它把原图传给多模态模型


def _render_page_png(doc: fitz.Document, page_index: int) -> bytes:
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(VL_ZOOM, VL_ZOOM))
    return pix.tobytes("png")


def _degenerate_reason(text: str, finish_reason: str | None) -> str | None:
    """判断一次转录是否失败，返回原因；正常则返回 None。

    两种失败都会静默地把垃圾写进索引，所以必须显式检查：
    一是撞上 max_tokens 被截断（后半页内容直接丢失），
    二是模型放弃读图、改成重复同一句套话逐个列举位号。
    """
    if finish_reason == "length":
        return "输出撞到 max_tokens 上限，页面内容不完整"

    # Markdown 表格行天然高度雷同，正常转录的表格不该被判成退化，先排除掉
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if len(line) > 12 and not line.startswith("|")]
    if len(lines) < DEGENERATE_MIN_LINES:
        return None

    counts = Counter(_TAG_RE.sub("#", line) for line in lines)
    repeated = {
        pattern: count
        for pattern, count in counts.items()
        if count >= DEGENERATE_MIN_REPEAT
        and len(_SUBSTANCE_RE.findall(pattern)) >= DEGENERATE_MIN_SUBSTANCE
    }
    # 看所有重复句式加起来占了多少行，而不是只看最高频的那一个——退化时模型经常是
    # 几个模板轮流用（"位于图中X处" / "连接到Y" / "属于Z系统"），只看 top1 会被摊薄漏掉
    mass = sum(repeated.values())
    if mass >= DEGENERATE_MIN_LINES and mass / len(lines) > DEGENERATE_LINE_RATIO:
        top_pattern, top_count = max(repeated.items(), key=lambda item: item[1])
        return f"{mass}/{len(lines)} 行是重复套话（如 {top_pattern[:40]!r} 出现 {top_count} 次）"
    return None


def _call_vl(client: OpenAI, b64: str, prompt: str) -> tuple[str, str | None]:
    response = client.chat.completions.create(
        model=VL_MODEL,
        max_tokens=4000,  # 安全上限：防止模型在重复性表格上失控续写，兜底截断而不是无限生成
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    choice = response.choices[0]
    return choice.message.content or "", choice.finish_reason


def _transcribe_page(client: OpenAI, png_bytes: bytes) -> str:
    """转录一页；检测到截断或重复套话时，换用更严格的 prompt 重试若干次。

    退化是偶发的，同一页同一 prompt 的结果在多次调用之间差别很大，所以重试本身就有意义，
    不能只靠调 prompt 规避。但只重试一次不够——实测最顽固的两页里，严格 prompt 在 p146 上
    三次只成功一次，单次重试有很大概率白跑。改成最多试 STRICT_RETRIES 次，拿到干净结果
    就停。全部失败时返回最长的那次：内容不全也好过没有。
    """
    b64 = base64.b64encode(png_bytes).decode()
    text, finish_reason = _call_vl(client, b64, VL_PROMPT)
    if _degenerate_reason(text, finish_reason) is None:
        return text

    attempts = [text]
    for _ in range(STRICT_RETRIES):
        retry_text, retry_finish = _call_vl(client, b64, STRICT_VL_PROMPT)
        if _degenerate_reason(retry_text, retry_finish) is None:
            return retry_text
        attempts.append(retry_text)
    return max(attempts, key=len)


def extract_pages(
    api_key: str,
    file_bytes: bytes,
    source_name: str,
    images_dir: Path,
    max_workers: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[PageText]:
    """把每一页渲染成图片，用 Qwen-VL 转录为 Markdown（表格转 markdown 表格，
    接线图/流程图给出文字描述），比纯文本提取能保留更多手册中的结构化信息。
    渲染出的原图会存到 images_dir，供回答阶段的多模态模型回看原图核实转录漏掉的细节。
    """
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        n_pages = len(doc)
        page_images = [_render_page_png(doc, i) for i in range(n_pages)]
    finally:
        doc.close()

    images_dir.mkdir(parents=True, exist_ok=True)
    safe_source = re.sub(r"[/\\]", "_", source_name)
    image_filenames = []
    for i, png_bytes in enumerate(page_images):
        filename = f"{safe_source}_p{i + 1}.png"
        (images_dir / filename).write_bytes(png_bytes)
        image_filenames.append(filename)

    results: list[str | None] = [None] * n_pages
    done_count = 0

    def worker(index: int) -> tuple[int, str]:
        try:
            text = _transcribe_page(client, page_images[index])
        except Exception as e:
            text = f"[第{index + 1}页转录失败: {e}]"
        return index, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(n_pages)]
        for future in concurrent.futures.as_completed(futures):
            index, text = future.result()
            results[index] = text
            done_count += 1
            if on_progress:
                on_progress(done_count, n_pages)

    pages = []
    for i, text in enumerate(results):
        if text and text.strip():
            pages.append(
                PageText(source=source_name, page=i + 1, text=text, image_path=image_filenames[i])
            )
    return pages
