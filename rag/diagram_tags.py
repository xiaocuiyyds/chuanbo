"""从图纸页里抽取设备位号。

手册里的管路图、接线图是以位图形式嵌在 PDF 里的，位号印在图上、不在文字层里
（整本 Machinery 手册的文字层中 GFV、GFP 都是 0 次），所以只能靠视觉模型读。

但直接把整页丢给视觉模型要它"描述图中的组件、标注和连接关系"，结果是两种失败：
要么给一篇三千字的泛泛描述、一个真位号都不含；要么退化成把同一句套话套到几十个
位号上，其中的位号还是编造的——索引里 Machinery p112 那份坏转录就编出了 162 个
GFP 开头的位号，而这个前缀全书不存在。

这里换一套做法，三点缺一不可：

1. 按嵌入位图的原生分辨率渲染页面。入库流程固定用 2.5 倍渲染，把 3902 像素宽的
   图降到 2977，位号本来就小，一降更难认。注意是"渲染"而不是直接取嵌入图字节——
   有些页面存的位图是翻转的，靠页面矩阵摆正，直接取字节会拿到上下颠倒又镜像的图。
2. 切成小块分别识别，并且 prompt 只要位号原文、明令禁止写描述。
   "描述连接关系"这类指令正是幻觉的诱因。
3. 同一块跑多遍，按多数票保留。模型输出的跑间方差很大——同一页四遍分别给出
   182、125、121、53 个候选。早先用"两遍取交集"，等价于要求 2/2 全票，
   结果只要有一遍发挥差就把好位号一票否决掉（实测 p112 因此丢了人工核对过的
   GFV19~GFV22）。改成三遍里至少两遍读到，单次坏结果不再具有否决权。
   实测人工核对的 21 个基准位号在各档阈值下都是 21/21 全召回，
   而噪声候选从 ≥1 票的 251 个降到 ≥3 票的 21 个。

只抽位号，不抽器件类型和连接关系——后两者实测不可靠：装在风机出口立管上的阀门
GFV21/GFV22 会被判成"泵/风机"（被紧邻的风机符号带偏），相邻编号的管路
GVA11 和 GVA21 会混淆。位号是"抄"，类型和拓扑是"判读"，模型只有前者靠得住。
"""

import base64
import concurrent.futures
import io
import re
from collections import Counter

import fitz
from openai import OpenAI
from PIL import Image

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VL_MODEL = "qwen-vl-plus"

TILE_GRID = 3  # 切成 3x3
TILE_OVERLAP = 0.08  # 相邻块重叠比例，避免位号正好落在切缝上被切成两半
RUNS = 3  # 每块跑几遍
MIN_VOTES = 2  # 至少几遍读到才算数（多数票）
MIN_DIAGRAM_PIXELS = 4_000_000  # 判定为图纸页的嵌入图最小像素数
MAX_DIAGRAM_TEXT = 1500  # 图纸页的文字层字符数上限（标注都在图里，文字层必然稀疏）
MIN_RENDER_ZOOM = 2.5  # 渲染倍数下限，跟入库流程保持一致
MAX_RENDER_ZOOM = 6.0  # 上限，防止超大图纸渲染出几亿像素

# 刻意不在 prompt 里举真实位号的例子。最初版本举了几个，结果模型把示例原样抄进
# 输出——舱底布置图上竟然出现了燃气系统图才有的那个位号，而且两遍都出现，连一致性
# 过滤都骗过去了。示例会泄漏成答案，所以这里只描述形态。
TAG_PROMPT = (
    "这是一张船舶图纸（管路图、接线图或布置图）的局部放大图。"
    "请只把图中出现的设备位号、编号标签逐字抄录出来，每行一个。"
    "不要写任何描述、不要解释、不要归类、不要说明它们之间的关系。"
    "位号一般是几个大写字母后面跟数字，有的还带管径或规格后缀。"
    "只抄你确实看清的那几个字符；看不清的宁可漏掉，"
    "绝对不要猜测、不要按编号顺序补全、不要写图上没有的内容。"
    "这块区域没有位号就输出：无"
)

# 1~6 个大写字母 + 数字，可带规格后缀或尾随字母。放宽到 1 个字母是因为舱柜编号
# （D800）和接口编号（A1、X1）都是单字母开头；放宽带来的噪声由一致性过滤兜底。
_TAG_RE = re.compile(r"\b[A-Z]{1,6}[-\s]?\d{1,6}(?:-[0-9ØΦ][\d.X×]*)?[A-Z]{0,2}\b")


GIST_PROMPT = (
    "这是船舶手册里的一张系统图。请用一到两句话说明：这是什么系统的图，图上有哪几类主要设备。"
    "不要列举位号编号，不要描述元件之间的连接关系，不要描述元件在图中的位置。"
    "看不清的一律不写。总共不超过 60 字。"
)
GIST_MAX_TOKENS = 150  # 输出必须短：长度本身就是防止重复循环的护栏


def summarize_diagram(api_key: str, image: Image.Image) -> str:
    """用一两句话说明这张图是什么系统，供语义检索用；失败返回空串。

    这是视觉模型在图纸上唯一可靠的用法。判器件类型（实测 7/12 且自信地错）和说连接关系
    （把 GVA11 说成 GVA21）都不能用，但"这是什么系统"是要点题，不需要看清小标注就能答。
    四页各跑两遍实测：内容全部正确、长度 28~48 字、两遍高度一致。

    输出限死在一两句话，既是为了让它进检索片段时不喧宾夺主，也因为退化的本质是
    "做不到又停不下来"——输出短，就没有失控的空间。原先那版 prompt 要求描述
    "关键组件、标注和连接关系"，在同一页上产出了 3100 字散文和 162 个编造位号。
    """
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=VL_MODEL,
            max_tokens=GIST_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GIST_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(buffer.getvalue()).decode()
                            },
                        },
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def is_diagram_page(doc: fitz.Document, page_index: int) -> bool:
    """判断是不是图纸页：嵌了大位图，且文字层稀疏（说明标注都在图里）。"""
    page = doc[page_index]
    images = page.get_images(full=True)
    if not images:
        return False
    if len(page.get_text().strip()) >= MAX_DIAGRAM_TEXT:
        return False
    largest = max(
        (doc.extract_image(img[0]) for img in images),
        key=lambda info: info["width"] * info["height"],
    )
    return largest["width"] * largest["height"] >= MIN_DIAGRAM_PIXELS


def load_diagram_image(doc: fitz.Document, page_index: int) -> Image.Image | None:
    """按嵌入图的原生分辨率渲染整页，返回图像。

    不能直接用 extract_image 取嵌入位图的原始字节：PDF 里的图像是带变换矩阵放置的，
    有些页面存的是翻转或旋转过的位图，靠页面矩阵摆正。直接取原始字节会拿到一张上下
    颠倒又镜像的图（Machinery p11 就是），模型读不出任何位号。

    所以走渲染这条路，让 PyMuPDF 应用变换；再按嵌入图的像素宽度反推缩放倍数，
    保证分辨率不低于原图——整页按固定 2.5 倍渲染会把 3902 像素宽的图降到 2977，
    位号本来就小，这一降就更难认了。
    """
    page = doc[page_index]
    images = page.get_images(full=True)
    if not images:
        return None
    largest = max(
        (doc.extract_image(img[0]) for img in images),
        key=lambda info: info["width"] * info["height"],
    )

    page_width = page.rect.width or 1
    zoom = min(MAX_RENDER_ZOOM, max(MIN_RENDER_ZOOM, largest["width"] / page_width))
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def _tiles(image: Image.Image, grid: int = TILE_GRID, overlap: float = TILE_OVERLAP) -> list[bytes]:
    width, height = image.size
    out = []
    for row in range(grid):
        for col in range(grid):
            x0 = max(0, int(col * width / grid - overlap * width))
            x1 = min(width, int((col + 1) * width / grid + overlap * width))
            y0 = max(0, int(row * height / grid - overlap * height))
            y1 = min(height, int((row + 1) * height / grid + overlap * height))
            buf = io.BytesIO()
            image.crop((x0, y0, x1, y1)).save(buf, "PNG")
            out.append(buf.getvalue())
    return out


def _read_tile(client: OpenAI, png: bytes) -> set[str]:
    b64 = base64.b64encode(png).decode()
    response = client.chat.completions.create(
        model=VL_MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TAG_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    )
    text = response.choices[0].message.content or ""
    return {m.replace(" ", "").upper() for m in _TAG_RE.findall(text.upper())}


# 图纸上一些"看着像位号"的东西其实不是：NOTE2 来自注释编号，DN100 是公称通径，
# SUS316L 是材质牌号，CLASS3 是管路等级。它们会稳定地被抄出来，一致性过滤拦不住，
# 只能按前缀显式排除。
_NOT_A_TAG = re.compile(r"^(NOTE|DN|SUS|CLASS|PAGE|FIG|NO|REV|SCALE|TYPE|MAX|MIN)\d", re.I)


def normalize_tags(tags: set[str]) -> set[str]:
    """归一化：修掉直径符号的误识，并合并被截断的重复项。

    三件事：排除掉 NOTE2、DN100 这类看着像位号的非位号；把管路规格里被读成 0 的
    直径符号还原成 Ø（图上印的是 Ø88.9X3.05，模型常抄成 088.9X3.05）；
    同一个位号常会同时抄出完整版和截断版，短的是长的前缀时只保留长的。
    """
    fixed = {re.sub(r"-0(?=\d)", "-Ø", tag) for tag in tags if not _NOT_A_TAG.match(tag)}
    result = set()
    for tag in sorted(fixed, key=len, reverse=True):
        if not any(longer.startswith(tag) for longer in result):
            result.add(tag)
    return result


def extract_page_tags(
    api_key: str,
    image: Image.Image,
    runs: int = RUNS,
    grid: int = TILE_GRID,
    max_workers: int = 5,
    min_votes: int = MIN_VOTES,
) -> tuple[set[str], set[str]]:
    """抽取一页图纸上的位号。

    返回 (稳定位号, 不稳定位号)。稳定 = 至少 min_votes 遍读到，可以入库；
    不稳定 = 票数不够，多半是误识或编造，留给调用方决定要不要人工看。

    用多数票而不是全票：模型跑间方差大，要求全票时一次差的输出就能把真位号否决掉。
    """
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    tiles = _tiles(image, grid=grid)

    def one_run() -> set[str]:
        found: set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for tags in pool.map(lambda png: _read_tile(client, png), tiles):
                found |= tags
        return normalize_tags(found)

    results = [one_run() for _ in range(runs)]
    if not results:
        return set(), set()
    votes: Counter[str] = Counter()
    for found in results:
        votes.update(found)
    threshold = min(min_votes, runs)
    stable = {tag for tag, count in votes.items() if count >= threshold}
    unstable = set(votes) - stable
    return stable, unstable
