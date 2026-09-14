"""从图纸页里提取彩色管线走向，以及图例的颜色到介质的对应。

手册里的系统图按介质给管线上色，页面右下角的图例给出颜色和介质名的对照
（Fuel Gas / Air / Nitrogen / Vent / Exhaust / Drain 之类）。关键在于这些彩色线条
和图例色块都是 PDF 里的矢量对象，不在嵌入的位图里——也就是说管线走向可以按坐标
精确读出来，不需要任何识别，也就不存在幻觉。

这一点是试过让视觉模型"看图说连接关系"之后才发现的。模型会把相邻编号的管路
GVA11 和 GVA21 弄混，而这些坐标本来就明明白白躺在 PDF 里。

覆盖范围：全库 87 张图纸页里 68 页（78%）含可提取的彩色矢量管线，
Machinery 手册 73 页图纸中有 62 页、共 6609 条线段。

能拿到什么、拿不到什么，要分清楚：

- 能拿到：每条彩色管线的介质、精确端点坐标、以及按端点邻接聚出来的回路。
  适合回答"燃气系统在这张图上走哪些路径"。
- 拿不到：黑色线条画的管路细节和阀件符号都在位图里，矢量里没有。
  聚出来的回路也是碎的（p112 的 21 条透气管只能拼成 10 段），
  所以它说不了"GFV19 接在 GFV20 的上游"这种器件级拓扑。
"""

import collections
import re
from dataclasses import dataclass, field

import fitz

LEGEND_SWATCH_MAX_WIDTH = 50  # 图例色块是很短的一截横线
LEGEND_SWATCH_MAX_HEIGHT = 1.5  # 且几乎没有高度
LEGEND_LABEL_MAX_DY = 6  # 图例文字和色块的垂直偏差容忍度
LEGEND_LABEL_MAX_DX = 60  # 图例文字必须紧挨着色块右侧，太远的不算
WATERMARK_RE = re.compile(r"DRAFT MANUAL FOR REVIEW", re.I)
JOIN_TOLERANCE = 3.0  # 两条线段端点相距多少以内算相连（PDF 点）
MIN_VECTOR_ITEMS = 15  # 少于这么多彩色元素的页面，认为没有可用的矢量管线


@dataclass
class PipeRun:
    """一条连通的管线回路。"""

    medium: str  # 介质名，来自图例；图例里没有对应的填空串
    color: tuple[float, float, float]
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for seg in self.segments for p in seg]
        ys = [p[1] for seg in self.segments for p in seg]
        return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def _round_color(color) -> tuple[float, float, float]:
    return tuple(round(float(c), 3) for c in color)


def extract_legend(page: fitz.Page) -> dict[tuple[float, float, float], str]:
    """读出图例的 颜色 -> 介质名 对照。

    图例色块是一截很短的水平线，右边紧跟着介质名。色块是矢量对象、文字在文字层里，
    两者按纵向位置就近配对即可，不需要认图。

    按整行而不是按单个词匹配：介质名多是两个词（Fuel Gas、Fresh Water），
    按词匹配只会取到第一个；而且每页边缘都有一长条"DRAFT MANUAL FOR REVIEW"水印，
    按词匹配时它里面的 FOR 曾被当成介质名配给了 Air。
    """
    lines: list[tuple[float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text or WATERMARK_RE.search(text):
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append(((y0 + y1) / 2, x0, text))

    legend: dict[tuple[float, float, float], str] = {}
    for drawing in page.get_drawings():
        color = drawing.get("color")
        rect = drawing["rect"]
        if not color:
            continue
        if rect.width > LEGEND_SWATCH_MAX_WIDTH or abs(rect.height) > LEGEND_SWATCH_MAX_HEIGHT:
            continue
        mid_y = (rect.y0 + rect.y1) / 2
        candidates = [
            (line_x - rect.x1, text)
            for line_y, line_x, text in lines
            if 0 <= line_x - rect.x1 < LEGEND_LABEL_MAX_DX
            and abs(line_y - mid_y) < LEGEND_LABEL_MAX_DY
        ]
        if candidates:
            legend[_round_color(color)] = min(candidates)[1]
    return legend


def _segments_by_color(page: fitz.Page) -> dict[tuple[float, float, float], list]:
    by_color: dict[tuple, list] = collections.defaultdict(list)
    for drawing in page.get_drawings():
        color = drawing.get("color")
        if not color:
            continue
        key = _round_color(color)
        for item in drawing["items"]:
            if item[0] == "l":
                by_color[key].append(((item[1].x, item[1].y), (item[2].x, item[2].y)))
            elif item[0] == "re":
                r = item[1]
                corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                by_color[key].extend(zip(corners, corners[1:] + corners[:1]))
    return by_color


def _connected_runs(segments: list, tolerance: float = JOIN_TOLERANCE) -> list[list]:
    """按端点邻近把线段聚成连通回路（并查集）。"""
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if any(
                abs(pa[0] - pb[0]) < tolerance and abs(pa[1] - pb[1]) < tolerance
                for pa in segments[i]
                for pb in segments[j]
            ):
                union(i, j)

    groups: dict[int, list] = collections.defaultdict(list)
    for i in range(len(segments)):
        groups[find(i)].append(segments[i])
    return list(groups.values())


def extract_pipe_runs(page: fitz.Page) -> list[PipeRun]:
    """提取一页上的所有彩色管线回路，按段数从多到少排序。

    图例自身的色块也是彩色线段，会被聚成很短的回路；调用方若不需要可按段数过滤。
    """
    legend = extract_legend(page)
    runs: list[PipeRun] = []
    for color, segments in _segments_by_color(page).items():
        medium = legend.get(color, "")
        for group in _connected_runs(segments):
            runs.append(PipeRun(medium=medium, color=color, segments=group))
    runs.sort(key=lambda run: len(run.segments), reverse=True)
    return runs


def has_vector_pipes(page: fitz.Page) -> bool:
    count = sum(
        1
        for drawing in page.get_drawings()
        if drawing.get("color")
        for item in drawing["items"]
        if item[0] in ("l", "re")
    )
    return count >= MIN_VECTOR_ITEMS


def summarize_page(page: fitz.Page) -> dict:
    """汇总一页的管线信息，形态适合直接进检索索引。"""
    legend = extract_legend(page)
    runs = [run for run in extract_pipe_runs(page) if run.medium]
    by_medium: dict[str, dict] = {}
    for run in runs:
        entry = by_medium.setdefault(run.medium, {"runs": 0, "segments": 0})
        entry["runs"] += 1
        entry["segments"] += len(run.segments)
    return {
        "legend": {medium: list(color) for color, medium in legend.items()},
        "media": by_medium,
    }
