"""用模板匹配在图纸上检测器件符号。

为什么不用视觉模型：让它判读器件类型实测不可靠——装在风机出口立管上的阀门
GFV21/GFV22 会被判成"泵/风机"，因为紧挨着风机符号。而 P&ID 的符号是标准化的、
从 CAD 符号库画出来的，同一张图里同一种器件的符号像素级一致，这正是模板匹配的场合。

实测：拿 GFV21 的领结符号当模板，在整页上以 0.6 阈值匹配，精确命中 GFV21 和
GFV22 两个，零误报；但漏掉 GFV19/GFV20——它们是横置的，而且样式不同（两个三角
都是空心）。所以模板要按"类型 x 朝向"覆盖，单个模板只能召回同型同向的。

两个前置处理是必需的：

1. 只保留黑色线条。管线上的彩色高亮会盖在符号上（GFV21 的领结里填了黄色），
   直接在彩色图上匹配会因为填色不同而失配。
2. 按嵌入图原生分辨率渲染页面，而不是直接取嵌入图字节——部分页面存的位图是翻转的，
   靠页面矩阵摆正。这一点见 rag/diagram_tags.load_diagram_image。

注意连通域分析在这类图上没用：符号和管线画在一起，整张管网是一个连通域，
实测最大连通域占了整页 63% x 19%。
"""

import collections
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

BLACK_MAX = 120  # 三通道都低于这个值才算黑线
BLACK_MAX_CHROMA = 40  # 且通道差要小，排除彩色高亮和灰度底纹
MATCH_THRESHOLD = 0.6  # 模板匹配相关度阈值
NMS_OVERLAP = 0.3
ROTATIONS = (0, 90, 180, 270)


@dataclass
class SymbolHit:
    kind: str  # 符号类型名，来自模板库
    rotation: int
    score: float
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1)，页面渲染图的像素坐标

    @property
    def center(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) // 2, (y0 + y1) // 2)


def to_black_mask(image: Image.Image) -> np.ndarray:
    """只保留黑色线条，去掉管线的彩色高亮和灰色底纹。

    符号本身是黑线画的，但管线高亮会把颜色填进符号内部，不去掉的话同一种阀门
    因为填色不同就匹配不上了。
    """
    array = np.asarray(image.convert("RGB")).astype(np.int16)
    brightest = array.max(axis=2)
    chroma = brightest - array.min(axis=2)
    mask = (brightest < BLACK_MAX) & (chroma < BLACK_MAX_CHROMA)
    return (mask.astype(np.uint8)) * 255


def _rotate(template: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return template
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(template, code[degrees])


def detect_symbols(
    page_mask: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = MATCH_THRESHOLD,
    rotations: tuple[int, ...] = ROTATIONS,
) -> list[SymbolHit]:
    """在黑线掩膜上匹配所有模板的所有朝向，跨模板统一做 NMS。

    跨模板一起 NMS 很关键：同一个符号常会被相近的几个模板同时命中（比如同一个阀门
    的 0 度和 180 度模板都匹配上），不统一抑制就会重复计数。
    """
    boxes: list[list[int]] = []
    scores: list[float] = []
    meta: list[tuple[str, int]] = []

    for kind, template in templates.items():
        for degrees in rotations:
            rotated = _rotate(template, degrees)
            h, w = rotated.shape
            if h > page_mask.shape[0] or w > page_mask.shape[1]:
                continue
            response = cv2.matchTemplate(page_mask, rotated, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(response >= threshold)
            for x, y in zip(xs, ys):
                boxes.append([int(x), int(y), int(w), int(h)])
                scores.append(float(response[y, x]))
                meta.append((kind, degrees))

    if not boxes:
        return []

    keep = cv2.dnn.NMSBoxes(boxes, scores, threshold, NMS_OVERLAP)
    hits = []
    for i in np.array(keep).flatten():
        x, y, w, h = boxes[int(i)]
        kind, degrees = meta[int(i)]
        hits.append(SymbolHit(kind=kind, rotation=degrees, score=scores[int(i)], box=(x, y, x + w, y + h)))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits


def crop_template(image: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    """从页面上框一块作为模板，返回黑线掩膜。模板库就是这么建起来的。"""
    x0, y0, x1, y1 = box
    return to_black_mask(image.crop((x0, y0, x1, y1)))


def summarize(hits: list[SymbolHit]) -> dict[str, int]:
    return dict(collections.Counter(hit.kind for hit in hits))


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_to_vocabulary(tag: str, vocabulary: set[str], max_edits: int = 1) -> str:
    """把读到的位号吸附到已知位号表里最接近的一个，改不动就原样返回。

    符号旁边的标签是小范围裁图读出来的，偶尔会错一两个字符——实测把 GFV22 读成了
    GEV22（F 认成 E）。而整页切片抽取（rag/diagram_tags）已经给出了这一页的位号表，
    拿它做吸附就能修掉这类单字符误读。

    只在唯一最近邻时才吸附：如果有多个候选都是同样的距离，说明分不清，宁可保留原样。
    """
    if not tag or tag in vocabulary:
        return tag
    scored = [(_edit_distance(tag, known), known) for known in vocabulary]
    scored = [(dist, known) for dist, known in scored if dist <= max_edits]
    if not scored:
        return tag
    best = min(scored)[0]
    closest = [known for dist, known in scored if dist == best]
    return closest[0] if len(closest) == 1 else tag
