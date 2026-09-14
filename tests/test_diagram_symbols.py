"""符号检测里纯函数部分的回归测试（不依赖 PDF、不调用模型）。"""

import numpy as np
import pytest
from PIL import Image

from rag.diagram_symbols import (
    detect_symbols,
    snap_to_vocabulary,
    summarize,
    to_black_mask,
)


def make_image(width=200, height=120):
    """白底画一个黑色方块，再叠一条彩色高亮压在方块上。"""
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    array[40:70, 40:70] = 0                      # 黑色符号
    array[50:60, 30:120] = (230, 170, 30)        # 彩色管线高亮，盖住符号一部分
    return Image.fromarray(array)


def test_black_mask_keeps_line_work():
    mask = to_black_mask(make_image())
    assert mask[45, 45] == 255  # 符号上没被高亮盖到的部分保留


def test_black_mask_drops_colour_highlight():
    """管线高亮会填进符号内部，不去掉的话同型阀门会因填色不同而失配。"""
    mask = to_black_mask(make_image())
    assert mask[55, 100] == 0   # 纯高亮区域应被滤掉
    assert mask[10, 10] == 0    # 白底也应被滤掉


def test_detect_finds_the_template_in_the_page():
    image = make_image()
    mask = to_black_mask(image)
    template = mask[40:70, 40:70]
    hits = detect_symbols(mask, {"方块": template}, threshold=0.8)
    assert len(hits) == 1
    assert hits[0].kind == "方块"
    cx, cy = hits[0].center
    assert abs(cx - 55) <= 3 and abs(cy - 55) <= 3


def test_detect_returns_nothing_when_template_absent():
    mask = to_black_mask(make_image())
    absent = np.zeros((20, 20), dtype=np.uint8)
    absent[5:15, 5:15] = 255
    hits = detect_symbols(mask, {"不存在": absent}, threshold=0.95)
    assert hits == []


def test_summarize_counts_by_kind():
    from rag.diagram_symbols import SymbolHit

    hits = [
        SymbolHit("阀", 0, 0.9, (0, 0, 10, 10)),
        SymbolHit("阀", 90, 0.8, (20, 0, 30, 10)),
        SymbolHit("泵", 0, 0.7, (40, 0, 50, 10)),
    ]
    assert summarize(hits) == {"阀": 2, "泵": 1}


def test_snap_fixes_a_single_character_misread():
    """实测把 GFV22 读成了 GEV22，用整页抽出的位号表可以修回来。"""
    assert snap_to_vocabulary("GEV22", {"GFV21", "GFV22", "GFV19"}) == "GFV22"


def test_snap_keeps_an_exact_match():
    assert snap_to_vocabulary("GFV21", {"GFV21", "GFV22"}) == "GFV21"


def test_snap_leaves_unknown_tags_alone():
    assert snap_to_vocabulary("ZZZ999", {"GFV21", "GFV22"}) == "ZZZ999"


def test_snap_refuses_when_two_candidates_are_equally_close():
    """GFV21 和 GFV23 跟 GFV2X 距离相同，分不清就别改。"""
    assert snap_to_vocabulary("GFV2X", {"GFV21", "GFV23"}) == "GFV2X"


@pytest.mark.parametrize("tag", ["", "无"])
def test_snap_handles_empty_or_missing(tag):
    assert snap_to_vocabulary(tag, {"GFV21"}) == tag
