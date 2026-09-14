"""矢量管线提取的回归测试（不依赖 PDF、不调用任何模型）。

用 PyMuPDF 现造一个带图例和彩色管线的页面来测——图例匹配那几个阈值是对着真实
手册调出来的，直接拿真实 PDF 做测试会让用例依赖那几个大文件。
"""

import fitz
import pytest

from rag.diagram_pipes import (
    JOIN_TOLERANCE,
    _connected_runs,
    extract_legend,
    extract_pipe_runs,
    has_vector_pipes,
    summarize_page,
)

ORANGE = (0.98, 0.65, 0.1)
BLUE = (0.2, 0.45, 0.73)


def make_page(with_legend: bool = True, watermark: bool = False) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    if with_legend:
        # 图例：一截短横线 + 紧挨右侧的介质名
        page.draw_line(fitz.Point(100, 300), fitz.Point(130, 300), color=ORANGE, width=2)
        page.insert_text(fitz.Point(140, 303), "Fuel Gas", fontsize=8)
        page.draw_line(fitz.Point(100, 320), fitz.Point(130, 320), color=BLUE, width=2)
        page.insert_text(fitz.Point(140, 323), "Air", fontsize=8)
    if watermark:
        # 真实手册每页都有这条水印，曾把里面的 FOR 当成介质名配给了 Air
        page.insert_text(fitz.Point(140, 323), "DRAFT MANUAL FOR REVIEW & COMMENT", fontsize=8)
    # 页面主体的管线：一条橙色折线（两段相连）+ 一条独立的蓝色线
    page.draw_line(fitz.Point(50, 50), fitz.Point(200, 50), color=ORANGE, width=2)
    page.draw_line(fitz.Point(200, 50), fitz.Point(200, 150), color=ORANGE, width=2)
    page.draw_line(fitz.Point(400, 50), fitz.Point(500, 50), color=BLUE, width=2)
    return page


def test_legend_maps_colours_to_media():
    legend = extract_legend(make_page())
    assert legend[tuple(round(c, 3) for c in ORANGE)] == "Fuel Gas"
    assert legend[tuple(round(c, 3) for c in BLUE)] == "Air"


def test_legend_keeps_multi_word_media_names():
    """介质名多是两个词，按单词匹配只会取到第一个。"""
    assert "Fuel Gas" in extract_legend(make_page()).values()


def test_watermark_is_not_mistaken_for_a_legend_label():
    """每页都有的 DRAFT MANUAL FOR REVIEW 水印里的 FOR 曾被当成介质名。"""
    assert "FOR" not in extract_legend(make_page(watermark=True)).values()


def test_page_without_legend_yields_empty_mapping():
    assert extract_legend(make_page(with_legend=False)) == {}


def test_connected_segments_are_grouped():
    """端点相接的线段属于同一条回路。"""
    joined = [((0, 0), (10, 0)), ((10, 0), (10, 10))]
    assert len(_connected_runs(joined)) == 1


def test_disconnected_segments_stay_separate():
    apart = [((0, 0), (10, 0)), ((100, 100), (110, 100))]
    assert len(_connected_runs(apart)) == 2


def test_join_tolerance_bridges_small_gaps():
    """端点不完全重合时按容差判连通，图纸里线段端点常差零点几个点。"""
    nearly = [((0, 0), (10, 0)), ((10 + JOIN_TOLERANCE / 2, 0), (10, 10))]
    assert len(_connected_runs(nearly)) == 1


def test_pipe_runs_carry_medium_and_geometry():
    runs = [r for r in extract_pipe_runs(make_page()) if r.medium == "Fuel Gas"]
    body = max(runs, key=lambda r: len(r.segments))
    assert len(body.segments) == 2  # 折线的两段被聚在一起
    x0, y0, x1, y1 = body.bbox
    assert (x0, y0, x1, y1) == (50, 50, 200, 150)


def test_summarize_page_groups_by_medium():
    summary = summarize_page(make_page())
    assert set(summary["legend"]) == {"Fuel Gas", "Air"}
    assert summary["media"]["Fuel Gas"]["segments"] >= 2
    assert summary["media"]["Air"]["segments"] >= 1


def test_has_vector_pipes_rejects_a_bare_page():
    doc = fitz.open()
    assert has_vector_pipes(doc.new_page()) is False


@pytest.mark.parametrize("with_legend", [True, False])
def test_extraction_never_raises_on_odd_pages(with_legend):
    extract_pipe_runs(make_page(with_legend=with_legend))
