"""图纸片段生成的回归测试。

图纸页的正文转录抓不到图上的位号，这条片段是它们进入检索的唯一通道，
所以位号必须原样出现在文本里，BM25 才能按位号命中。
"""

from rag.diagram_index import DiagramPage, build_pages, to_chunk_text


def make_page(**kwargs) -> DiagramPage:
    defaults = dict(
        source="手册.pdf",
        page=112,
        title="Illustration 2.6.3b Fuel Gas Supply System (ii)",
        tags=["GFV19", "GFV21", "GVA11-Ø88.9X3.05"],
        media=["Fuel Gas", "Air", "Nitrogen"],
        symbols={"阀-领结实心顶": 4, "滤器-双X方框": 1},
    )
    defaults.update(kwargs)
    return DiagramPage(**defaults)


def test_chunk_text_contains_tags_verbatim():
    """用户会直接输位号来找图，位号必须原样出现，不能被改写或截断。"""
    text = to_chunk_text(make_page())
    for tag in ("GFV19", "GFV21", "GVA11-Ø88.9X3.05"):
        assert tag in text


def test_chunk_text_contains_title_and_provenance():
    text = to_chunk_text(make_page())
    assert "Fuel Gas Supply System" in text
    assert "手册.pdf" in text and "第112页" in text


def test_chunk_text_lists_media_and_symbols():
    text = to_chunk_text(make_page())
    assert "Fuel Gas" in text and "Nitrogen" in text
    assert "阀-领结实心顶" in text and "4 个" in text


def test_chunk_text_flags_tags_as_recognised_not_authoritative():
    """位号识别有截断和误读的残留，措辞必须让人知道要回看原图。"""
    text = to_chunk_text(make_page())
    assert "识别" in text and "以原图为准" in text


def test_chunk_text_omits_empty_sections():
    text = to_chunk_text(make_page(tags=[], media=[], symbols={}))
    assert "位号" not in text
    assert "介质" not in text
    assert "器件符号" not in text
    assert "手册.pdf" in text  # 出处始终保留


def test_chunk_text_survives_a_page_with_no_title():
    assert "手册.pdf" in to_chunk_text(make_page(title=""))


def test_build_pages_round_trip():
    raw = {
        "B.pdf": {"5": {"title": "图 B5", "tags": ["X1"], "media": ["Air"], "symbols": {}}},
        "A.pdf": {"12": {"title": "图 A12", "tags": [], "media": [], "symbols": {"阀": 2}}},
    }
    pages = build_pages(raw)
    assert [(p.source, p.page) for p in pages] == [("A.pdf", 12), ("B.pdf", 5)]  # 按文档、页码排序
    assert pages[0].symbols == {"阀": 2}
    assert pages[1].tags == ["X1"]


def test_build_pages_tolerates_missing_fields():
    pages = build_pages({"A.pdf": {"3": {}}})
    assert pages[0].tags == [] and pages[0].media == [] and pages[0].symbols == {}


def test_build_pages_on_empty_index():
    assert build_pages({}) == []
