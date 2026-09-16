"""MinerU 解析结果的转换逻辑测试。

不调用任何接口，测的是把 MinerU 的输出改造成本项目格式的那两步：HTML 表格转 Markdown，
以及行内 LaTeX 还原。这两步都不能出错——表格格式错了 chunker 的表格保护会整个失效，
LaTeX 没还原则用户输 235m³ 时 BM25 匹配不上。

样例取自真实的解析结果。
"""

from rag.mineru import _latex_to_plain, _table_to_markdown, blocks_to_pages

REAL_TABLE = (
    "<table>"
    "<tr><td>Equipment</td><td>Manufacturer</td><td>Capacity</td></tr>"
    "<tr><td>ME LO Pumps</td><td>Shinko</td><td> $235m^{3}/hour$  at 0.6.8MPa</td></tr>"
    "<tr><td>Location</td><td colspan=\"2\">Engine room floor</td></tr>"
    "</table>"
)


def test_table_becomes_markdown_with_header_separator():
    """chunker 靠行首的 | 识别表格，靠分隔行判断表头，两者都必须有。"""
    md = _table_to_markdown(REAL_TABLE)
    lines = md.split("\n")
    assert lines[0].startswith("| Equipment |")
    assert set(lines[1]) <= set("|-")
    assert all(line.startswith("|") for line in lines)


def test_table_preserves_cell_values_verbatim():
    md = _table_to_markdown(REAL_TABLE)
    assert "Shinko" in md
    assert "0.6.8MPa" in md, "原文自带的排印错误也要原样保留，不能擅自修正"


def test_table_expands_colspan_so_rows_line_up():
    """colspan 不补齐的话各行列数不一致，Markdown 表格会错位。"""
    rows = [line for line in _table_to_markdown(REAL_TABLE).split("\n") if line.startswith("|")]
    widths = {line.count("|") for line in rows}
    assert len(widths) == 1, f"各行列数不一致：{widths}"


def test_table_escapes_pipes_inside_cells():
    md = _table_to_markdown("<table><tr><td>a|b</td><td>c</td></tr></table>")
    assert "a\\|b" in md


def test_empty_table_yields_nothing():
    assert _table_to_markdown("<table></table>") == ""


def test_latex_superscripts_become_unicode():
    assert _latex_to_plain("流量 $235m^{3}/hour$ 稳定") == "流量 235m³/hour 稳定"
    assert "²" in _latex_to_plain("$12m^{2}$")


def test_latex_conversion_leaves_ordinary_text_alone():
    plain = "压力低于 0.4MPa 时启动备用泵"
    assert _latex_to_plain(plain) == plain


def test_latex_conversion_does_not_eat_currency_like_text():
    """没有配对的 $ 不该被当成公式处理。"""
    assert _latex_to_plain("成本约 $500 左右") == "成本约 $500 左右"


def make_blocks() -> list[dict]:
    return [
        {"type": "text", "text": "PRIME PURITY", "text_level": "1", "page_idx": "0"},
        {"type": "text", "text": "正文第一段", "page_idx": "0"},
        {"type": "header", "text": "页眉不要", "page_idx": "0"},
        {"type": "footer", "text": "页脚不要", "page_idx": "0"},
        {"type": "page_number", "text": "14", "page_idx": "0"},
        {"type": "table", "table_body": REAL_TABLE, "table_caption": ["设备规格"], "page_idx": "1"},
        {"type": "text", "text": "流量 $29m^{3}/hour$", "page_idx": "1"},
    ]


def test_blocks_are_grouped_by_page():
    pages = blocks_to_pages(make_blocks(), "手册.pdf", lambda p: f"手册.pdf_p{p}.png")
    assert [p.page for p in pages] == [1, 2], "page_idx 从 0 起，页码要加一"
    assert pages[0].source == "手册.pdf"
    assert pages[0].image_path == "手册.pdf_p1.png"


def test_headers_and_footers_are_dropped():
    text = blocks_to_pages(make_blocks(), "手册.pdf", lambda p: None)[0].text
    assert "页眉不要" not in text and "页脚不要" not in text
    assert "正文第一段" in text


def test_heading_levels_become_markdown_headings():
    text = blocks_to_pages(make_blocks(), "手册.pdf", lambda p: None)[0].text
    assert text.startswith("# PRIME PURITY")


def test_table_and_latex_are_converted_inside_pages():
    page = blocks_to_pages(make_blocks(), "手册.pdf", lambda p: None)[1]
    assert "设备规格" in page.text
    assert "| Equipment |" in page.text
    assert "29m³/hour" in page.text


def test_blocks_without_page_index_are_skipped_not_crashing():
    pages = blocks_to_pages([{"type": "text", "text": "x", "page_idx": None}], "手册.pdf", lambda p: None)
    assert pages == []


def test_empty_input():
    assert blocks_to_pages([], "手册.pdf", lambda p: None) == []
