"""切块逻辑的回归测试。

切块直接决定检索粒度，而且是纯函数、不依赖任何外部服务，最值得固化下来。
重点保证两件事：表格不会被从中间切断丢掉表头；超大块会被拆开而不是原样塞进索引。
"""

from rag.chunker import MAX_CHUNK_SIZE, TARGET_CHUNK_SIZE, chunk_pages
from rag.pdf_loader import PageText


def make_page(text: str, source: str = "手册.pdf", page: int = 1) -> PageText:
    return PageText(source=source, page=page, text=text, image_path=f"{source}_p{page}.png")


def test_chunk_carries_page_metadata():
    chunks = chunk_pages([make_page("# 标题\n\n正文内容。", source="A.pdf", page=7)])
    assert chunks
    assert all(c.source == "A.pdf" and c.page == 7 for c in chunks)
    assert all(c.image_path == "A.pdf_p7.png" for c in chunks)


def test_empty_page_yields_no_chunks():
    assert chunk_pages([make_page("   \n\n  \n")]) == []


def test_small_blocks_are_packed_together():
    """多个小段落应该被合并，而不是每段一个片段——否则检索粒度太碎。"""
    text = "\n\n".join(f"第{i}段落，内容不长。" for i in range(10))
    chunks = chunk_pages([make_page(text)])
    assert len(chunks) < 10
    assert all(len(c.text) <= TARGET_CHUNK_SIZE for c in chunks)


def test_oversized_table_is_split_and_every_part_keeps_the_header():
    """大表格按行拆开后，每一段都要重复表头，否则后半段单独看不懂、也检索不到。"""
    header = "| 报警代码 | 说明 | 处理 |\n|---|---|---|"
    rows = [f"| AL-{i:03d} | 第{i}号报警的详细说明文字 | 检查对应设备并复位 |" for i in range(120)]
    chunks = chunk_pages([make_page(header + "\n" + "\n".join(rows))])

    assert len(chunks) > 1, "这么大的表格应该被拆成多段"
    for chunk in chunks:
        assert "| 报警代码 | 说明 | 处理 |" in chunk.text
        assert len(chunk.text) <= MAX_CHUNK_SIZE

    # 拆开之后不能丢行
    for i in range(120):
        assert f"AL-{i:03d}" in "".join(c.text for c in chunks)


def test_table_smaller_than_max_is_not_split():
    header = "| 项目 | 数值 |\n|---|---|"
    rows = [f"| 参数{i} | {i} |" for i in range(5)]
    table = header + "\n" + "\n".join(rows)
    chunks = chunk_pages([make_page(table)])
    table_chunks = [c for c in chunks if "| 项目 | 数值 |" in c.text]
    assert len(table_chunks) == 1, "小表格不该被拆开"


def test_no_chunk_exceeds_max_size():
    """兜底：无论输入什么，都不该有片段超过硬上限。"""
    pages = [
        make_page("散文" * 5000, page=1),
        make_page("| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i} |" for i in range(500)), page=2),
        make_page("# 标题\n\n" + "短句。" * 400, page=3),
    ]
    for chunk in chunk_pages(pages):
        assert len(chunk.text) <= MAX_CHUNK_SIZE, f"第{chunk.page}页切出了 {len(chunk.text)} 字符的片段"


def test_pages_are_chunked_independently_and_in_order():
    pages = [make_page(f"第{p}页的内容。" * 60, page=p) for p in (1, 2, 3)]
    chunks = chunk_pages(pages)
    assert [c.page for c in chunks] == sorted(c.page for c in chunks)
    for chunk in chunks:
        assert f"第{chunk.page}页的内容" in chunk.text, "片段不应混入其他页的内容"


def test_retrieval_text_includes_context_but_text_does_not():
    """上下文只进检索文本，不能污染展示和喂给回答模型的原文。"""
    chunk = chunk_pages([make_page("阀门 LOV-101 的操作说明。")])[0]
    original = chunk.text
    assert chunk.retrieval_text == original

    chunk.context = "该片段属于主机滑油系统章节。"
    assert chunk.retrieval_text.startswith("该片段属于主机滑油系统章节。")
    assert original in chunk.retrieval_text
    assert chunk.text == original
