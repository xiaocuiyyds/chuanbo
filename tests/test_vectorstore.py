"""向量库的回归测试：删除文档时的索引一致性，以及邻接扩展。

这里不调用任何外部服务，向量用随机数构造——测的是索引维护的正确性，不是检索效果。
最关键的不变量：chunks 列表和 FAISS 索引里的向量必须始终一一对应。一旦错位，
检索会返回张冠李戴的内容，而且不会报错。
"""

import numpy as np
import pytest

from rag.chunker import Chunk
from rag.llm import build_context
from rag.vectorstore import VectorStore

DIM = 8


def make_store(layout: list[tuple[str, int, int]]) -> VectorStore:
    """layout 为 [(文档名, 页码, 该页片段数), ...]，按给定顺序建库。"""
    chunks = []
    for source, page, count in layout:
        for i in range(count):
            chunks.append(Chunk(source=source, page=page, text=f"{source}-p{page}-{i}"))
    rng = np.random.default_rng(0)
    vectors = rng.random((len(chunks), DIM)).astype("float32")
    store = VectorStore(dim=DIM)
    store.add(vectors, chunks)
    return store


def assert_consistent(store: VectorStore) -> None:
    assert store.index.ntotal == len(store.chunks), "向量数与片段数必须一致"
    assert store.bm25 is not None or store.size == 0, "非空时 BM25 必须已重建"


def test_remove_source_keeps_index_consistent():
    store = make_store([("A.pdf", 1, 3), ("A.pdf", 2, 2), ("B.pdf", 1, 4)])
    assert_consistent(store)

    removed = store.remove_source("A.pdf")
    assert removed == 5
    assert store.size == 4
    assert store.sources == ["B.pdf"]
    assert_consistent(store)


def test_remove_source_preserves_remaining_vectors():
    """删除后剩下的片段，向量必须还是原来那一条，不能跟着下标一起错位。"""
    store = make_store([("A.pdf", 1, 2), ("B.pdf", 1, 2), ("C.pdf", 1, 2)])
    keep = {c.text: store.index.reconstruct(i) for i, c in enumerate(store.chunks) if c.source != "B.pdf"}

    store.remove_source("B.pdf")
    for i, chunk in enumerate(store.chunks):
        np.testing.assert_allclose(store.index.reconstruct(i), keep[chunk.text], rtol=1e-6)


def test_remove_unknown_source_is_a_noop():
    store = make_store([("A.pdf", 1, 3)])
    assert store.remove_source("不存在.pdf") == 0
    assert store.size == 3
    assert_consistent(store)


def test_remove_all_sources_leaves_empty_store():
    store = make_store([("A.pdf", 1, 2)])
    assert store.remove_source("A.pdf") == 2
    assert store.size == 0
    assert store.index.ntotal == 0
    assert store.search(np.ones(DIM, dtype="float32"), "任何问题", k=5) == []


def test_expand_with_neighbors_pulls_adjacent_chunks():
    store = make_store([("A.pdf", 1, 5)])
    hit = store.chunks[2]
    expanded = store.expand_with_neighbors([(hit, 1.0)], window=1)
    assert [c.text for c in expanded] == ["A.pdf-p1-1", "A.pdf-p1-2", "A.pdf-p1-3"]


def test_expand_with_neighbors_does_not_cross_documents():
    """相邻下标可能属于另一份手册，扩展时绝不能跨文档带进来。"""
    store = make_store([("A.pdf", 1, 2), ("B.pdf", 1, 2)])
    hit = store.chunks[1]  # A.pdf 的最后一个片段，下一个就是 B.pdf
    expanded = store.expand_with_neighbors([(hit, 1.0)], window=1)
    assert all(c.source == "A.pdf" for c in expanded)


def test_expand_with_neighbors_deduplicates_and_preserves_order():
    store = make_store([("A.pdf", 1, 6)])
    hits = [(store.chunks[1], 1.0), (store.chunks[2], 0.9)]  # 两个命中的邻域重叠
    expanded = store.expand_with_neighbors(hits, window=1)
    assert [c.text for c in expanded] == [f"A.pdf-p1-{i}" for i in range(4)]


def test_expand_at_list_boundaries():
    store = make_store([("A.pdf", 1, 3)])
    assert len(store.expand_with_neighbors([(store.chunks[0], 1.0)], window=1)) == 2
    assert len(store.expand_with_neighbors([(store.chunks[-1], 1.0)], window=1)) == 2


@pytest.mark.parametrize("window", [0, 1, 2])
def test_expand_window_sizes(window):
    store = make_store([("A.pdf", 1, 9)])
    expanded = store.expand_with_neighbors([(store.chunks[4], 1.0)], window=window)
    assert len(expanded) == 2 * window + 1


def test_build_context_merges_consecutive_chunks_of_same_page():
    """同页连续片段合并成一块，被切开的步骤才能连起来读，来源标题也不必重复。"""
    chunks = [
        Chunk(source="A.pdf", page=1, text="步骤 1-4"),
        Chunk(source="A.pdf", page=1, text="步骤 5-8"),
        Chunk(source="A.pdf", page=2, text="下一页内容"),
    ]
    context = build_context(chunks)
    assert context.count("[来源: A.pdf 第1页]") == 1
    assert context.count("[来源: A.pdf 第2页]") == 1
    assert context.index("步骤 1-4") < context.index("步骤 5-8")


def test_build_context_empty():
    assert build_context([]) == ""


def make_store_with_text(texts: list[tuple[str, int, str]]) -> VectorStore:
    chunks = [Chunk(source=s, page=p, text=t) for s, p, t in texts]
    rng = np.random.default_rng(1)
    store = VectorStore(dim=DIM)
    store.add(rng.random((len(chunks), DIM)).astype("float32"), chunks)
    return store


def test_exact_token_match_finds_a_tag_buried_in_a_long_question():
    """位号对向量检索是无意义的随机串，RRF 会把只有 BM25 命中的候选压下去。
    这条通道就是为了兜住它——实测 GFV21 曾因此完全检索不到。"""
    store = make_store_with_text([
        ("图纸.pdf", 112, "【图纸】燃气供应系统\nGFV19 GFV20 GFV21 GFV22"),
        ("正文.pdf", 31, "主机燃气系统概述，涉及 GAV 与 GVU 的切换逻辑。"),
    ])
    hits = store.exact_token_matches("GFV21 这个阀在什么位置，起什么作用", limit=1)
    assert hits and hits[0].page == 112


def test_exact_token_match_ignores_plain_language_queries():
    """普通中文问句不该触发精确匹配，否则会白占一个名额。"""
    store = make_store_with_text([("正文.pdf", 1, "应急消防泵启动前要先启真空泵。")])
    assert store.exact_token_matches("应急消防泵怎么启动") == []


def test_exact_token_match_returns_nothing_when_tag_absent():
    store = make_store_with_text([("正文.pdf", 1, "主机滑油系统说明。")])
    assert store.exact_token_matches("XYZ999 在哪里") == []


def test_exact_token_match_respects_limit():
    store = make_store_with_text([
        ("a.pdf", 1, "GFV21 出现在这里"),
        ("b.pdf", 2, "GFV21 也出现在这里"),
    ])
    assert len(store.exact_token_matches("GFV21", limit=1)) == 1


def test_exact_token_match_on_empty_store():
    assert VectorStore(dim=DIM).exact_token_matches("GFV21") == []
