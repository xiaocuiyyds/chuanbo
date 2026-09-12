"""转录持久化的回归测试。

转录是整个流程里最贵的产物，读写必须可靠：一旦往返丢内容，重建出来的索引就是错的，
而且不会报错。重点覆盖文档名里的特殊字符（真实手册名里有空格、加号、括号和中文）
以及重新上传时的残留清理。
"""

import json

import pytest

from rag import transcripts
from rag.pdf_loader import PageText

TRICKY_NAME = "KONGSBERG AC600_MEB_FPP +(翻译结果).pdf"


def make_pages(source: str, count: int) -> list[PageText]:
    return [
        PageText(source=source, page=i, text=f"# 第{i}页\n\n内容 {i}", image_path=f"{source}_p{i}.png")
        for i in range(1, count + 1)
    ]


def test_round_trip_preserves_everything(tmp_path):
    pages = make_pages("手册.pdf", 3)
    transcripts.save_pages(tmp_path, "手册.pdf", pages)
    loaded = transcripts.load_pages(tmp_path, "手册.pdf")

    assert len(loaded) == 3
    for original, restored in zip(pages, loaded):
        assert restored.source == original.source
        assert restored.page == original.page
        assert restored.text == original.text
        assert restored.image_path == original.image_path


def test_special_characters_in_source_name(tmp_path):
    """文档名里的空格、加号、括号、中文都要能正确往返。"""
    transcripts.save_pages(tmp_path, TRICKY_NAME, make_pages(TRICKY_NAME, 2))
    assert transcripts.list_sources(tmp_path) == [TRICKY_NAME]
    assert [p.source for p in transcripts.load_pages(tmp_path, TRICKY_NAME)] == [TRICKY_NAME] * 2


def test_source_name_comes_from_manifest_not_directory(tmp_path):
    """目录名是经过替换的，不可逆，所以原始名必须从 manifest 读。"""
    source = "sub/dir/手册.pdf"
    transcripts.save_pages(tmp_path, source, make_pages(source, 1))
    assert transcripts.list_sources(tmp_path) == [source]


def test_resave_removes_stale_pages(tmp_path):
    """重新上传后页数变少时，旧的多余页必须清掉，否则重建索引会读回幽灵页。"""
    transcripts.save_pages(tmp_path, "手册.pdf", make_pages("手册.pdf", 5))
    transcripts.save_pages(tmp_path, "手册.pdf", make_pages("手册.pdf", 2))

    loaded = transcripts.load_pages(tmp_path, "手册.pdf")
    assert [p.page for p in loaded] == [1, 2]
    doc_dir = transcripts.transcripts_dir(tmp_path) / "手册.pdf"
    assert not (doc_dir / "p5.md").exists()


def test_load_missing_source_returns_empty(tmp_path):
    assert transcripts.load_pages(tmp_path, "没传过.pdf") == []


def test_list_sources_on_empty_dir(tmp_path):
    assert transcripts.list_sources(tmp_path) == []


def test_remove_source(tmp_path):
    transcripts.save_pages(tmp_path, "A.pdf", make_pages("A.pdf", 2))
    transcripts.save_pages(tmp_path, "B.pdf", make_pages("B.pdf", 2))

    assert transcripts.remove_source(tmp_path, "A.pdf") is True
    assert transcripts.list_sources(tmp_path) == ["B.pdf"]
    assert transcripts.remove_source(tmp_path, "A.pdf") is False


def test_load_all_pages_spans_documents(tmp_path):
    transcripts.save_pages(tmp_path, "A.pdf", make_pages("A.pdf", 2))
    transcripts.save_pages(tmp_path, "B.pdf", make_pages("B.pdf", 3))
    assert len(transcripts.load_all_pages(tmp_path)) == 5


def test_missing_page_file_does_not_break_the_rest(tmp_path):
    """单页文件损坏或丢失时，其余页面仍应能重建。"""
    transcripts.save_pages(tmp_path, "手册.pdf", make_pages("手册.pdf", 3))
    (transcripts.transcripts_dir(tmp_path) / "手册.pdf" / "p2.md").unlink()
    assert [p.page for p in transcripts.load_pages(tmp_path, "手册.pdf")] == [1, 3]


def test_corrupt_manifest_is_skipped(tmp_path):
    transcripts.save_pages(tmp_path, "好的.pdf", make_pages("好的.pdf", 1))
    bad = transcripts.transcripts_dir(tmp_path) / "坏的.pdf"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ 不是合法 json", encoding="utf-8")
    assert transcripts.list_sources(tmp_path) == ["好的.pdf"]


def test_pages_are_sorted_in_manifest(tmp_path):
    pages = make_pages("手册.pdf", 3)[::-1]  # 乱序传入
    transcripts.save_pages(tmp_path, "手册.pdf", pages)
    manifest = json.loads(
        (transcripts.transcripts_dir(tmp_path) / "手册.pdf" / "manifest.json").read_text(encoding="utf-8")
    )
    assert [e["page"] for e in manifest["pages"]] == [1, 2, 3]


@pytest.mark.parametrize("text", ["", "只有一行", "带\n换行\n\n和空行", "| 表格 | 行 |\n|---|---|"])
def test_various_page_contents_round_trip(tmp_path, text):
    page = PageText(source="手册.pdf", page=1, text=text, image_path=None)
    transcripts.save_pages(tmp_path, "手册.pdf", [page])
    assert transcripts.load_pages(tmp_path, "手册.pdf")[0].text == text
