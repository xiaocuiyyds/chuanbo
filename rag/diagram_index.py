"""把图纸页的抽取结果转成可检索的片段。

图纸页的文字层只有页眉和图名，正文转录也抓不到图上的位号——这类页面在原来的索引里
几乎没有检索信号，用户输一个阀件位号根本找不到对应的图。

这里把三样抽取结果拼成一段结构化文字，作为普通 chunk 进正常的向量和 BM25 索引，
不需要给检索链路加任何分支：

- 位号（rag/diagram_tags）：切片识别 + 多遍取交集，可靠
- 介质与管线走向（rag/diagram_pipes）：PDF 矢量坐标，精确
- 器件符号（rag/diagram_symbols）：模板匹配，精确率高但召回有限

措辞上刻意保守。位号识别有截断和误读的残留（GVA80-Ø34X3 会被抄成 GV80），
所以片段里写明这是"图上识别出的位号"，让回答模型和用户都清楚要回看原图确认，
而不是把它当成手册正文里的权威数据。
"""

from dataclasses import dataclass, field


@dataclass
class DiagramPage:
    source: str
    page: int
    title: str = ""
    tags: list[str] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    symbols: dict[str, int] = field(default_factory=dict)
    image_path: str | None = None


def to_chunk_text(page: DiagramPage) -> str:
    """拼成一段自然语言片段，让向量检索和 BM25 都能命中。

    位号单独成行且原样保留，BM25 的分词会把它们切成独立 token，
    用户直接输一个位号就能命中这一页。
    """
    lines = [f"【图纸】{page.title}" if page.title else "【图纸】"]
    lines.append(f"出处：{page.source} 第{page.page}页")

    if page.media:
        lines.append(f"图中标注的介质管路：{'、'.join(page.media)}")
    if page.symbols:
        detail = "、".join(f"{kind} {count} 个" for kind, count in sorted(page.symbols.items()))
        lines.append(f"识别到的器件符号：{detail}")
    if page.tags:
        lines.append(f"图上识别出的设备位号（共 {len(page.tags)} 个，以原图为准）：")
        lines.append(" ".join(page.tags))

    return "\n".join(lines)


def build_pages(raw: dict) -> list[DiagramPage]:
    """把 data/diagram_index.json 的内容还原成 DiagramPage 列表。"""
    pages = []
    for source, by_page in raw.items():
        for page_number, entry in by_page.items():
            pages.append(
                DiagramPage(
                    source=source,
                    page=int(page_number),
                    title=entry.get("title", ""),
                    tags=entry.get("tags", []),
                    media=entry.get("media", []),
                    symbols=entry.get("symbols", {}),
                    image_path=entry.get("image_path"),
                )
            )
    pages.sort(key=lambda p: (p.source, p.page))
    return pages
