"""把图纸页的抽取结果转成可检索的片段。

图纸页的文字层只有页眉和图名，正文转录也抓不到图上的位号——这类页面在原来的索引里
几乎没有检索信号，用户输一个阀件位号根本找不到对应的图。

这里把三样抽取结果拼成一段结构化文字，作为普通 chunk 进正常的向量和 BM25 索引，
不需要给检索链路加任何分支：

- 图名与概述：图名来自文字层，概述由视觉模型用一两句话给出（见 diagram_tags.summarize_diagram）
- 位号（rag/diagram_tags）：切片识别 + 多遍投票，可靠
- 介质与管线走向（rag/diagram_pipes）：PDF 矢量坐标，精确

概述是这里唯一由模型生成的自然语言，它补上位号和介质给不了的语义信号——用户问
"燃气阀组的通风怎么走"时，光靠位号列表匹配不上。但它被严格限制在一两句话、
只说"这是什么系统、有哪几类设备"：实测这种要点题模型答得准（四页两遍全部正确、
28~48 字），而让它判器件类型（7/12 且自信地错）或说连接关系（把 GVA11 说成
GVA21）都不可靠，所以那两样一概不写。

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
    gist: str = ""
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

    if page.gist:
        lines.append(page.gist)
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
                    gist=entry.get("gist", ""),
                    tags=entry.get("tags", []),
                    media=entry.get("media", []),
                    symbols=entry.get("symbols", {}),
                    image_path=entry.get("image_path"),
                )
            )
    pages.sort(key=lambda p: (p.source, p.page))
    return pages
