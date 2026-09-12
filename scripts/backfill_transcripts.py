"""从已有的 chunks.pkl 反推出页面转录，补写到 data/transcripts/。

一次性脚本。转录持久化是后加的，在此之前入库的文档只剩切好的 chunk，原始的整页转录
已经丢了。好在同一页的 chunk 在列表里是按切分顺序连续存放的，把它们按顺序拼回去就能
还原整页内容，之后就能用 rebuild_index.py 重建索引而不必重跑视觉模型。

还原是有损的：chunk 之间用 "\\n\\n" 连接，原文里如果有更多连续空行，这里恢复不出来。
对切块和检索没有影响——切块本身就会把连续空行折叠掉。

已经有转录的文档默认跳过，避免用有损的还原结果覆盖掉完整的原始转录。

用法：
    PYTHONPATH=. .venv/bin/python scripts/backfill_transcripts.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/backfill_transcripts.py
"""

import argparse
import collections
import pickle
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag import transcripts  # noqa: E402
from rag.pdf_loader import PageText  # noqa: E402

DATA_DIR = BASE_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="已有转录的文档也重新覆盖")
    args = parser.parse_args()

    chunks_path = DATA_DIR / "chunks.pkl"
    if not chunks_path.exists():
        sys.exit("找不到 data/chunks.pkl")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # 按 (文档, 页码) 聚合，保持 chunk 在列表里的原始顺序
    pages: dict[tuple[str, int], list] = collections.defaultdict(list)
    image_paths: dict[tuple[str, int], str | None] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        pages[key].append(chunk.text)
        image_paths.setdefault(key, chunk.image_path)

    by_source: dict[str, list[PageText]] = collections.defaultdict(list)
    for (source, page), parts in pages.items():
        by_source[source].append(
            PageText(
                source=source,
                page=page,
                text="\n\n".join(parts),
                image_path=image_paths[(source, page)],
            )
        )

    existing = set(transcripts.list_sources(DATA_DIR))
    print(f"chunks.pkl 中有 {len(by_source)} 份文档、{len(pages)} 页")
    if existing:
        print(f"已存在转录的文档: {len(existing)} 份")

    for source, page_list in sorted(by_source.items()):
        if source in existing and not args.force:
            print(f"  跳过（已有转录） {source[:44]:<46} {len(page_list)} 页")
            continue
        if args.dry_run:
            print(f"  将写入 {source[:44]:<46} {len(page_list)} 页")
            continue
        doc_dir = transcripts.save_pages(DATA_DIR, source, page_list)
        print(f"  已写入 {source[:44]:<46} {len(page_list)} 页 -> {doc_dir.name}")

    if not args.dry_run:
        total = sum(len(transcripts.load_pages(DATA_DIR, s)) for s in transcripts.list_sources(DATA_DIR))
        print(f"\n完成：transcripts/ 现有 {len(transcripts.list_sources(DATA_DIR))} 份文档、{total} 页")


if __name__ == "__main__":
    main()
