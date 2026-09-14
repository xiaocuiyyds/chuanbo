"""抽取图纸页上的设备位号，存成位号索引。

图纸上的位号只印在位图里、不在 PDF 文字层中，所以现有的页面转录抓不到它们——
实测 Machinery p112 的整页转录一个真位号都没有，反倒编造了 162 个不存在的。
这个脚本用切片加多遍取交集的方式把它们抄出来，见 rag/diagram_tags.py。

结果写到 data/diagram_tags.json：

    {"文档名": {"页码": {"title": 图名, "tags": [...], "unstable": [...]}}}

tags 是每一遍都读到的，unstable 是只有部分遍次读到的（多半是误识），分开存，
入库只用 tags，unstable 留作人工抽查的线索。

用法：
    PYTHONPATH=. .venv/bin/python scripts/extract_diagram_tags.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/extract_diagram_tags.py --limit 5
    PYTHONPATH=. .venv/bin/python scripts/extract_diagram_tags.py --source "Prime Purity Machinery Manual First Draft - July 2026.pdf"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import fitz
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.diagram_tags import extract_page_tags, is_diagram_page, load_diagram_image  # noqa: E402

DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "diagram_tags.json"


def page_title(page: fitz.Page) -> str:
    """图纸页的文字层通常只剩标题和页眉，取其中的 Illustration 行作为图名。"""
    lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
    for line in lines:
        if "Illustration" in line:
            return line[:120]
    return lines[0][:120] if lines else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只列出图纸页，不调用接口")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 页")
    parser.add_argument("--source", default="", help="只处理这一份文档")
    parser.add_argument("--runs", type=int, default=2, help="每块跑几遍取交集")
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not args.dry_run and not api_key:
        sys.exit("缺少 DASHSCOPE_API_KEY")

    pdfs = sorted(BASE_DIR.glob("*.pdf"))
    if args.source:
        pdfs = [p for p in pdfs if p.name == args.source]
        if not pdfs:
            sys.exit(f"找不到 {args.source}")

    targets = []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        for i in range(len(doc)):
            if is_diagram_page(doc, i):
                targets.append((pdf, i, page_title(doc[i])))
        doc.close()

    print(f"检出图纸页 {len(targets)} 页")
    if args.limit:
        targets = targets[: args.limit]
        print(f"--limit {args.limit}，本次处理 {len(targets)} 页")
    if args.dry_run:
        for pdf, i, title in targets:
            print(f"   {pdf.name[:34]:<36} p{i + 1:<5} {title}")
        return
    if not targets:
        return

    result: dict = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else {}
    start = time.time()
    total_stable = total_unstable = 0

    # 同一个 PDF 的页面连续处理，避免反复打开关闭
    open_doc, open_path = None, None
    try:
        for n, (pdf, index, title) in enumerate(targets, 1):
            if open_path != pdf:
                if open_doc:
                    open_doc.close()
                open_doc, open_path = fitz.open(pdf), pdf

            image = load_diagram_image(open_doc, index)
            if image is None:
                continue
            stable, unstable = extract_page_tags(api_key, image, runs=args.runs)
            total_stable += len(stable)
            total_unstable += len(unstable)

            result.setdefault(pdf.name, {})[str(index + 1)] = {
                "title": title,
                "tags": sorted(stable),
                "unstable": sorted(unstable),
            }
            print(
                f"  [{n}/{len(targets)}] {pdf.name[:26]:<28} p{index + 1:<5} "
                f"{image.size[0]}x{image.size[1]}  稳定 {len(stable):>3}  不稳定 {len(unstable):>3}  "
                f"{time.time() - start:.0f}s"
            )
            OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if open_doc:
            open_doc.close()

    pages = sum(len(v) for v in result.values())
    print(
        f"\n完成：本次 {len(targets)} 页，稳定位号 {total_stable} 个、不稳定 {total_unstable} 个"
        f"（{total_unstable / max(1, total_stable + total_unstable):.0%} 被判为不可靠而剔除）"
    )
    print(f"累计已覆盖 {pages} 页，写入 {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
