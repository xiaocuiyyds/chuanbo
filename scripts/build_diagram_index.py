"""跑完图纸页的三项抽取，写成 data/diagram_index.json。

三项抽取的可靠程度差别很大，这里如实记录，不做混淆：

- 介质与管线（矢量坐标）：精确，零成本，全库 68 页图例识别成功率 68/68
- 位号（切片识别 + 多遍取交集）：可靠，人工核对过的页面 26/26 全召回；
  但有截断和误读的残留，是全流程里唯一花钱的部分（约 27 秒/页）
- 器件符号（模板匹配）：精确率高（核对过的页面零误报），召回有限，
  当前模板库只覆盖三种符号，漏掉其他样式的阀和法兰类

脚本可续跑：已经处理过的页面默认跳过，中断后重跑不会重复花钱。

用法：
    PYTHONPATH=. .venv/bin/python scripts/build_diagram_index.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/build_diagram_index.py --limit 10
    PYTHONPATH=. .venv/bin/python scripts/build_diagram_index.py --no-tags   # 只做免费的矢量部分
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

from rag.diagram_pipes import has_vector_pipes, summarize_page  # noqa: E402
from rag.diagram_symbols import crop_template, detect_symbols, summarize, to_black_mask  # noqa: E402
from rag.diagram_tags import (  # noqa: E402
    MIN_VOTES,
    RUNS,
    TILE_GRID,
    extract_page_tags,
    is_diagram_page,
    load_diagram_image,
    summarize_diagram,
)

DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "diagram_index.json"

# 位号识别方式的标识，写进每条记录。识别方式改了以后，靠它认出哪些页的数据是旧的——
# 否则脚本只能按"这一页在不在索引里"判断，算法一改就分不清新旧，只能整体重跑。
# 上一次就吃了这个亏：跑到 52 页时接口欠费中断，剩下的页仍是旧算法的结果，
# 而脚本报告"已完成 87 页、待处理 0 页"。
TAG_METHOD = f"tiles{TILE_GRID}x{TILE_GRID}-runs{RUNS}-votes{MIN_VOTES}"


def _is_stale(entry: dict) -> bool:
    """这一页的位号是不是用旧方式跑的。没有 tags 的条目不算旧，那是本来就没抽到。"""
    return entry.get("tag_method") != TAG_METHOD

# 模板库：从 Machinery p112 上框出来的符号。扩充模板就是往这里加条目，
# 键是符号类型名，值是 (页索引, x0, y0, x1, y1)。
TEMPLATE_SPECS = {
    "阀-领结实心顶": ("Prime Purity Machinery Manual First Draft - July 2026.pdf", 111, (1608, 340, 1646, 394)),
    "阀-领结空心": ("Prime Purity Machinery Manual First Draft - July 2026.pdf", 111, (1664, 402, 1714, 440)),
    "滤器-双X方框": ("Prime Purity Machinery Manual First Draft - July 2026.pdf", 111, (1762, 262, 1806, 286)),
}


def load_templates() -> dict:
    templates = {}
    for kind, (source, page_index, box) in TEMPLATE_SPECS.items():
        path = BASE_DIR / source
        if not path.exists():
            continue
        doc = fitz.open(path)
        try:
            image = load_diagram_image(doc, page_index)
            if image is not None:
                templates[kind] = crop_template(image, box)
        finally:
            doc.close()
    return templates


def page_title(page: fitz.Page) -> str:
    lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
    for line in lines:
        if "Illustration" in line:
            return line[:120]
    return lines[0][:120] if lines else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source", default="")
    parser.add_argument("--no-tags", action="store_true", help="跳过位号识别，只做免费的矢量与符号部分")
    parser.add_argument("--redo", action="store_true", help="已处理过的页面也重新跑")
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not (args.dry_run or args.no_tags or api_key):
        sys.exit("缺少 DASHSCOPE_API_KEY（或加 --no-tags 只跑免费部分）")

    result: dict = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else {}

    pdfs = sorted(BASE_DIR.glob("*.pdf"))
    if args.source:
        pdfs = [p for p in pdfs if p.name == args.source]

    targets = []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        for i in range(len(doc)):
            if not is_diagram_page(doc, i):
                continue
            existing = result.get(pdf.name, {}).get(str(i + 1))
            if existing is not None and not args.redo and not _is_stale(existing):
                continue
            targets.append((pdf, i))
        doc.close()

    fresh = sum(1 for pp in result.values() for e in pp.values() if not _is_stale(e))
    print(
        f"待处理图纸页 {len(targets)} 页"
        f"（已完成 {sum(len(v) for v in result.values())} 页，其中 {fresh} 页是当前识别方式）"
    )
    if args.limit:
        targets = targets[: args.limit]
        print(f"--limit {args.limit}，本次处理 {len(targets)} 页")
    if args.dry_run or not targets:
        return

    templates = load_templates()
    print(f"模板库 {len(templates)} 个符号：{list(templates)}")

    start = time.time()
    open_doc, open_path = None, None
    try:
        for n, (pdf, index) in enumerate(targets, 1):
            if open_path != pdf:
                if open_doc:
                    open_doc.close()
                open_doc, open_path = fitz.open(pdf), pdf

            page = open_doc[index]
            entry: dict = {"title": page_title(page)}

            # 矢量部分：精确、免费
            if has_vector_pipes(page):
                summary = summarize_page(page)
                entry["media"] = sorted(summary["media"])
            else:
                entry["media"] = []

            image = load_diagram_image(open_doc, index)
            if image is not None and templates:
                hits = detect_symbols(to_black_mask(image), templates)
                entry["symbols"] = summarize(hits)
            else:
                entry["symbols"] = {}

            if args.no_tags or image is None:
                entry["tags"], entry["unstable"] = [], []
            else:
                stable, unstable = extract_page_tags(api_key, image)
                entry["tags"], entry["unstable"] = sorted(stable), sorted(unstable)
                entry["gist"] = summarize_diagram(api_key, image)

            entry["tag_method"] = TAG_METHOD
            result.setdefault(pdf.name, {})[str(index + 1)] = entry
            OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"  [{n}/{len(targets)}] {pdf.name[:24]:<26} p{index + 1:<5} "
                f"介质 {len(entry['media']):>2}  符号 {sum(entry['symbols'].values()):>3}  "
                f"位号 {len(entry['tags']):>3}   {time.time() - start:.0f}s"
            )
    finally:
        if open_doc:
            open_doc.close()

    pages = sum(len(v) for v in result.values())
    print(f"\n完成，累计覆盖 {pages} 页，写入 {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
