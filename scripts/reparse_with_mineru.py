"""用 MinerU 重新解析手册，结果写入 data/transcripts/。

只改转录层，下游一律不动：写完之后跑 scripts/rebuild_index.py 就能重建索引，
切块、上下文、向量、检索全都沿用原有流程。这正是当初把转录单独落盘的意义。

为什么换：全库 837 页里 782 页（93%）带可提取的文字层，而原来的流程把每一页都渲染成
图片交给视觉模型重新读一遍——等于放着原文不用去重新生成。五页样本实测，位号保真率
视觉模型 60%、MinerU 100%，且曾经退化的两页完全正常。

图纸页默认保留原有转录（--include-diagrams 可覆盖）。那些页的文字层本来就稀疏，
MinerU 只能给出页眉和图例，而图上的设备位号由 rag/diagram_tags.py 另行提取。

改动前会把现有 transcripts/ 整体备份，出问题可以整目录还原。

用法：
    PYTHONPATH=. .venv/bin/python scripts/reparse_with_mineru.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/reparse_with_mineru.py --source "KC-700 Manual.pdf"
    PYTHONPATH=. .venv/bin/python scripts/reparse_with_mineru.py
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import fitz
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag import transcripts  # noqa: E402
from rag.diagram_tags import is_diagram_page  # noqa: E402
from rag.mineru import blocks_to_pages, parse_pdf  # noqa: E402

DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = DATA_DIR / "transcripts.vl.bak"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计，不调用接口")
    parser.add_argument("--source", default="", help="只处理这一份手册")
    parser.add_argument(
        "--include-diagrams",
        action="store_true",
        help="图纸页也用 MinerU 的结果覆盖（默认保留原有转录）",
    )
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("MINERU_API_KEY", "")
    if not args.dry_run and not api_key:
        sys.exit("缺少 MINERU_API_KEY，请在 .env 中配置")

    pdfs = sorted(BASE_DIR.glob("*.pdf"))
    if args.source:
        pdfs = [p for p in pdfs if p.name == args.source]
        if not pdfs:
            sys.exit(f"找不到 {args.source}")

    print(f"{'手册':<44}{'总页':>6}{'图纸页':>7}{'将替换':>7}")
    plan = []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        diagrams = {i + 1 for i in range(len(doc)) if is_diagram_page(doc, i)}
        replace = len(doc) if args.include_diagrams else len(doc) - len(diagrams)
        print(f"{pdf.name[:42]:<44}{len(doc):>6}{len(diagrams):>7}{replace:>7}")
        plan.append((pdf, diagrams, len(doc)))
        doc.close()
    if args.dry_run:
        return

    if BACKUP_DIR.exists():
        print(f"\n备份目录已存在，跳过备份：{BACKUP_DIR.name}")
    else:
        shutil.copytree(transcripts.transcripts_dir(DATA_DIR), BACKUP_DIR)
        print(f"\n已备份现有转录到 {BACKUP_DIR.name}")

    for pdf, diagrams, total in plan:
        start = time.time()
        print(f"\n=== {pdf.name[:46]}（{total} 页）")
        try:
            blocks = parse_pdf(
                api_key,
                pdf.read_bytes(),
                pdf.name,
                on_progress=lambda msg: print(f"    {msg}  {time.time()-start:.0f}s"),
            )
        except Exception as e:
            print(f"    解析失败，保留原有转录：{e}")
            continue

        safe = re.sub(r"[/\\]", "_", pdf.name)
        parsed = blocks_to_pages(blocks, pdf.name, lambda p: f"{safe}_p{p}.png")
        if not parsed:
            print("    未解析出任何内容，跳过")
            continue

        # 已有转录作为底子，只覆盖本次解析到的页；图纸页默认保持不动
        merged = {p.page: p for p in transcripts.load_pages(DATA_DIR, pdf.name)}
        replaced = skipped = 0
        for page in parsed:
            if not args.include_diagrams and page.page in diagrams:
                skipped += 1
                continue
            merged[page.page] = page
            replaced += 1

        transcripts.save_pages(DATA_DIR, pdf.name, sorted(merged.values(), key=lambda p: p.page))
        print(
            f"    完成：替换 {replaced} 页，保留图纸页 {skipped} 页，"
            f"转录层现有 {len(merged)} 页，用时 {time.time()-start:.0f}s"
        )

    print("\n接下来跑 scripts/rebuild_index.py 重建索引，再跑 scripts/eval_retrieval.py 对比指标")
    print(f"要回滚：删掉 transcripts/ 并把 {BACKUP_DIR.name} 改回 transcripts/")


if __name__ == "__main__":
    main()
