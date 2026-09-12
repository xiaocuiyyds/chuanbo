"""从 data/transcripts/ 重建切块和向量索引，不重跑视觉模型。

这是转录持久化的兑现点。调整切块参数、改写上下文 prompt、换嵌入模型之后，用这个脚本
重建即可，不必把几百页手册重新过一遍视觉模型——那是整个流程里最贵最慢的一步。

重建的是文本这一路（chunks.pkl + index.faiss）。页面图像索引（image_index.faiss）
不受切块和上下文影响，默认不动；只有换了多模态嵌入模型才需要 --rebuild-images。

默认复用已有 chunk 的上下文：切块参数没变时，同样的片段没必要重新生成一遍上下文，
能省下大量调用。改了 CONTEXT_SYSTEM_PROMPT 或者想重新生成时加 --regenerate-contexts。

用法：
    PYTHONPATH=. .venv/bin/python scripts/rebuild_index.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/rebuild_index.py
    PYTHONPATH=. .venv/bin/python scripts/rebuild_index.py --regenerate-contexts
"""

import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag import transcripts  # noqa: E402
from rag.chunker import chunk_pages  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402
from rag.llm import contextualize_chunks  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402

DATA_DIR = BASE_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只切块并统计，不调用接口、不写盘")
    parser.add_argument(
        "--regenerate-contexts",
        action="store_true",
        help="所有片段都重新生成上下文，而不是复用已有的",
    )
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")

    sources = transcripts.list_sources(DATA_DIR)
    if not sources:
        sys.exit(
            "data/transcripts/ 是空的。\n"
            "如果索引是在转录持久化之前建的，先跑 scripts/backfill_transcripts.py 把转录补出来。"
        )

    pages = transcripts.load_all_pages(DATA_DIR)
    chunks = chunk_pages(pages)
    print(f"来源 {len(sources)} 份文档、{len(pages)} 页 -> 切出 {len(chunks)} 个片段")

    chunks_path = DATA_DIR / "chunks.pkl"
    index_path = DATA_DIR / "index.faiss"

    # 复用旧上下文：按 (文档, 页码, 片段正文) 对齐。切块参数没变时能全部命中，
    # 调整过参数则只有内容完全一致的片段能复用，其余重新生成。
    reused = 0
    if not args.regenerate_contexts and chunks_path.exists():
        with open(chunks_path, "rb") as f:
            old_chunks = pickle.load(f)
        old_contexts = {(c.source, c.page, c.text): c.context for c in old_chunks if c.context}
        for chunk in chunks:
            context = old_contexts.get((chunk.source, chunk.page, chunk.text))
            if context:
                chunk.context = context
                reused += 1
        print(f"复用已有上下文 {reused} 个，需新生成 {len(chunks) - reused} 个")

    todo = [c for c in chunks if not c.context]
    if args.dry_run:
        print(f"[dry-run] 将调用 {len(todo)} 次上下文生成 + {len(chunks)} 次嵌入，未写盘")
        return
    if todo and not (deepseek_key and dashscope_key):
        sys.exit("缺少 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")

    if todo:
        start = time.time()
        done = [0]

        def on_progress(n: int, total: int) -> None:
            done[0] = n
            if n % 50 == 0 or n == total:
                print(f"  生成上下文 {n}/{total}  用时 {time.time()-start:.0f}s")

        contextualize_chunks(deepseek_key, pages, todo, args.workers, on_progress)
        failed = sum(1 for c in todo if not c.context)
        if failed:
            print(f"  注意：{failed} 个片段的上下文生成失败，将以无上下文入库")

    print(f"嵌入 {len(chunks)} 个片段...")
    vectors = embed_texts(dashscope_key, [c.retrieval_text for c in chunks])

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)

    if chunks_path.exists():
        shutil.copy2(chunks_path, DATA_DIR / "chunks.pkl.prerebuild.bak")
    if index_path.exists():
        shutil.copy2(index_path, DATA_DIR / "index.faiss.prerebuild.bak")
    store.save(DATA_DIR)

    print(f"完成：{store.size} 个片段，向量 {store.index.ntotal}，来源 {len(store.sources)} 份")
    print("原文件已备份为 chunks.pkl.prerebuild.bak / index.faiss.prerebuild.bak")
    print("注意：图像索引未改动；重启服务后 BM25 会自动重建。")


if __name__ == "__main__":
    main()
