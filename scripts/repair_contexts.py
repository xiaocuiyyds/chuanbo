"""重新生成被截断的 chunk 上下文，并同步更新向量索引。

背景：建库时 generate_chunk_context 用的是 max_tokens=120，而 deepseek-v4-flash 默认开
思考模式、思考内容同样计入这个预算，导致大部分 chunk 的定位性上下文在写到一半时就被截断
（少数直接返回空字符串）。rag/llm.py 里关掉思考模式后新入库的文档不再有这个问题，但已经
建好的索引里存量数据还是坏的，需要用这个脚本就地修一遍。

chunk.context 会进 retrieval_text，也就是会影响 embedding 和 BM25，所以重新生成之后
必须把对应的向量一起重算，否则索引里的向量跟 chunk 内容对不上。

用法：
    PYTHONPATH=. .venv/bin/python scripts/repair_contexts.py --dry-run      # 只统计不调用
    PYTHONPATH=. .venv/bin/python scripts/repair_contexts.py --limit 20     # 先试 20 条
    PYTHONPATH=. .venv/bin/python scripts/repair_contexts.py                # 全量修复
"""

import argparse
import collections
import concurrent.futures
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.embeddings import embed_texts  # noqa: E402
from rag.llm import generate_chunk_context  # noqa: E402

DATA_DIR = BASE_DIR / "data"
MIN_CONTEXT_LEN = 15
SENTENCE_END_RE = re.compile(r"[。．.!?！？]\s*$")


def is_damaged(context: str | None) -> bool:
    """空、过短、或末尾没有句子终止符（写到一半被 max_tokens 砍断）都算坏。"""
    if not context:
        return True
    if len(context) < MIN_CONTEXT_LEN:
        return True
    return not SENTENCE_END_RE.search(context)


def rebuild_page_texts(chunks: list) -> dict[tuple[str, int], str]:
    """从 chunk 还原每页的转录全文，作为生成上下文时的背景。

    原始的 PageText 没有落盘，但同一页的 chunk 在列表里是按切分顺序连续存放的，
    拼回去足够用来判断某个片段在整页中的位置。
    """
    pages: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for chunk in chunks:
        pages[(chunk.source, chunk.page)].append(chunk.text)
    return {key: "\n\n".join(parts) for key, parts in pages.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计损坏数量，不调用接口")
    parser.add_argument("--limit", type=int, default=0, help="只修复前 N 条，用于小样本验证")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not args.dry_run and not (deepseek_key and dashscope_key):
        sys.exit("缺少 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")

    chunks_path = DATA_DIR / "chunks.pkl"
    index_path = DATA_DIR / "index.faiss"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(chunks):
        sys.exit(f"索引({index.ntotal})和 chunk({len(chunks)})数量对不上，先别动")

    targets = [i for i, c in enumerate(chunks) if is_damaged(c.context)]
    print(f"chunk 总数 {len(chunks)}，其中上下文损坏 {len(targets)} 条 ({len(targets)/len(chunks):.1%})")
    if args.dry_run:
        return
    if args.limit:
        targets = targets[: args.limit]
        print(f"--limit {args.limit}，本次只修前 {len(targets)} 条")
    if not targets:
        print("没有需要修复的 chunk")
        return

    page_texts = rebuild_page_texts(chunks)
    new_contexts: dict[int, str] = {}
    failures = 0
    start = time.time()

    def worker(i: int) -> tuple[int, str]:
        chunk = chunks[i]
        page_text = page_texts.get((chunk.source, chunk.page), chunk.text)
        try:
            return i, generate_chunk_context(deepseek_key, page_text, chunk.text)
        except Exception:
            return i, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, i) for i in targets]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            i, context = future.result()
            if context:
                new_contexts[i] = context
            else:
                failures += 1
            if done % 50 == 0 or done == len(targets):
                print(f"  生成上下文 {done}/{len(targets)}  失败 {failures}  用时 {time.time()-start:.0f}s")

    if not new_contexts:
        sys.exit("全部生成失败，索引未改动")

    # 只有确认拿到新上下文的 chunk 才改；写进 chunk 后 retrieval_text 随之改变
    for i, context in new_contexts.items():
        chunks[i].context = context

    changed = sorted(new_contexts)
    print(f"重新嵌入 {len(changed)} 条...")
    vectors = embed_texts(dashscope_key, [chunks[i].retrieval_text for i in changed])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # IndexFlatIP 不支持原地改某一行，把全部向量取出来替换掉对应行再重建
    all_vectors = index.reconstruct_n(0, index.ntotal)
    all_vectors[changed] = vectors
    new_index = faiss.IndexFlatIP(all_vectors.shape[1])
    new_index.add(all_vectors)

    shutil.copy2(chunks_path, chunks_path.with_suffix(".pkl.bak"))
    shutil.copy2(index_path, index_path.with_suffix(".faiss.bak"))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(new_index, str(index_path))

    remaining = sum(1 for c in chunks if is_damaged(c.context))
    print(f"完成：修复 {len(changed)} 条，失败 {failures} 条，仍损坏 {remaining} 条")
    print("原文件已备份为 chunks.pkl.bak / index.faiss.bak（BM25 在服务启动时自动重建）")


if __name__ == "__main__":
    main()
