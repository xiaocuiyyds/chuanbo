"""重新转录退化的页面，并把索引里对应的 chunk 一并换掉。

背景：对着位号密集的管路图，Qwen-VL 偶尔会放弃真正读图，改成把同一句套话套到几十上百个
位号上（Machinery p148 有 233 行都是"**XXX**: A pipe leading to the main engine."）。
这种输出对检索是纯噪声，还会被当成手册原文喂给回答模型。rag/pdf_loader.py 里加了检测和
自动重试，但存量索引里的坏页还在，需要用这个脚本重转一遍。

退化是偶发的——同一页同一 prompt 重跑一次通常就正常了，所以重转录本身就能解决大部分问题；
新结果仍然退化时保留原样，不会越修越差。

必须等 repair_contexts.py 跑完再执行，两个脚本改的是同一份 chunks.pkl 和 index.faiss。

用法：
    PYTHONPATH=. .venv/bin/python scripts/repair_pages.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/repair_pages.py --limit 3
    PYTHONPATH=. .venv/bin/python scripts/repair_pages.py
"""

import argparse
import collections
import concurrent.futures
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.chunker import chunk_pages  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402
from rag.llm import contextualize_chunks  # noqa: E402
from rag.pdf_loader import (  # noqa: E402
    DASHSCOPE_BASE_URL,
    PageText,
    _degenerate_reason,
    _transcribe_page,
)

DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "page_images"


def find_damaged(chunks: list) -> dict[tuple[str, int], str]:
    """按页把 chunk 拼回整页转录，挑出仍然是退化输出的页。"""
    pages: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for chunk in chunks:
        pages[(chunk.source, chunk.page)].append(chunk.text)
    damaged = {}
    for key, parts in pages.items():
        reason = _degenerate_reason("\n\n".join(parts), "stop")
        if reason:
            damaged[key] = reason
    return damaged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
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

    damaged = find_damaged(chunks)
    print(f"检出退化页 {len(damaged)} 页，涉及 chunk {sum(1 for c in chunks if (c.source, c.page) in damaged)} 条")
    if args.dry_run:
        for key, reason in sorted(damaged.items()):
            print(f"   {key[0][:30]:<32} p{key[1]:<4} {reason}")
        return

    targets = sorted(damaged)[: args.limit] if args.limit else sorted(damaged)
    image_names = {(c.source, c.page): c.image_path for c in chunks if c.image_path}
    client = OpenAI(api_key=dashscope_key, base_url=DASHSCOPE_BASE_URL)
    start = time.time()

    def worker(key: tuple[str, int]) -> tuple[tuple[str, int], str | None]:
        name = image_names.get(key)
        if not name or not (IMAGES_DIR / name).exists():
            return key, None  # 页图丢了就没法重转，跳过
        try:
            text = _transcribe_page(client, (IMAGES_DIR / name).read_bytes())
        except Exception as e:
            print(f"   {key[0][:24]} p{key[1]} 转录失败: {e}")
            return key, None
        if _degenerate_reason(text, "stop"):
            return key, None  # 重转之后还是退化，保留原样，不要越修越差
        return key, text

    new_texts: dict[tuple[str, int], str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, key) for key in targets]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key, text = future.result()
            if text:
                new_texts[key] = text
            print(f"  [{done}/{len(targets)}] {key[0][:26]:<28} p{key[1]:<4} "
                  f"{'已重转 ' + str(len(text)) + ' 字' if text else '未改善，保留原样'}  {time.time()-start:.0f}s")

    if not new_texts:
        print("没有页面得到改善，索引未改动")
        return

    new_pages = [PageText(source=s, page=p, text=t, image_path=image_names.get((s, p)))
                 for (s, p), t in new_texts.items()]
    new_chunks = chunk_pages(new_pages)
    print(f"重新切块 {len(new_chunks)} 条，生成上下文...")
    contextualize_chunks(deepseek_key, new_pages, new_chunks, 5)
    new_vectors = embed_texts(dashscope_key, [c.retrieval_text for c in new_chunks])
    new_vectors = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)

    # 在原来的位置上做替换：一页的旧 chunk 全部删掉，新 chunk 插在该页第一条的位置，
    # 这样 chunk 列表仍然按 文档 -> 页码 的顺序排列
    old_vectors = index.reconstruct_n(0, index.ntotal)
    by_page: dict[tuple[str, int], list] = collections.defaultdict(list)
    for chunk in new_chunks:
        by_page[(chunk.source, chunk.page)].append(chunk)
    new_vec_by_chunk = {id(c): v for c, v in zip(new_chunks, new_vectors)}

    kept_chunks, kept_vectors, inserted = [], [], set()
    for i, chunk in enumerate(chunks):
        key = (chunk.source, chunk.page)
        if key not in new_texts:
            kept_chunks.append(chunk)
            kept_vectors.append(old_vectors[i])
            continue
        if key in inserted:
            continue  # 这一页的旧 chunk 已经被整体替换掉了
        inserted.add(key)
        for replacement in by_page[key]:
            kept_chunks.append(replacement)
            kept_vectors.append(new_vec_by_chunk[id(replacement)])

    rebuilt = faiss.IndexFlatIP(old_vectors.shape[1])
    rebuilt.add(np.array(kept_vectors, dtype="float32"))

    shutil.copy2(chunks_path, DATA_DIR / "chunks.pkl.prepages.bak")
    shutil.copy2(index_path, DATA_DIR / "index.faiss.prepages.bak")
    with open(chunks_path, "wb") as f:
        pickle.dump(kept_chunks, f)
    faiss.write_index(rebuilt, str(index_path))

    print(f"完成：{len(new_texts)} 页重转录，chunk {len(chunks)} -> {len(kept_chunks)}")
    print(f"仍然退化的页: {len(find_damaged(kept_chunks))}")
    print("原文件已备份为 chunks.pkl.prepages.bak / index.faiss.prepages.bak")


if __name__ == "__main__":
    main()
