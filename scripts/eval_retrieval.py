"""检索回归评测：跑一组标注好答案页的问题，报告 recall@k 和 MRR。

用途是回归，不是刷分。改了切块参数、检索融合方式、查询预处理之后跑一遍，看命中率
有没有掉——没有这个，任何改动都只能靠感觉判断好坏。

评测集在 evals/retrieval.jsonl，每行一道题：

    {"q": 问题, "source": 答案所在文档, "page": 答案所在页, "evidence": 该页必含的字符串}

evidence 是用来校验标注本身的：标注一律先用 --verify 对着语料检查一遍，确认那句话
真的在标注的页上，否则整个评测就建立在错误的基准上。标注来自语料内容，不是从系统的
检索结果反推的——后者是循环论证，永远会得满分。

这个脚本会真实调用接口（查询改写、嵌入、重排），跑一遍有成本。

用法：
    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py --verify   # 只校验标注，不调接口
    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py
    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py --no-rerank
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.image_index import ImageIndex  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402

DATA_DIR = BASE_DIR / "data"
EVAL_PATH = BASE_DIR / "evals" / "retrieval.jsonl"


def load_cases() -> list[dict]:
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_cases(cases: list[dict], store: VectorStore) -> int:
    """检查每道题标注的页确实存在、且确实包含 evidence 那句话。"""
    pages: dict[tuple[str, int], str] = {}
    for chunk in store.chunks:
        key = (chunk.source, chunk.page)
        pages[key] = pages.get(key, "") + "\n" + chunk.text

    bad = 0
    for case in cases:
        key = (case["source"], case["page"])
        if key not in pages:
            print(f"  ✗ 标注的页不存在: {case['source'][:30]} p{case['page']}  ({case['q']})")
            bad += 1
        elif case["evidence"] not in pages[key]:
            print(f"  ✗ 该页不含 evidence {case['evidence']!r}: {case['source'][:30]} p{case['page']}")
            bad += 1
    print(f"标注校验：{len(cases) - bad}/{len(cases)} 条通过")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="只校验标注，不调用接口")
    parser.add_argument("--no-rerank", action="store_true", help="关掉重排，用于对比它的贡献")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    store = VectorStore.load(DATA_DIR)
    if store is None:
        sys.exit("知识库为空，先建库")

    cases = load_cases()
    if verify_cases(cases, store) and not args.verify:
        sys.exit("标注有误，先修正评测集再跑评测")
    if args.verify:
        return

    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not (dashscope_key and deepseek_key):
        sys.exit("缺少 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")

    from server.main import retrieve  # 延迟导入：它在模块加载时会读索引

    image_index = ImageIndex.load(DATA_DIR)
    start = time.time()

    def run(case: dict) -> tuple[dict, int | None]:
        """返回标注页在检索结果中的名次（从 1 开始），没命中则为 None。"""
        try:
            _, results = retrieve(
                store, image_index, dashscope_key, deepseek_key, [], case["q"],
                use_rerank=not args.no_rerank,
            )
        except Exception as e:
            print(f"  检索失败 {case['q']}: {e}")
            return case, None
        for rank, (chunk, _) in enumerate(results, 1):
            if (chunk.source, chunk.page) == (case["source"], case["page"]):
                return case, rank
        return case, None

    ranks: list[tuple[dict, int | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case, rank in pool.map(run, cases):
            ranks.append((case, rank))

    n = len(ranks)
    recall1 = sum(1 for _, r in ranks if r == 1) / n
    recall3 = sum(1 for _, r in ranks if r and r <= 3) / n
    recall5 = sum(1 for _, r in ranks if r and r <= 5) / n
    mrr = sum(1 / r for _, r in ranks if r) / n

    print(f"\n评测集 {n} 题   重排 {'关' if args.no_rerank else '开'}   用时 {time.time()-start:.0f}s")
    print(f"  recall@1  {recall1:.1%}")
    print(f"  recall@3  {recall3:.1%}")
    print(f"  recall@5  {recall5:.1%}")
    print(f"  MRR       {mrr:.3f}")

    missed = [(c, r) for c, r in ranks if r is None]
    if missed:
        print(f"\n未命中 {len(missed)} 题：")
        for case, _ in missed:
            print(f"  {case['q']}\n      标注: {case['source'][:34]} p{case['page']}")


if __name__ == "__main__":
    main()
