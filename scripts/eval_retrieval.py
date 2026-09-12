"""检索回归评测：跑一组标注好答案页的问题，报告 recall@k 和 MRR。

用途是回归，不是刷分。改了切块参数、检索融合方式、查询预处理之后跑一遍，看命中率
有没有掉——没有这个，任何改动都只能靠感觉判断好坏。

评测集在 evals/retrieval.jsonl，每行一道题：

    单页题  {"q": .., "source": .., "page": 12, "evidence": "该页必含的字符串"}
    复合题  {"q": .., "source": .., "pages": [12, 15], "evidence": {"12": .., "15": ..}, "kind": "multi"}

复合题的答案需要跨页才完整（例如"泵的规格是多少、启动前要先做什么"分别在两页上）。
它们用"所需页是否都进了 top5"来衡量，单页题的 recall 看不出这种差别。
曾用它们评估过"让模型判断缺口再补检一轮"的多跳方案，结论是覆盖率毫无改善、延迟翻倍，
已经回退；这组题保留下来，作为衡量跨页覆盖能力的基准。

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
    """读评测集，把单页题和复合题统一成 pages/evidence 两个字段。"""
    cases = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            if "pages" not in case:
                case["pages"] = [case["page"]]
                case["evidence"] = {str(case["page"]): case["evidence"]}
            case["evidence"] = {int(k): v for k, v in case["evidence"].items()}
            case.setdefault("kind", "single")
            cases.append(case)
    return cases


def verify_cases(cases: list[dict], store: VectorStore) -> int:
    """检查每道题标注的页确实存在、且确实包含 evidence 那句话。"""
    pages: dict[tuple[str, int], str] = {}
    for chunk in store.chunks:
        key = (chunk.source, chunk.page)
        pages[key] = pages.get(key, "") + "\n" + chunk.text

    bad = 0
    for case in cases:
        for page in case["pages"]:
            key = (case["source"], page)
            evidence = case["evidence"][page]
            if key not in pages:
                print(f"  ✗ 标注的页不存在: {case['source'][:30]} p{page}  ({case['q']})")
                bad += 1
            elif evidence not in pages[key]:
                print(f"  ✗ 该页不含 evidence {evidence!r}: {case['source'][:30]} p{page}")
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

    def run(case: dict) -> tuple[dict, int | None, float]:
        """返回 (题目, 首个标注页的名次, 标注页被覆盖的比例)。

        名次用于单页题的 recall/MRR；覆盖率用于复合题——它问的是"答全这道题所需的
        几页里，检索到了几页"，这才是多跳机制真正该改善的指标。"""
        try:
            _, results = retrieve(
                store, image_index, dashscope_key, deepseek_key, [], case["q"],
                use_rerank=not args.no_rerank,
            )
        except Exception as e:
            print(f"  检索失败 {case['q']}: {e}")
            return case, None, 0.0

        retrieved = {(c.source, c.page) for c, _ in results}
        wanted = {(case["source"], p) for p in case["pages"]}
        coverage = len(retrieved & wanted) / len(wanted)

        best = None
        for rank, (chunk, _) in enumerate(results, 1):
            if (chunk.source, chunk.page) in wanted:
                best = rank
                break
        return case, best, coverage

    ranks: list[tuple[dict, int | None, float]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case, rank, coverage in pool.map(run, cases):
            ranks.append((case, rank, coverage))

    single = [(c, r, cov) for c, r, cov in ranks if c["kind"] == "single"]
    multi = [(c, r, cov) for c, r, cov in ranks if c["kind"] == "multi"]
    n = len(single) or 1
    recall1 = sum(1 for _, r, _ in single if r == 1) / n
    recall3 = sum(1 for _, r, _ in single if r and r <= 3) / n
    recall5 = sum(1 for _, r, _ in single if r and r <= 5) / n
    mrr = sum(1 / r for _, r, _ in single if r) / n

    print(f"\n评测集 {n} 题   重排 {'关' if args.no_rerank else '开'}   用时 {time.time()-start:.0f}s")
    print(f"  recall@1  {recall1:.1%}")
    print(f"  recall@3  {recall3:.1%}")
    print(f"  recall@5  {recall5:.1%}")
    print(f"  MRR       {mrr:.3f}")

    if multi:
        full = sum(1 for _, _, cov in multi if cov == 1.0)
        avg = sum(cov for _, _, cov in multi) / len(multi)
        print(f"\n复合题 {len(multi)} 道（答案跨页，用来检验补充检索）")
        print(f"  所需页全部检索到  {full}/{len(multi)}")
        print(f"  平均页面覆盖率    {avg:.1%}")
        for case, _, cov in multi:
            if cov < 1.0:
                print(f"    覆盖 {cov:.0%}  {case['q']}  (需 p{case['pages']})")

    missed = [(c, r) for c, r, _ in ranks if r is None]
    if missed:
        print(f"\n完全未命中 {len(missed)} 题：")
        for case, _ in missed:
            print(f"  {case['q']}\n      标注: {case['source'][:34]} p{case['pages']}")


if __name__ == "__main__":
    main()
