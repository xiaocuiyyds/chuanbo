"""二次精排：用 DashScope 的 rerank 模型对 RRF 融合出的候选做精排。

RRF 融合是纯粹按排名求和，一个候选只在某一路（比如图像检索）里排第一、
其余两路完全没信号时，容易被"三路都沾一点边"的候选压过去，排不到前面。
reranker 不关心候选是从哪一路召回的，只看候选内容跟问题的真实相关性打分，
能纠正这种排名失真。
"""

import requests

DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3-rerank"


def rerank(api_key: str, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """返回 [(原始 documents 下标, 相关性分数), ...]，按分数从高到低排序。"""
    response = requests.post(
        DASHSCOPE_RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": RERANK_MODEL,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()["output"]["results"]
    return [(r["index"], r["relevance_score"]) for r in results]
