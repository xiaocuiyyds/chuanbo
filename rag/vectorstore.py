import pickle
import re
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunker import Chunk

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.pkl"

RRF_K = 60  # RRF 融合的平滑常数，值越大排名靠后的结果权重差异越小（业界惯用值）
CANDIDATE_POOL = 20  # 向量检索和 BM25 各自先取的候选数，融合后再截断到 k

_NON_TOKEN_RE = re.compile(r"^[\s\W_]+$")


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：中文走 jieba 分词，字母数字编号（型号/报警码）保持完整。"""
    return [tok.lower() for tok in jieba.cut(text) if tok.strip() and not _NON_TOKEN_RE.match(tok)]


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.index.add(normalized)
        self.chunks.extend(chunks)
        self._rebuild_bm25()

    def remove_source(self, source: str) -> int:
        """删掉某个文档的全部 chunk，返回删除条数。

        重复上传同一份 PDF 时必须先调用它——否则新旧 chunk 会同时留在索引里，
        同一段内容被检索到两次，还会挤掉本该出现的其他结果。
        IndexFlatIP 没有按下标删除的接口，只能把保留的向量取出来重建。
        """
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.source != source]
        removed = len(self.chunks) - len(keep)
        if not removed:
            return 0
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        if keep:
            self.index.add(vectors[keep])
        self.chunks = [self.chunks[i] for i in keep]
        self._rebuild_bm25()
        return removed

    def _rebuild_bm25(self) -> None:
        self.bm25 = (
            BM25Okapi([_tokenize(c.retrieval_text) for c in self.chunks]) if self.chunks else None
        )

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        k: int = 5,
        image_page_ranks: dict[tuple[str, int], int] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """向量检索 + BM25 关键词检索 + （可选）图像检索命中的页面，用 RRF 融合排名。

        向量和 BM25 两路分数量纲不同（余弦相似度 vs BM25 分数），直接加权很难调；RRF 只看排名，
        对报警代码/型号这类关键词能命中但语义检索容易漏的情况更稳健。
        image_page_ranks 是外部图像检索（对页面原图做 text-to-image 检索）得到的命中页排名，
        格式为 {(source, page): rank}；用来弥补转录文字过于简略、导致文本检索找不到相关页的情况——
        命中的页面会把该页所有 chunk 一并计入融合排名。
        """
        if self.index.ntotal == 0:
            return []

        pool = min(CANDIDATE_POOL, self.index.ntotal)

        normalized = query_vector / np.linalg.norm(query_vector)
        _, vec_indices = self.index.search(normalized.reshape(1, -1), pool)
        vec_ranks = {int(idx): rank for rank, idx in enumerate(vec_indices[0]) if idx != -1}

        bm25_ranks: dict[int, int] = {}
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(_tokenize(query_text))
            top_bm25 = np.argsort(bm25_scores)[::-1][:pool]
            bm25_ranks = {int(idx): rank for rank, idx in enumerate(top_bm25) if bm25_scores[idx] > 0}

        fused: dict[int, float] = {}
        for idx, rank in vec_ranks.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for idx, rank in bm25_ranks.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        if image_page_ranks:
            page_to_chunk_indices: dict[tuple[str, int], list[int]] = {}
            for i, c in enumerate(self.chunks):
                page_to_chunk_indices.setdefault((c.source, c.page), []).append(i)
            for page_key, rank in image_page_ranks.items():
                for idx in page_to_chunk_indices.get(page_key, []):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_idx = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[idx], score) for idx, score in top_idx]

    def expand_with_neighbors(self, results: list[tuple[Chunk, float]], window: int = 1) -> list[Chunk]:
        """把命中片段前后紧邻的 chunk 一并取出来，按原文顺序返回（含命中片段本身）。

        手册里"操作步骤 1-12"、跨页的报警代码表这类内容会被切成好几块，只把命中的那一块交给
        回答模型，很容易给出半截流程——对照着手册操作的人来说，半截流程比查不到更危险。
        chunks 列表本身就是按 文档 -> 页码 -> 页内顺序 排列的，所以相邻下标就是原文的上下文；
        只在同一个文档内扩展，不会跨到别的手册去。
        """
        if not self.chunks:
            return []
        position = {id(chunk): i for i, chunk in enumerate(self.chunks)}
        wanted: set[int] = set()
        for chunk, _ in results:
            i = position.get(id(chunk))
            if i is None:
                continue
            for j in range(max(0, i - window), min(len(self.chunks), i + window + 1)):
                if self.chunks[j].source == chunk.source:
                    wanted.add(j)
        return [self.chunks[i] for i in sorted(wanted)]

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        seen = dict.fromkeys(chunk.source for chunk in self.chunks)
        return list(seen)

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / INDEX_FILENAME))
        with open(dir_path / CHUNKS_FILENAME, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, dir_path: Path) -> "VectorStore | None":
        index_path = dir_path / INDEX_FILENAME
        chunks_path = dir_path / CHUNKS_FILENAME
        if not (index_path.exists() and chunks_path.exists()):
            return None
        store = cls.__new__(cls)
        store.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            store.chunks = pickle.load(f)
        store._rebuild_bm25()
        return store

    @staticmethod
    def clear(dir_path: Path) -> None:
        for filename in (INDEX_FILENAME, CHUNKS_FILENAME):
            path = dir_path / filename
            if path.exists():
                path.unlink()
