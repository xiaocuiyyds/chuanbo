"""页面级图像向量索引：用多模态 embedding 直接对手册原图做检索。

跟 rag/vectorstore.py 里的文本向量索引是互补关系——文本索引依赖 Qwen-VL 转录的文字质量，
转录得越简略（比如复杂表格、大量色块），检索信号就越弱；这里绕开转录，直接对图片做
text-to-image 检索，即使转录文字不够精确，也能凭视觉内容把相关页面召回。
"""

import base64
import json
from pathlib import Path

import faiss
import numpy as np
import requests

DASHSCOPE_MM_EMBED_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
)
MM_EMBED_MODEL = "qwen3-vl-embedding"

INDEX_FILENAME = "image_index.faiss"
PAGES_FILENAME = "image_pages.json"


def _embed(api_key: str, contents: list[dict]) -> list[list[float]]:
    response = requests.post(
        DASHSCOPE_MM_EMBED_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MM_EMBED_MODEL, "input": {"contents": contents}},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = sorted(data["output"]["embeddings"], key=lambda e: e["index"])
    return [e["embedding"] for e in embeddings]


def embed_image(api_key: str, image_bytes: bytes) -> list[float]:
    b64 = base64.b64encode(image_bytes).decode()
    return _embed(api_key, [{"image": f"data:image/png;base64,{b64}"}])[0]


def embed_text_query(api_key: str, text: str) -> list[float]:
    return _embed(api_key, [{"text": text}])[0]


class ImageIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.pages: list[tuple[str, int]] = []  # (source, page)，跟 self.index 里的向量一一对应

    def add(self, vectors: np.ndarray, pages: list[tuple[str, int]]) -> None:
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.index.add(normalized.astype("float32"))
        self.pages.extend(pages)

    def remove_source(self, source: str) -> int:
        """删掉某个文档的全部页面向量，返回删除条数。跟 VectorStore.remove_source 配套使用，
        否则重复上传后图像检索仍会命中上一次入库的页面。"""
        keep = [i for i, (src, _) in enumerate(self.pages) if src != source]
        removed = len(self.pages) - len(keep)
        if not removed:
            return 0
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        if keep:
            self.index.add(vectors[keep])
        self.pages = [self.pages[i] for i in keep]
        return removed

    def search(self, query_vector: list[float], k: int = 10) -> list[tuple[tuple[str, int], float]]:
        if self.index.ntotal == 0:
            return []
        vec = np.array(query_vector, dtype="float32")
        normalized = vec / np.linalg.norm(vec)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(normalized.reshape(1, -1), k)
        return [
            (self.pages[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1
        ]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / INDEX_FILENAME))
        with open(dir_path / PAGES_FILENAME, "w", encoding="utf-8") as f:
            json.dump(self.pages, f, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: Path) -> "ImageIndex | None":
        index_path = dir_path / INDEX_FILENAME
        pages_path = dir_path / PAGES_FILENAME
        if not (index_path.exists() and pages_path.exists()):
            return None
        store = cls.__new__(cls)
        store.index = faiss.read_index(str(index_path))
        with open(pages_path, encoding="utf-8") as f:
            store.pages = [tuple(p) for p in json.load(f)]
        return store

    @staticmethod
    def clear(dir_path: Path) -> None:
        for filename in (INDEX_FILENAME, PAGES_FILENAME):
            path = dir_path / filename
            if path.exists():
                path.unlink()
