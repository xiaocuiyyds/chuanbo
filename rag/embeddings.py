import numpy as np
from openai import OpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_BATCH_SIZE = 10  # DashScope limit per request


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def embed_texts(api_key: str, texts: list[str]) -> np.ndarray:
    client = _client(api_key)
    vectors = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors, dtype="float32")


def embed_query(api_key: str, query: str) -> np.ndarray:
    return embed_texts(api_key, [query])[0]
