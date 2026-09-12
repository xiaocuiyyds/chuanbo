import concurrent.futures
import glob
import json
import os
import queue
import re
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import transcripts
from rag.chunker import chunk_pages
from rag.embeddings import embed_query, embed_texts
from rag.history import (
    delete_conversation,
    list_conversations,
    load_conversation,
    new_conversation_id,
    recent_turns,
    save_conversation,
)
from rag.image_index import ImageIndex, embed_image, embed_text_query
from rag.llm import answer_stream, contextualize_chunks, prepare_query
from rag.pdf_loader import extract_pages
from rag.reranker import rerank
from rag.vectorstore import VectorStore

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "page_images"
N_HISTORY_TURNS = 3

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="船舶操作知识问答 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

state: dict = {
    "vectorstore": VectorStore.load(DATA_DIR),
    "image_index": ImageIndex.load(DATA_DIR),
}

IMAGE_SEARCH_POOL = 10  # 图像检索先取的候选页数，供 RRF 融合
RERANK_POOL = 15  # RRF 融合先多捞的候选数，交给 reranker 精排后再截断到 5
IMAGE_TOP_MIN_SCORE = 0.35  # 图像检索最强命中低于这个分数就不算"有把握"，不做强制保留
NEIGHBOR_WINDOW = 1  # 喂给回答模型时，每个命中片段前后各带几个相邻片段


def retrieve(store, image_index, dashscope_key, deepseek_key, history, question, use_rerank=True):
    """查询改写 -> 向量+BM25+图像三路 RRF 融合 -> （可选）reranker 精排，返回最终 top5。

    RRF 只按排名求和，一个候选如果只在某一路（比如图像检索）里排第一、其余两路完全没有
    信号，容易被"三路都沾一点边"的候选压过去；reranker 不关心候选来自哪一路，只看候选
    内容跟问题的真实相关性重新打分，能纠正这种排名失真。reranker 调用失败时退回 RRF 原始排序。

    但 reranker 本身也是纯文本模型，只看候选片段的转录文字——如果一个页面正是因为转录
    文字过于简略才需要靠图像检索才能被召回，reranker 反而会因为文字信号弱把它判定为不
    相关、挤出结果，等于抵消了图像检索通道的作用。所以图像检索里把握最大的那一页（分数
    达到阈值）会被强制保留一个位置，不受 reranker 意见影响。
    """
    search_query, keyword_query = prepare_query(deepseek_key, history, question)

    # 文本 embedding 和图像通道的多模态 embedding 互不依赖，并行发出省掉一次串行往返
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_exec:
        query_vec_future = pool_exec.submit(embed_query, dashscope_key, search_query)
        image_vec_future = (
            pool_exec.submit(embed_text_query, dashscope_key, search_query)
            if image_index is not None
            else None
        )
        query_vec = query_vec_future.result()

        image_page_ranks = None
        top_image_page = None
        if image_vec_future is not None:
            try:
                image_hits = image_index.search(image_vec_future.result(), k=IMAGE_SEARCH_POOL)
                image_page_ranks = {page: rank for rank, (page, _) in enumerate(image_hits)}
                if image_hits and image_hits[0][1] >= IMAGE_TOP_MIN_SCORE:
                    top_image_page = image_hits[0][0]
            except Exception:
                image_page_ranks = None  # 图像检索通道失败不影响文本检索正常返回

    # BM25 用中英混合的关键词串，向量检索仍然用原问题（embedding 模型本身跨语言）
    pool = store.search(query_vec, keyword_query, k=RERANK_POOL, image_page_ranks=image_page_ranks)
    results = pool[:5]
    if use_rerank and pool:
        try:
            docs = [c.retrieval_text for c, _ in pool]
            reranked = rerank(dashscope_key, search_query, docs, top_n=5)
            results = [(pool[idx][0], score) for idx, score in reranked]
        except Exception:
            pass  # 保留上面 RRF 的兜底结果

    if top_image_page and not any((c.source, c.page) == top_image_page for c, _ in results):
        rescue = next((item for item in pool if (item[0].source, item[0].page) == top_image_page), None)
        if rescue:
            results = results[:-1] + [rescue]

    return search_query, results


def ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


# ---------- 对话历史 ----------


@app.get("/api/conversations")
def api_list_conversations():
    return list_conversations(DATA_DIR)


@app.post("/api/conversations")
def api_new_conversation():
    return {"id": new_conversation_id()}


@app.get("/api/conversations/{conv_id}")
def api_get_conversation(conv_id: str):
    return {"messages": load_conversation(DATA_DIR, conv_id)}


@app.delete("/api/conversations/{conv_id}")
def api_delete_conversation(conv_id: str):
    delete_conversation(DATA_DIR, conv_id)
    return {"ok": True}


# ---------- 知识库 ----------


@app.get("/api/knowledge-base")
def api_get_knowledge_base():
    store = state["vectorstore"]
    if store is None or store.size == 0:
        return {"size": 0, "sources": []}
    return {"size": store.size, "sources": store.sources}


@app.delete("/api/knowledge-base")
def api_clear_knowledge_base():
    VectorStore.clear(DATA_DIR)
    ImageIndex.clear(DATA_DIR)
    shutil.rmtree(transcripts.transcripts_dir(DATA_DIR), ignore_errors=True)
    shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    state["vectorstore"] = None
    state["image_index"] = None
    return {"ok": True}


@app.delete("/api/knowledge-base/{source:path}")
def api_delete_source(source: str):
    """删掉单个文档，而不是清空整个知识库——传错一份手册时不用把其他几百页重新跑一遍。"""
    store = state["vectorstore"]
    if store is None:
        raise HTTPException(404, "知识库为空")
    removed = store.remove_source(source)
    if not removed:
        raise HTTPException(404, f"知识库中没有《{source}》")
    store.save(DATA_DIR)

    image_index = state["image_index"]
    if image_index is not None and image_index.remove_source(source):
        image_index.save(DATA_DIR)

    transcripts.remove_source(DATA_DIR, source)

    safe_source = re.sub(r"[/\\]", "_", source)
    for path in IMAGES_DIR.glob(f"{glob.escape(safe_source)}_p*.png"):
        path.unlink(missing_ok=True)

    return {"ok": True, "removed": removed, "size": store.size, "sources": store.sources}


@app.post("/api/upload")
def api_upload(files: list[UploadFile]):
    if not DASHSCOPE_API_KEY:
        raise HTTPException(400, "未检测到 DASHSCOPE_API_KEY，请检查项目根目录下的 .env 文件")
    if not DEEPSEEK_API_KEY:
        raise HTTPException(400, "未检测到 DEEPSEEK_API_KEY（用于生成片段上下文），请检查项目根目录下的 .env 文件")
    if not files:
        raise HTTPException(400, "请先上传至少一个 PDF 文件")

    # UploadFile 的底层文件句柄在请求结束后会被关闭，流式响应开始前先读进内存
    file_payloads = [(f.filename, f.file.read()) for f in files]

    def stream() -> Iterator[str]:
        all_pages = []
        all_chunks = []
        for filename, file_bytes in file_payloads:
            # 转录是整个入库流程里最慢的一步（一本几百页的手册要跑很久），必须把逐页进度
            # 透出去，否则前端在这段时间里完全没有反馈。extract_pages 的 on_progress 是从
            # 工作线程同步回调的，没法直接 yield，所以跟下面生成上下文一样用队列转出来。
            page_queue: queue.Queue = queue.Queue()

            def run_extract(file_bytes=file_bytes, filename=filename):
                try:
                    return extract_pages(
                        DASHSCOPE_API_KEY,
                        file_bytes,
                        filename,
                        IMAGES_DIR,
                        on_progress=lambda done, total: page_queue.put((done, total)),
                    )
                finally:
                    page_queue.put(None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_extract)
                while True:
                    item = page_queue.get()
                    if item is None:
                        break
                    done, total = item
                    yield ndjson(
                        {"type": "page_progress", "file": filename, "done": done, "total": total}
                    )
                pages = future.result()

            # 转录先落盘再进下游。这样以后调整切块参数或换嵌入模型时，可以用
            # scripts/rebuild_index.py 直接从这里重建索引，不必重跑视觉模型。
            transcripts.save_pages(DATA_DIR, filename, pages)

            yield ndjson({"type": "file_done", "file": filename, "pages": len(pages)})
            all_pages.extend(pages)
            all_chunks.extend(chunk_pages(pages))

        if not all_chunks:
            yield ndjson({"type": "error", "message": "未能从上传的 PDF 中提取到任何内容，请检查文件是否损坏"})
            return

        # contextualize_chunks 的 on_progress 回调是从工作线程同步调用的，没法直接 yield；
        # 用一个队列把进度事件转出来，在生成器里实时消费。
        progress_queue: queue.Queue = queue.Queue()

        def on_context_progress(done: int, total: int):
            progress_queue.put((done, total))

        def run_contextualize():
            try:
                contextualize_chunks(DEEPSEEK_API_KEY, all_pages, all_chunks, 5, on_context_progress)
            finally:
                progress_queue.put(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run_contextualize)
            while True:
                item = progress_queue.get()
                if item is None:
                    break
                done, total = item
                yield ndjson({"type": "context_progress", "done": done, "total": total})

        try:
            vectors = embed_texts(DASHSCOPE_API_KEY, [c.retrieval_text for c in all_chunks])
        except Exception as e:
            yield ndjson({"type": "error", "message": f"向量化失败：{e}"})
            return

        store = state["vectorstore"]
        if store is None:
            store = VectorStore(dim=vectors.shape[1])

        # 重新上传同名文档时按"替换"处理：旧 chunk 不清掉的话会和新的并存，同一段内容
        # 被检索到两次，还会挤占本该留给其他文档的名额。页图文件名同样按文档名+页码生成，
        # 磁盘上本来就是覆盖写，这里把索引也对齐成同样的语义。
        uploaded_sources = {name for name, _ in file_payloads}
        image_index = state["image_index"]
        image_index_changed = False
        for source in uploaded_sources:
            removed = store.remove_source(source)
            if image_index is not None and image_index.remove_source(source):
                image_index_changed = True
            if removed:
                yield ndjson({"type": "replaced", "file": source, "removed": removed})
        # 下面新页面的向量不一定写得成（嵌入可能整批失败），删除本身必须先落盘，
        # 否则重启后图像检索还会命中上一次入库的旧页面
        if image_index_changed:
            image_index.save(DATA_DIR)

        store.add(vectors, all_chunks)
        store.save(DATA_DIR)
        state["vectorstore"] = store

        # 页面级图像向量：跟文本检索互补，弥补转录文字过于简略时文本检索找不到相关页的问题。
        # 只对有原图的页面做，且每页只嵌入一次（不重复处理同页的多个 chunk）。
        pages_with_image = {(p.source, p.page): p.image_path for p in all_pages if p.image_path}
        if pages_with_image:
            image_progress_queue: queue.Queue = queue.Queue()
            image_vectors: list[list[float] | None] = [None] * len(pages_with_image)
            page_keys = list(pages_with_image.keys())

            def embed_one(i: int) -> tuple[int, list[float] | None]:
                key = page_keys[i]
                path = IMAGES_DIR / pages_with_image[key]
                try:
                    return i, embed_image(DASHSCOPE_API_KEY, path.read_bytes())
                except Exception:
                    return i, None

            def run_image_embedding():
                done = 0
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                        futures = [pool.submit(embed_one, i) for i in range(len(page_keys))]
                        for future in concurrent.futures.as_completed(futures):
                            i, vec = future.result()
                            image_vectors[i] = vec
                            done += 1
                            image_progress_queue.put((done, len(page_keys)))
                finally:
                    image_progress_queue.put(None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(run_image_embedding)
                while True:
                    item = image_progress_queue.get()
                    if item is None:
                        break
                    done, total = item
                    yield ndjson({"type": "image_index_progress", "done": done, "total": total})

            ok_pairs = [(key, vec) for key, vec in zip(page_keys, image_vectors) if vec is not None]
            if ok_pairs:
                keys = [k for k, _ in ok_pairs]
                vecs = np.array([v for _, v in ok_pairs], dtype="float32")
                image_index = state["image_index"]
                if image_index is None:
                    image_index = ImageIndex(dim=vecs.shape[1])
                image_index.add(vecs, keys)
                image_index.save(DATA_DIR)
                state["image_index"] = image_index

        yield ndjson({"type": "complete", "size": store.size, "sources": store.sources})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------- 问答 ----------


class ChatRequest(BaseModel):
    conversation_id: str
    question: str


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not DEEPSEEK_API_KEY or not DASHSCOPE_API_KEY:
        raise HTTPException(400, "未检测到 API Key，请检查项目根目录下的 .env 文件是否配置了 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY")
    store = state["vectorstore"]
    if store is None or store.size == 0:
        raise HTTPException(400, "请先上传并处理 PDF 文档")

    messages = load_conversation(DATA_DIR, req.conversation_id)
    history = recent_turns(messages, n_turns=N_HISTORY_TURNS)
    messages = messages + [{"role": "user", "content": req.question}]

    def stream() -> Iterator[str]:
        search_query, results = retrieve(
            store, state["image_index"], DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, history, req.question
        )

        sources = [
            {
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                "score": score,
                "image_path": chunk.image_path,
            }
            for chunk, score in results
        ]
        yield ndjson({"type": "sources", "sources": sources})

        full_answer = ""
        # 展示给用户的来源仍然只是命中片段，但喂给模型的正文补上相邻片段，
        # 避免被切断的操作步骤或跨片段的表格只给出半截
        context_chunks = store.expand_with_neighbors(results, window=NEIGHBOR_WINDOW)
        for delta in answer_stream(
            DASHSCOPE_API_KEY, history, req.question, results, IMAGES_DIR, context_chunks
        ):
            full_answer += delta
            yield ndjson({"type": "delta", "text": delta})

        final_messages = messages + [
            {"role": "assistant", "content": full_answer, "sources": sources}
        ]
        save_conversation(DATA_DIR, req.conversation_id, final_messages)
        yield ndjson({"type": "done"})

    return StreamingResponse(stream(), media_type="application/x-ndjson")
