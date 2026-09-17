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
    load_summary,
    new_conversation_id,
    save_conversation,
    save_summary,
    session_history,
)
from rag.image_index import ImageIndex, embed_image, embed_text_query
from rag.llm import answer_stream, contextualize_chunks, prepare_query, summarize_session
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
TOP_K = 5  # 最终交给回答模型的片段数
EXACT_MATCH_SLOTS = 1  # 查询里出现位号/编号时，给精确匹配保留的名额


def _rerank_top(dashscope_key, query, pool, n):
    """用 reranker 从候选里挑出最相关的 n 条；调用失败时退回 RRF 的原始排序。"""
    if not pool or n <= 0:
        return []
    try:
        reranked = rerank(dashscope_key, query, [c.retrieval_text for c, _ in pool], top_n=n)
        return [(pool[idx][0], score) for idx, score in reranked]
    except Exception:
        return pool[:n]


def _retrieve_pool(store, image_index, dashscope_key, search_query, keyword_query):
    """跑一轮三路检索，返回 (候选列表, 图像检索最有把握的那一页)。

    向量和 BM25 两路分数量纲不同，直接加权很难调；RRF 只看排名，对报警代码/型号这类
    关键词能命中、语义检索容易漏的情况更稳健。图像检索通道用来弥补转录文字过于简略、
    文本检索找不到的页面。
    """
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
    return pool, top_image_page


def retrieve(
    store,
    image_index,
    dashscope_key,
    deepseek_key,
    history,
    question,
    use_rerank=True,
):
    """查询改写 -> 三路 RRF 融合 ->（可选）补充检索 -> reranker 精排，返回最终 top5。

    曾经尝试过在第一轮之后让模型判断"信息够不够、还缺什么"再补检一轮，实测是净负收益：
    复合题的页面覆盖率一点没涨（补充查询往往复述了已经覆盖的那一面，而不是真正缺的那一面），
    延迟却翻倍，给补充轮预留名额之后还把单点问题的 recall@5 从 96% 拖到 92%。
    详见 evals/retrieval.jsonl 里的复合题。

    reranker 不关心候选是哪一路、哪一跳召回的，只看内容跟问题的真实相关性重新打分，
    能纠正 RRF 纯按排名求和带来的排序失真；调用失败时退回 RRF 原始排序。

    但 reranker 本身是纯文本模型，只看片段的转录文字——如果一个页面正是因为转录文字
    过于简略才需要靠图像检索召回，reranker 反而会因为文字信号弱把它判定为不相关、
    挤出结果，等于抵消了图像检索通道的作用。所以图像检索里把握最大的那一页（分数达到
    阈值）会被强制保留一个位置，不受 reranker 意见影响。
    """
    search_query, keyword_query = prepare_query(deepseek_key, history, question)
    pool, top_image_page = _retrieve_pool(store, image_index, dashscope_key, search_query, keyword_query)

    results = _rerank_top(dashscope_key, search_query, pool, TOP_K) if use_rerank else pool[:TOP_K]

    # 位号、报警代码这类标识符对向量检索是无意义的随机串，只有 BM25 能命中；而 RRF 按排名
    # 求和，只有一路有分的候选会被"三路都沾点边"的候选压下去。实测查"GFV21 这个阀在什么
    # 位置"，BM25 把正确的图纸记录排在第 2，它却连候选池都进不去，最终完全丢失。
    #
    # 而且光靠 BM25 排名救不回来：图纸片段要列出整页几十个位号（p112 有 51 个），
    # 单个位号只占全片段 1/136 的词频，怎么调片段结构都会被稀释——实测把位号单独拆成
    # 短片段，BM25 排名也只从 93 升到 37，仍在候选池之外。
    #
    # 所以精确匹配不参与排序，直接置顶：用户既然输了一个完整的标识符，
    # 原样含有它的那一页就是最直接的答案，不该跟语义相关度比分数。
    exact = [c for c in store.exact_token_matches(keyword_query, limit=EXACT_MATCH_SLOTS)]
    if exact:
        rest = [item for item in results if not any(item[0] is c for c in exact)]
        results = [(c, 1.0) for c in exact] + rest[: TOP_K - len(exact)]

    if top_image_page and not any((c.source, c.page) == top_image_page for c, _ in results):
        rescue = next((item for item in pool if (item[0].source, item[0].page) == top_image_page), None)
        if rescue:
            results = results[: TOP_K - 1] + [rescue]

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

    # 会话摘要承载最近几轮之外的前提。排障对话到第五六轮时，第一轮确立的设备型号、
    # 已确认的设定值、已排除的可能都已经滑出窗口，而追问往往正依赖这些。
    summary, summary_upto = load_summary(DATA_DIR, req.conversation_id)
    kept_from = max(0, len(messages) - N_HISTORY_TURNS * 2)
    if kept_from > summary_upto:
        # 只对"即将滑出窗口"的那部分重新摘要，窗口内的消息本来就会原样喂进去
        new_summary = summarize_session(DEEPSEEK_API_KEY, messages[:kept_from])
        if new_summary:
            summary, summary_upto = new_summary, kept_from
            save_summary(DATA_DIR, req.conversation_id, summary, summary_upto)

    history = session_history(messages, summary, n_turns=N_HISTORY_TURNS)
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
