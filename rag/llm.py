import base64
import concurrent.futures
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from openai import OpenAI

from rag.chunker import Chunk
from rag.pdf_loader import PageText

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-v4-flash"  # deepseek-chat 已于 2026/07/24 弃用，改名为 deepseek-v4-flash 的兼容别名

# deepseek-v4-flash 默认开思考模式，会先输出一大段推理再给正文。查询改写和片段上下文都是
# 短输出的格式化任务，推理带来的收益抵不上它的开销：思考内容照样计入 max_tokens，几百 token
# 的预算经常在正文开始前就被耗光，接口返回 finish_reason="length" 且 content 为空字符串。
# 关掉之后同一个请求从 ~172 个输出 token 降到 ~22 个，耗时也从 1.4s 降到 0.9s。
DEEPSEEK_NO_THINKING = {"thinking": {"type": "disabled"}}

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VL_MODEL = "qwen3.7-plus"

SYSTEM_PROMPT = (
    "你是船舶设备专家助手，只根据下面提供的手册片段和对应页面图片回答用户的问题。"
    "手册片段是页面的文字转录，可能会遗漏图片里的细节（如接线图标注、仪表读数、指示灯颜色），"
    "文字片段信息不够或存疑时，请直接看图片核实。"
    "如果图片和文字片段都没有足够信息，明确说明手册中未找到相关内容，不要编造。"
    "回答时请注明依据来自哪个文档、第几页。结合之前的对话理解用户的追问。"
)

QUERY_SYSTEM_PROMPT = (
    "你在为一个船舶设备手册检索系统预处理用户提问，手册绝大部分是英文原文。"
    "请输出一个 JSON 对象，包含两个字段：\n"
    '"query"：根据对话历史，把用户最新的问题改写成不依赖上下文、意思完整独立的问题；'
    "如果最新问题已经独立完整就原样返回，保持用户原本使用的语言。\n"
    '"keywords"：上面这个问题在英文手册里会出现的英文检索关键词，用空格分隔；'
    '例如"滑油低压"写成 "lubricating oil low pressure"、"主机"写成 "main engine"；'
    "问题中出现的设备位号、型号、报警代码原样保留，不要翻译。\n"
    "只输出 JSON 本身，不要解释，不要加代码块标记。"
)

SUMMARY_SYSTEM_PROMPT = (
    "下面是一段船员和船舶设备助手的排障对话。请把其中后续追问还会用到的信息压缩成要点，"
    "控制在 150 字以内，用陈述句罗列，不要评价、不要复述完整问答。重点保留：\n"
    "涉及的船舶系统和设备型号；已经确认的事实（设定值、阀件位号、报警代码、当前状态）；"
    "已经排除的可能性；用户明确表达过的目标或约束。\n"
    "对话里没提到的不要补充。只输出要点本身。"
)

CONTEXT_SYSTEM_PROMPT = (
    "你会看到一整页船舶设备手册的内容，以及从这页里切出来的一个片段。"
    "请用一到两句话简要说明这个片段在整页中的位置、属于哪个章节或主题、涉及什么设备或功能，"
    "帮助后续做检索时能识别这个片段的上下文（例如片段里出现的编号、站点、部件属于哪个系统）。"
    "只输出这段说明本身，不要复述片段原文，不要解释，不要加引号。"
)


def _deepseek_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _vl_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def build_context(chunks: list[Chunk]) -> str:
    """把片段拼成给回答模型的上下文。同一页连续的多个片段合并成一块，
    这样跨片段被切开的操作步骤或表格能连起来读，也少重复几次来源标题。"""
    parts: list[str] = []
    current_key: tuple[str, int] | None = None
    current_texts: list[str] = []

    def flush() -> None:
        if current_key and current_texts:
            source, page = current_key
            parts.append(f"[来源: {source} 第{page}页]\n" + "\n\n".join(current_texts))

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key != current_key:
            flush()
            current_key = key
            current_texts = []
        current_texts.append(chunk.text)
    flush()
    return "\n\n".join(parts)


def _unique_page_image_urls(results: list[tuple[Chunk, float]], images_dir: Path) -> list[str]:
    """取命中片段对应的原始页面图（按 source+page 去重），编码成 data URI 供多模态模型查看。"""
    seen: set[tuple[str, int]] = set()
    data_urls = []
    for chunk, _ in results:
        key = (chunk.source, chunk.page)
        if not chunk.image_path or key in seen:
            continue
        path = images_dir / chunk.image_path
        if not path.exists():
            continue
        seen.add(key)
        b64 = base64.b64encode(path.read_bytes()).decode()
        data_urls.append(f"data:image/png;base64,{b64}")
    return data_urls


def prepare_query(api_key: str, history: list[dict], question: str) -> tuple[str, str]:
    """把用户提问处理成 (语义检索用的问题, BM25 用的关键词串)。

    两件事一次调用完成：一是结合最近几轮对话，把带代词/省略的追问改写成独立完整的问题；
    二是给出对应的英文关键词。

    英文关键词是必要的——手册里 95% 的内容是英文原文（四本 Prime Purity 手册的中文占比
    只有 0.2%~0.8%），而用户基本用中文提问。jieba 把中文问题切出来的词在英文手册里一个
    都不存在，BM25 那一路对这几本手册完全没有信号，只在少数几本带中文的手册上有分，
    反而会把 RRF 融合的结果系统性地拉向那几本。把中英文拼在一起喂给 BM25，
    两种语言的手册就都能被关键词通道命中。

    调用或解析失败时退回原问题（等价于改造前的行为），不影响检索正常进行。
    """
    client = _deepseek_client(api_key)
    messages = [{"role": "system", "content": QUERY_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=False,
            max_tokens=300,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body=DEEPSEEK_NO_THINKING,
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return question, question

    search_query = str(data.get("query") or "").strip() or question
    keywords = str(data.get("keywords") or "").strip()
    # BM25 同时拿到原问题和英文关键词，中文手册和英文手册都能命中
    keyword_query = f"{search_query} {keywords}".strip() if keywords else search_query
    return search_query, keyword_query


def summarize_session(api_key: str, messages: list[dict]) -> str:
    """把一段对话压缩成后续追问还用得上的要点；失败时返回空串。

    只喂最近三轮给模型，意味着对话稍长一点，开头确立的前提就被切掉了：第一轮说明了
    "主机是 MAN B&W ME-B 配 AutoChief 600"，到第五轮问"那个报警怎么复位"时，
    查询改写已经看不到那句话，没法把"那个报警"正确展开。排障本身就是多轮的，
    这类前提必须跨轮保留。

    摘要只保留事实性要点（设备型号、设定值、已排除的可能），不保留完整问答——
    后者会把上下文重新撑大，失去压缩的意义。
    """
    if not messages:
        return ""

    transcript = "\n\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:600]}" for m in messages
    )
    client = _deepseek_client(api_key)
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            stream=False,
            max_tokens=400,
            temperature=0,
            extra_body=DEEPSEEK_NO_THINKING,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def generate_chunk_context(api_key: str, page_text: str, chunk_text: str) -> str:
    """给一个 chunk 生成定位性上下文（Contextual Retrieval），用整页内容作为背景。"""
    client = _deepseek_client(api_key)
    user_content = f"整页内容：\n{page_text}\n\n片段内容：\n{chunk_text}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=False,
        max_tokens=200,
        temperature=0,
        extra_body=DEEPSEEK_NO_THINKING,
    )
    return (response.choices[0].message.content or "").strip()


def contextualize_chunks(
    api_key: str,
    pages: list[PageText],
    chunks: list[Chunk],
    max_workers: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """给每个 chunk 生成上下文并写入 chunk.context，原地修改，不改变 chunk.text。
    单个 chunk 生成失败时跳过（保留 context=None），不影响其余片段和整体入库流程。
    """
    page_text_by_key = {(p.source, p.page): p.text for p in pages}

    def worker(index: int) -> tuple[int, str]:
        chunk = chunks[index]
        page_text = page_text_by_key.get((chunk.source, chunk.page), chunk.text)
        try:
            context = generate_chunk_context(api_key, page_text, chunk.text)
        except Exception:
            context = ""
        return index, context

    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(len(chunks))]
        for future in concurrent.futures.as_completed(futures):
            index, context = future.result()
            chunks[index].context = context or None
            done_count += 1
            if on_progress:
                on_progress(done_count, len(chunks))


def answer_stream(
    dashscope_api_key: str,
    history: list[dict],
    question: str,
    results: list[tuple[Chunk, float]],
    images_dir: Path,
    context_chunks: list[Chunk] | None = None,
) -> Iterator[str]:
    """results 决定引用哪些来源和回看哪几页原图；context_chunks 是实际喂给模型的正文，
    默认就是命中片段本身，调用方可以传入扩展了相邻片段的版本来补全被切断的步骤和表格。"""
    context = build_context(context_chunks if context_chunks is not None else [c for c, _ in results])
    image_urls = _unique_page_image_urls(results, images_dir)

    content: list[dict] = [{"type": "text", "text": f"手册片段：\n{context}\n\n用户问题：{question}"}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": content})

    client = _vl_client(dashscope_api_key)
    stream = client.chat.completions.create(
        model=VL_MODEL,
        messages=messages,
        stream=True,
        # 关掉思考模式：答案完全来自上面给定的片段和页图，属于接地式问答，不需要模型先自行推理。
        # 实测同一问题首 token 从 ~12.8s 降到 ~1.7s，而检索全链路本身只占 0.9s。
        extra_body={"enable_thinking": False},
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
