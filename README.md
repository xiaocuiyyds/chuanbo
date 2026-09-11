# 船舶操作知识问答系统

面向船舶设备手册的多模态 RAG 问答系统。上传设备手册 PDF，用自然语言提问，系统给出带文档名和页码引用的回答，并附上原始页面图片。

后端 FastAPI，前端 Vue 3 + Vite，向量检索用 FAISS。

## 它解决的问题

船舶设备手册有几个特点让通用 RAG 不太好用：

- **内容主要在图里**。接线图、管路图、报警代码表承载了大部分有用信息，纯文本提取会丢掉它们。所以这里不做文本提取，而是把每一页渲染成图片交给视觉模型转录成 Markdown，回答时再把原始页图一并交给多模态模型核对。
- **手册是英文，提问是中文**。本项目索引的六份手册里，四份 Prime Purity 手册的中文占比只有 0.2%~0.8%。中文问题经 jieba 分词后得到的词在英文手册里一个都不存在，BM25 通道会完全失效。所以查询预处理会额外产出一份英文关键词，让关键词检索对两种语言的手册都有信号。
- **答案常常跨片段**。"操作步骤 1-12" 被切成两块时，只把命中的那块交给模型会给出半截流程——对照着手册操作的人来说这比查不到更危险。所以命中后会把相邻片段一并带上。

## 流程

### 入库

```
PDF ─→ 逐页渲染 PNG (180 DPI)
     ─→ Qwen-VL 转录为 Markdown（表格转表格，示意图给文字描述）
     ─→ 按标题/表格边界切块（目标 800 字符，大表格按行拆分并重复表头）
     ─→ 用整页作背景，为每个片段生成定位性上下文（Contextual Retrieval）
     ─→ 文本向量 → index.faiss
     └─→ 页面图像向量 → image_index.faiss
```

转录阶段会检测两类失败并自动换更严格的 prompt 重试一次：撞上 `max_tokens` 被截断，以及模型放弃读图、改成把同一句套话套到几十个位号上。

### 检索与回答

```
问题 ─→ 查询改写 + 英文关键词（一次 LLM 调用同时产出）
     ─→ 向量检索 ┐
        BM25    ├─→ RRF 融合 ─→ qwen3-rerank 精排 ─→ top 5
        图像检索 ┘
     ─→ 取相邻片段补全被切断的步骤和表格
     ─→ 片段原文 + 命中页原图 → 多模态模型流式作答
```

图像检索通道是为了弥补转录文字过于简略的页面——这类页面文本信号弱，纯文本检索找不到。但 reranker 本身是文本模型，会因为同样的原因把它们判为不相关，所以图像检索里把握最大的那一页会被强制保留一个位置。

## 用到的模型

| 用途 | 模型 | 服务 |
|---|---|---|
| 页面转录 | `qwen-vl-plus` | DashScope |
| 文本向量 | `text-embedding-v3` | DashScope |
| 图像向量 | `qwen3-vl-embedding` | DashScope |
| 重排 | `qwen3-rerank` | DashScope |
| 回答 | `qwen3.7-plus` | DashScope |
| 查询改写 / 片段上下文 | `deepseek-v4-flash` | DeepSeek |

两个回答类模型都关掉了思考模式。接地式问答的答案完全来自给定片段，推理带来的收益抵不上开销——实测首 token 从约 12.8s 降到约 1.7s。DeepSeek 那边尤其要注意：思考内容同样计入 `max_tokens`，几百 token 的预算经常在正文开始前就被耗光，接口会返回 `finish_reason="length"` 且内容为空字符串。

注意两家的参数写法不同，不能混用：

- DashScope：`extra_body={"enable_thinking": False}`
- DeepSeek：`extra_body={"thinking": {"type": "disabled"}}`

## 安装

代码用到 `X | None` 这类注解语法，需要 Python 3.10 以上；开发与测试环境是 Python 3.13.13 + Node 24.15.0，其他版本没有验证过。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install --prefix frontend
```

在项目根目录建 `.env`（可以从 `.env.example` 复制）：

```
DEEPSEEK_API_KEY=你的key
DASHSCOPE_API_KEY=你的key
```

## 运行

两个终端分别启动：

```bash
.venv/bin/uvicorn server.main:app --port 8000
```

```bash
npm run dev --prefix frontend
```

打开 http://localhost:5173。Vite 会把 `/api` 和 `/images` 代理到 8000 端口。

## 首次使用

知识库数据都在 `data/` 下，这个目录不入版本库，所以克隆下来之后是空的，需要先建库：

1. 打开页面，在左侧「上传手册 PDF」选择一个或多个 PDF
2. 点「处理文档」

处理过程会实时显示进度：转录页数 → 生成片段上下文 → 建立图像索引。**这一步很慢且要花钱**——每一页都要调一次视觉模型，每个片段再调一次文本模型。作为参考，本项目索引的六份手册共 837 页，转录加建索引跑了较长时间。建议先用一份小手册跑通流程。

建好之后 `data/` 下会有：

```
data/
├── index.faiss          文本向量索引
├── chunks.pkl           片段内容与元信息
├── image_index.faiss    页面图像向量索引
├── image_pages.json     图像索引对应的 (文档, 页码)
├── page_images/         每页的渲染图，回答时交给多模态模型
└── conversations/       对话历史，一个对话一个 JSON
```

`page_images/` 会比较大（837 页约 400MB）。

## 知识库管理

- 重复上传同名文件会**替换**原有内容，不会重复入库
- 侧边栏可以按文档单独删除，不必清空整个知识库
- 「清空知识库」会删掉全部索引和页图

## 维护脚本

两个脚本都支持 `--dry-run`（只统计不调用接口）和 `--limit N`（先试 N 条），并且在改动前自动备份 `chunks.pkl` 和 `index.faiss`。

### `scripts/repair_contexts.py`

重新生成被截断或为空的片段上下文，并同步更新向量索引。

片段上下文会进入检索文本，所以重新生成后必须重算向量，否则索引里的向量和内容对不上——脚本已经处理了这一点。

```bash
PYTHONPATH=. .venv/bin/python scripts/repair_contexts.py --dry-run
PYTHONPATH=. .venv/bin/python scripts/repair_contexts.py
```

### `scripts/repair_pages.py`

检出退化的页面并重新转录，然后替换索引里对应的片段。

退化是偶发的——同一页同一 prompt 重跑一次通常就正常了，所以重转录本身就能解决大部分问题。重转之后仍然退化的页面会保留原样，不会越修越差。

```bash
PYTHONPATH=. .venv/bin/python scripts/repair_pages.py --dry-run
PYTHONPATH=. .venv/bin/python scripts/repair_pages.py
```

两个脚本都会改写同一份 `chunks.pkl` 和 `index.faiss`，**不要同时运行**。

## 已知问题

- **没有鉴权，CORS 开放 `*`**。目前只适合本地或可信内网使用。
- **单进程内存状态，没有加锁**。上传时重建 BM25 索引与查询检索会竞争同一个 `VectorStore`，多人同时使用可能出问题。
- **没有"停止生成"**。客户端断开后，后端仍会把回答生成完并存入历史。
- **异常大多被静默吞掉，且没有日志**。API 余额不足这类问题会表现为"某些片段没有上下文"，不会直接报错。
- **没有测试**。
- **整本 PDF 的页面图会一次性读进内存**。几百页的手册占用可观，上千页可能撑爆内存。

## 目录结构

```
rag/
├── pdf_loader.py    页面渲染、VL 转录、退化检测与重试
├── chunker.py       按结构切块，表格不从中间断开
├── llm.py           查询预处理、片段上下文生成、流式作答
├── embeddings.py    文本向量
├── image_index.py   页面图像向量索引
├── vectorstore.py   FAISS + BM25，RRF 融合，邻接扩展
├── reranker.py      重排
└── history.py       对话历史存取
server/main.py       FastAPI 接口
frontend/src/        Vue 3 前端
scripts/             索引维护脚本
docs/                设计文档
```
