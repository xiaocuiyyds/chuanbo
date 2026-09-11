# 船舶操作知识智能问答系统 PPT 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 生成一份 5 页、16:9、可编辑的中文项目汇报 PPTX，以技术架构为主线讲解船舶操作知识智能问答系统。

**架构：** 在仓库外的临时工作区中使用 `@oai/artifact-tool` 编写纯 JavaScript ES 模块，使用“工程白板”风格组合文字、真实手册截图和两张简化技术流程图。导出后使用 LibreOffice/Poppler 链路逐页渲染，并运行溢出检测。

**技术栈：** JavaScript ES Modules、`@oai/artifact-tool`、PowerPoint `.pptx`、LibreOffice、Poppler、项目内 PNG 素材。

---

## 文件结构

- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/source-notes.txt`，记录项目事实、页数、片段数、问答摘要和素材来源。
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/deck.mjs`，定义五页演示文稿的样式、布局和内容。
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/qa/qa-ledger.txt`，记录逐页视觉检查结果和修改。
- 创建：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx`，最终交付文件。

### 任务 1：建立演示文稿工作区与内容依据

**文件：**
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/source-notes.txt`
- 读取：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/app.py`
- 读取：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/rag/*.py`
- 读取：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/data/chunks.pkl`
- 读取：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/data/chat_history.json`

- [ ] **步骤 1：创建仓库外工作区**

运行：

```bash
mkdir -p "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/qa" "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs"
node "/Users/xiaocui/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/setup_artifact_tool_workspace.mjs" --workspace "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp"
```

预期：工作区内可解析 `@oai/artifact-tool`，项目中出现 `outputs/` 目录。

- [ ] **步骤 2：固化当前项目事实**

在 `source-notes.txt` 中写入以下已验证数据：

```text
KC-700 Manual.pdf: 171 pages
KONGSBERG AC600_MEB_FPP +(翻译结果).pdf: 112 pages
Current vector store: 504 chunks
Chunk split: target 800, max 1600, overlap 100
Retrieval: normalized inner product, top_k=5
Conversation context: most recent 3 turns
Models: qwen-vl-plus, text-embedding-v3, deepseek-chat
```

- [ ] **步骤 3：验证素材存在**

运行：

```bash
test -f "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/ocr_samples/KC-700 Manual_p40.png"
test -f "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/ocr_samples/KONGSBERG AC600_MEB_FPP +(翻译结果)_p90.png"
```

预期：两条命令都以状态码 0 结束。

### 任务 2：实现五页 PPTX

**文件：**
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/deck.mjs`
- 创建：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx`

- [ ] **步骤 1：在 `deck.mjs` 中定义全局主题**

实现内容：16:9 画布；白底；主色 `#174A78`；强调色 `#00A8B5`；浅灰线 `#DCE4EA`；标题不小于 35pt；正文不小于 16pt。

- [ ] **步骤 2：实现封面和背景页**

第 1 页使用大标题、副标题与“PDF → 多模态解析 → 向量检索 → 可溯源答案”简化流程。第 2 页使用真实手册图片和 `2 / 283 / 504` 三个关键数字。

- [ ] **步骤 3：实现两张技术链路页**

第 3 页展示入库链路与 `180 DPI / 800 / 1600 / 100` 参数。第 4 页展示最近 3 轮历史、问题改写、Top 5 检索、上下文生成和来源溯源。

- [ ] **步骤 4：实现成果与路线页**

第 5 页展示一条 AutoChief 600 指示面板问答摘要，以及“已实现 / 下一步”两类信息。下一步仅保留评测集、混合检索、重排序三项。

- [ ] **步骤 5：运行生成模块**

运行：

```bash
node "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/deck.mjs"
```

预期：生成文件 `/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx`，且文件大小大于 50 KB。

### 任务 3：渲染并逐页检查

**文件：**
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/qa/qa-ledger.txt`
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/preview/slide-1.png` 至 `slide-5.png`

- [ ] **步骤 1：渲染全部页面**

运行：

```bash
uv run --with pdf2image --with pillow python "/Users/xiaocui/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/render_slides.py" "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx" --output_dir "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/preview"
```

预期：输出 5 张 PNG。如工具不支持 `--output_dir`，则在临时工作区复制 PPTX 后按默认路径渲染。

- [ ] **步骤 2：检查每张单页图像**

使用图像查看工具以原始分辨率查看 5 张 PNG，在 `qa-ledger.txt` 中逐页记录：标题是否换行、正文是否溢出、图片是否裁切异常、流程线是否穿越文字、页码是否一致。

- [ ] **步骤 3：修正视觉问题并重新渲染**

每次只修改 `deck.mjs` 中与发现问题对应的元素，重新运行生成与渲染命令，直到 `qa-ledger.txt` 的 5 页均为 PASS。

### 任务 4：自动化验证与交付

**文件：**
- 测试：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx`

- [ ] **步骤 1：运行溢出检测**

运行：

```bash
uv run --with python-pptx python "/Users/xiaocui/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/slides_test.py" "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx"
```

预期：返回状态码 0，不报告画布外元素。

- [ ] **步骤 2：验证文件可打开且页数正确**

运行：

```bash
soffice --headless --convert-to pdf --outdir "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/qa" "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx"
pdfinfo "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/qa/船舶操作知识智能问答系统_项目汇报.pdf" | rg '^Pages:\s+5$'
```

预：LibreOffice 转换成功，PDF 页数为 5。

- [ ] **步骤 3：检查最终交付物**

运行：

```bash
test -s "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报.pptx"
```

预期：状态码 0。

## 计划自检

- 覆盖封面、项目背景、入库链路、RAG 架构、成果与后续优化五页内容。
- 覆盖“工程白板”风格、项目真实素材、可编辑 PPTX 与逐页渲染验收。
- 当前目录不是 Git 仓库，因此本计划不包含无法执行的 commit 步骤。
