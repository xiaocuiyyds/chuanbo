# 项目汇报 PPT 第 2 页去图版实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 保留原 5 页 PPTX，仅删除第 2 页的“3.14 序列过程图像”手册截图并改为全宽信息布局。

**架构：** 复用第一版在仓库外工作区中的 `@oai/artifact-tool` JavaScript 源文件，另存一份去图版生成脚本。删除第 2 页图片与图注，将指标、难点和技术目标扩展至全宽，其他四页代码保持不变。

**技术栈：** JavaScript ES Modules、`@oai/artifact-tool`、PowerPoint `.pptx`、LibreOffice、Poppler。

---

### 任务 1：创建去图版生成源文件

**文件：**
- 读取：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/deck.mjs`
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/deck_no_image.mjs`

- [ ] **步骤 1：运行红灯检查**

运行：

```bash
test -f "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报_去图版.pptx"
```

预期：状态码非 0，因为去图版尚未生成。

- [ ] **步骤 2：复制原生成脚本并更改输出路径**

将 `OUT` 更改为：

```javascript
const OUT = "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报_去图版.pptx";
```

将 `PREVIEW` 更改为独立目录 `tmp/preview-no-image`。

- [ ] **步骤 3：重排第 2 页**

删除 `manual-frame`、`slide.images.add(...)` 和图注。将三个指标均匀分布在 `x=96, 456, 816`，将三个技术难点以全宽水平分隔带排布在指标下方，底部保留技术目标。

### 任务 2：生成并验证去图版 PPTX

**文件：**
- 创建：`/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报_去图版.pptx`
- 创建：`/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/qa/no-image-qa.txt`

- [ ] **步骤 1：运行生成脚本**

```bash
node "/var/folders/mw/hz7yprfd75gc20lzcdqklm800000gn/T/codex-presentations/manual-20260714/ship-knowledge-agent-ppt/tmp/deck_no_image.mjs"
```

预期：生成非空的去图版 PPTX 和 5 张预览图。

- [ ] **步骤 2：逐页渲染检查**

使用 `render_slides.py` 渲染 5 页，以原始分辨率检查。第 2 页必须无手册截图、无“3.14”文字，其他四页的标题、流程和图片保持正常。

- [ ] **步骤 3：运行溢出检测**

```bash
uv run --with python-pptx --with pdf2image --with pillow python "/Users/xiaocui/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/slides_test.py" "/Users/xiaocui/Desktop/school_work/船舶操作知识agent/outputs/船舶操作知识智能问答系统_项目汇报_去图版.pptx"
```

预期：`Test passed. No overflow detected.`

- [ ] **步骤 4：验证页数与输出目录**

使用 LibreOffice 转换为 PDF，并使用 `pdfinfo` 确认页数为 5、页面比例为 16:9。最终 `outputs/` 同时保留原版和去图版 PPTX，不删除 PowerPoint 打开时产生的临时锁文件。

## 计划自检

- 只修改第 2 页，不改动其他四页。
- 不以其他手册截图替换删除的图片。
- 保留原 PPTX，另存去图版。
- 覆盖逐页检查、溢出检测、页数和比例验证。
- 当前目录不是 Git 仓库，因此不包含 commit 步骤。
