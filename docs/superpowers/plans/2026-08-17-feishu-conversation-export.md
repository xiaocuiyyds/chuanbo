# 飞书云文档导出当前会话实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用户在飞书客户端中点击按钮后，将当前聊天会话一键导出为自己云空间中的新飞书 Docx 文档，而不改变既有聊天、会话或知识库行为。

**架构：** 前端仅获取飞书网页应用的一次性授权码并调用新后端接口。后端的新 `rag/feishu_docs.py` 在请求时使用 `.env` 中的应用凭证换取 `user_access_token`，创建 Docx 文档、分批创建纯文本块并返回文档 URL；`server/main.py` 保持现有路由协议不变，只增加独立导出路由。

**技术栈：** Python 3、FastAPI、Pydantic、标准库 `urllib.request`、pytest、Vue 3、Vite、飞书 Docx OpenAPI 与 H5 JS SDK。

---

## 文件结构

- 创建：`rag/feishu_docs.py`——飞书鉴权、文档块构建、Docx 创建与写入；无 FastAPI 依赖。
- 创建：`tests/test_feishu_docs.py`——飞书服务的单元测试；所有 HTTP 调用均使用 mock。
- 创建：`tests/test_feishu_export_api.py`——导出路由和既有聊天路由的回归测试。
- 修改：`server/main.py`——新增一个请求模型和 `POST /api/conversations/{conv_id}/feishu-doc` 路由；既有接口和 `/api/chat` 不改动。
- 修改：`requirements.txt`——加入测试依赖 `pytest` 与 `httpx`。
- 修改：`frontend/src/api.js`——新增单独的导出请求函数。
- 修改：`frontend/src/components/ChatView.vue`——新增导出按钮、飞书授权码获取、独立导出状态和文档链接。
- 修改：`.env.example`——记录飞书应用所需环境变量，不写入真实密钥。
- 修改：`README.md`——补充开发者后台权限、环境变量和本地联调说明。

## 任务 1：建立飞书文档服务的失败测试

**文件：**
- 创建：`tests/test_feishu_docs.py`
- 修改：`requirements.txt`

- [ ] **步骤 1：添加测试运行依赖**

在 `requirements.txt` 末尾加入：

```text
pytest
httpx
```

- [ ] **步骤 2：写出文档块构建的失败测试**

创建 `tests/test_feishu_docs.py`：

```python
from rag.feishu_docs import build_conversation_blocks


def test_build_conversation_blocks_keeps_turn_order_and_sources():
    messages = [
        {"role": "user", "content": "主机报警怎么办？"},
        {
            "role": "assistant",
            "content": "先确认报警代码。",
            "sources": [{"source": "KC-700 Manual.pdf", "page": 12, "score": 0.8, "text": "报警处理步骤"}],
        },
    ]

    blocks = build_conversation_blocks(messages, "2026-08-17 10:00")

    text_runs = [block[next(k for k in ("text", "heading1", "heading2") if k in block)]["elements"][0]["text_run"]["content"] for block in blocks]
    assert text_runs == [
        "船舶操作知识问答",
        "导出时间：2026-08-17 10:00",
        "问题",
        "主机报警怎么办？",
        "回答",
        "先确认报警代码。",
        "参考来源",
        "KC-700 Manual.pdf 第 12 页（相关度 0.800）\n报警处理步骤",
    ]
    assert [block["block_type"] for block in blocks] == [3, 2, 4, 2, 4, 2, 4, 2]


def test_build_conversation_blocks_rejects_empty_messages():
    import pytest
    from rag.feishu_docs import FeishuDocumentError

    with pytest.raises(FeishuDocumentError, match="暂无可导出的内容"):
        build_conversation_blocks([], "2026-08-17 10:00")
```

- [ ] **步骤 3：运行测试，确认按“模块不存在”失败**

运行：

```bash
.venv/bin/python -m pytest tests/test_feishu_docs.py -v
```

预期：测试收集失败，提示 `ModuleNotFoundError: No module named 'rag.feishu_docs'`。

## 任务 2：实现最小飞书文档服务

**文件：**
- 创建：`rag/feishu_docs.py`
- 测试：`tests/test_feishu_docs.py`

- [ ] **步骤 1：实现领域异常和安全配置读取**

创建以下基础代码；任何一个环境变量缺失均在请求时抛出明确错误，不在导入模块或 FastAPI 启动阶段请求飞书：

```python
import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuDocumentError(Exception):
    pass


def _require_credentials() -> tuple[str, str]:
    app_id = os.getenv("FEISHU_APP_ID", "")
    app_secret = os.getenv("FEISHU_APP_SECRET", "")
    if not app_id or not app_secret:
        raise FeishuDocumentError("尚未配置飞书应用凭证，请联系管理员")
    return app_id, app_secret
```

- [ ] **步骤 2：实现可测试的 JSON POST 封装和用户令牌交换**

实现 `_post_json(url, payload, authorization=None)`，使用 `urllib.request.Request` 发送 `application/json; charset=utf-8`，只将 JSON 响应返回为字典。将 HTTP、网络、非 JSON 响应和飞书返回 `code != 0` 统一转换为 `FeishuDocumentError("飞书服务请求失败，请稍后重试")`，日志不得包含令牌。

实现：

```python
def exchange_user_access_token(authorization_code: str) -> str:
    if not authorization_code.strip():
        raise FeishuDocumentError("未完成飞书授权，请重试")
    app_id, app_secret = _require_credentials()
    response = _post_json(
        f"{FEISHU_API_BASE}/authen/v2/oauth/token",
        {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "client_id": app_id,
            "client_secret": app_secret,
        },
    )
    token = response.get("access_token")
    if not token:
        raise FeishuDocumentError("未完成飞书授权，请重试")
    return token
```

若飞书最新授权码换取接口的字段与此处不同，以开发者后台 API 调试台生成的请求为准，并同步更新测试 mock。

- [ ] **步骤 3：实现纯文本块构建**

实现 `build_conversation_blocks(messages, exported_at)`，严格只读取 `role`、`content` 和可选的 `sources`。使用下列工厂函数，避免将 Markdown/HTML 直接交给飞书：

```python
def _text_block(content: str) -> dict:
    return {
        "block_type": 2,
        "text": {"elements": [{"text_run": {"content": content}}]},
    }


def _heading_block(content: str, level: int) -> dict:
    key = "heading1" if level == 1 else "heading2"
    return {
        "block_type": 3 if level == 1 else 4,
        key: {"elements": [{"text_run": {"content": content}}]},
    }
```

第一个块为一级标题，问题/回答/参考来源为二级标题；来源正文使用 `source`、`page`、`score`、`text` 的安全字符串表示，缺失字段以空字符串或“未知”兜底。

- [ ] **步骤 4：扩展失败测试到 HTTP 调用顺序**

在 `tests/test_feishu_docs.py` 增加：

```python
from unittest.mock import patch
from rag.feishu_docs import export_conversation


@patch("rag.feishu_docs._post_json")
def test_export_conversation_creates_document_then_appends_blocks(post_json, monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    post_json.side_effect = [
        {"access_token": "u-token"},
        {"data": {"document": {"document_id": "dox_test"}}},
        {"data": {"children": []}},
    ]

    result = export_conversation(
        title="靠泊检查",
        messages=[{"role": "user", "content": "检查什么？"}],
        authorization_code="auth-code",
        exported_at="2026-08-17 10:00",
    )

    assert result == {"document_id": "dox_test", "url": "https://feishu.cn/docx/dox_test"}
    assert post_json.call_args_list[1].args[0].endswith("/docx/v1/documents")
    assert post_json.call_args_list[2].args[0].endswith("/docx/v1/documents/dox_test/blocks/dox_test/children")
```

- [ ] **步骤 5：实现文档创建与分批写入**

实现 `export_conversation(title, messages, authorization_code, exported_at=None)`：

1. 调用 `exchange_user_access_token`。
2. 以 `POST /docx/v1/documents` 创建标题为 `船舶操作知识问答 - <title> - <timestamp>` 的文档。
3. 从响应读取 `data.document.document_id`；缺失时抛出 `FeishuDocumentError`。
4. 对根块调用 `POST /docx/v1/documents/{document_id}/blocks/{document_id}/children`，每次最多发送 50 个 `children`，并带 `Authorization: Bearer <user_access_token>`。
5. 返回 `{"document_id": document_id, "url": f"https://feishu.cn/docx/{document_id}"}`。

- [ ] **步骤 6：运行服务单元测试，确认转绿**

运行：

```bash
.venv/bin/python -m pytest tests/test_feishu_docs.py -v
```

预期：所有 `test_feishu_docs.py` 测试通过。

## 任务 3：为导出路由补充失败测试并接入 FastAPI

**文件：**
- 创建：`tests/test_feishu_export_api.py`
- 修改：`server/main.py`
- 测试：`tests/test_feishu_export_api.py`

- [ ] **步骤 1：写出路由失败测试**

创建 `tests/test_feishu_export_api.py`。为避免导入时加载真实向量索引，先 mock `VectorStore.load` 和 `ImageIndex.load`，然后断言：

```python
from fastapi.testclient import TestClient
import pytest

import server.main as main
from rag.history import load_conversation, save_conversation


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    return TestClient(main.app)


def test_export_rejects_empty_conversation(client):
    response = client.post("/api/conversations/missing/feishu-doc", json={"authorization_code": "code"})
    assert response.status_code == 400
    assert response.json()["detail"] == "当前会话暂无可导出的内容"


def test_export_returns_document_url_without_mutating_messages(client, monkeypatch, tmp_path):
    original_messages = [{"role": "user", "content": "靠泊前检查什么？"}]
    save_conversation(tmp_path, "conv1", original_messages)
    monkeypatch.setattr(
        main,
        "export_conversation",
        lambda **kwargs: {"document_id": "dox_test", "url": "https://feishu.cn/docx/dox_test"},
    )
    response = client.post(
        "/api/conversations/conv1/feishu-doc",
        json={"authorization_code": "code"},
    )
    assert response.status_code == 200
    assert response.json()["url"] == "https://feishu.cn/docx/dox_test"
    assert load_conversation(tmp_path, "conv1") == original_messages
```

再新增 `/api/chat` 回归测试，mock `answer_stream`、`retrieve` 与 API key，断言现有 NDJSON 响应仍以 `sources`、`delta`、`done` 顺序返回。

- [ ] **步骤 2：运行路由测试，确认按“路由不存在”失败**

运行：

```bash
.venv/bin/python -m pytest tests/test_feishu_export_api.py -v
```

预期：导出测试因 HTTP 404 失败；聊天回归测试在未改动生产代码时通过。

- [ ] **步骤 3：添加最小请求模型和路由**

在 `server/main.py` 的对话历史路由之后添加：

```python
class FeishuExportRequest(BaseModel):
    authorization_code: str


@app.post("/api/conversations/{conv_id}/feishu-doc")
def api_export_conversation_to_feishu(conv_id: str, req: FeishuExportRequest):
    messages = load_conversation(DATA_DIR, conv_id)
    if not messages:
        raise HTTPException(400, "当前会话暂无可导出的内容")
    try:
        return export_conversation(
            title=_derive_title(messages),
            messages=messages,
            authorization_code=req.authorization_code,
        )
    except FeishuDocumentError as exc:
        status_code = 400 if "授权" in str(exc) else 502
        raise HTTPException(status_code, str(exc)) from exc
```

不要调用或修改 `api_chat`、`save_conversation`、`recent_turns`、知识库路由，也不要在启动时验证飞书配置。由于 `_derive_title` 当前是私有函数，任务实施时将其改名为公开的 `derive_conversation_title` 并在 `save_conversation` 和新路由中复用；为该改名增加历史模块单测。

- [ ] **步骤 4：运行路由与聊天回归测试，确认转绿**

运行：

```bash
.venv/bin/python -m pytest tests/test_feishu_export_api.py -v
```

预期：导出成功/失败场景全部通过，且聊天流式回归测试通过。

## 任务 4：为前端导出交互编写构建前的失败检查

**文件：**
- 修改：`frontend/src/api.js`
- 修改：`frontend/src/components/ChatView.vue`

- [ ] **步骤 1：记录导出 API 函数的预期调用**

在 `frontend/src/api.js` 末尾预先加入函数签名（暂时抛错）：

```javascript
export async function exportConversationToFeishu(conversationId, authorizationCode) {
  throw new Error("尚未实现飞书导出");
}
```

在 `ChatView.vue` 添加一个暂时点击后显示“尚未实现飞书导出”的禁用/占位按钮。运行构建确认现有页面仍可编译；这是 UI 接线前的基线检查，不把占位版本交付给用户。

- [ ] **步骤 2：运行前端构建基线**

运行：

```bash
npm run build
```

工作目录：`frontend/`。

预期：构建成功；现有聊天 UI 未被破坏。

## 任务 5：实现前端 API 与用户交互

**文件：**
- 修改：`frontend/src/api.js`
- 修改：`frontend/src/components/ChatView.vue`
- 修改：`frontend/src/style.css`（仅当按钮布局需要全局样式时）

- [ ] **步骤 1：替换占位 API 为真实请求**

在 `frontend/src/api.js` 实现：

```javascript
export async function exportConversationToFeishu(conversationId, authorizationCode) {
  const res = await fetch(`${BASE}/conversations/${conversationId}/feishu-doc`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ authorization_code: authorizationCode }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "飞书文档创建失败，请稍后重试");
  }
  return res.json();
}
```

- [ ] **步骤 2：实现隔离的导出状态与授权码获取**

在 `ChatView.vue` 新增且仅用于导出的状态：

```javascript
const exporting = ref(false);
const exportError = ref("");
const exportedDocumentUrl = ref("");
```

实现 `requestFeishuAuthorizationCode()`：检查 `window.tt?.requestAuthCode`；缺失时抛出 `Error("请在飞书客户端内打开应用后导出")`；成功回调/Promise 结果中只提取 `code`。实现 `exportCurrentConversation()`：当 `!conversationId`、`!localMessages.length`、`isStreaming` 或 `exporting` 时直接返回；否则清空旧导出错误，获取授权码、调用 `api.exportConversationToFeishu`，成功后保存 URL，失败后只设置 `exportError`，最终清除 `exporting`。不得修改 `errorMessage`、`localMessages` 或聊天提交逻辑。

- [ ] **步骤 3：添加可访问的导出按钮与结果链接**

在 `ChatView.vue` 的标题区域增加：

```vue
<div class="chat-header">
  <h1 class="app-title">🚢 船舶操作知识问答</h1>
  <button
    class="btn"
    type="button"
    :disabled="!conversationId || !localMessages.length || isStreaming || exporting"
    @click="exportCurrentConversation"
  >
    {{ exporting ? "正在导出…" : "导出到飞书" }}
  </button>
</div>
<p v-if="exportError" class="error-banner">{{ exportError }}</p>
<p v-if="exportedDocumentUrl" class="export-success">
  已导出：<a :href="exportedDocumentUrl" target="_blank" rel="noopener">打开飞书文档</a>
</p>
```

为 `.chat-header`、`.export-success` 增加局部样式，不调整 `.messages`、`.composer`、`.sidebar` 的既有布局规则。

- [ ] **步骤 4：运行前端构建验证**

运行：

```bash
npm run build
```

工作目录：`frontend/`。

预期：构建成功，且无 Vite/Vue 编译错误。

## 任务 6：补齐配置说明并做全量验证

**文件：**
- 创建：`.env.example`
- 修改：`README.md`
- 测试：`tests/test_feishu_docs.py`
- 测试：`tests/test_feishu_export_api.py`

- [ ] **步骤 1：创建不含真实密钥的环境变量模板**

创建 `.env.example`：

```dotenv
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
FEISHU_APP_ID=cli_your_app_id
FEISHU_APP_SECRET=your_app_secret
```

- [ ] **步骤 2：在 README 增加最小联调清单**

写明：开发者后台为该应用开通 `docx:document` 权限并发布；将 App ID 和 App Secret 仅填入项目根目录 `.env`；在飞书桌面客户端打开网页应用；发送至少一轮问答；点击「导出到飞书」；打开返回链接确认文档归当前用户所有。不要在 README 写真实 token。

- [ ] **步骤 3：运行后端全量测试**

运行：

```bash
.venv/bin/python -m pytest tests -v
```

预期：所有测试通过，特别是既有 `/api/chat` NDJSON 回归测试通过。

- [ ] **步骤 4：运行前端生产构建**

运行：

```bash
npm run build
```

工作目录：`frontend/`。

预期：构建成功。

- [ ] **步骤 5：进行人工飞书联调**

1. 在开发者后台开通 `docx:document` 并发布新版本。
2. 确认 `.env` 中仅本机存在 `FEISHU_APP_ID` 与 `FEISHU_APP_SECRET`。
3. 让前端和后端保持运行，从飞书桌面客户端打开网页应用。
4. 选择一条已有对话或发送一轮新问答，点击「导出到飞书」。
5. 确认新 Docx 在当前用户空间中可打开，内容顺序、来源信息正确。
6. 再发送一条普通问答、切换会话、上传或清空知识库，确认这些原有功能仍可使用。
