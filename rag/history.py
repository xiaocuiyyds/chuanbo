import json
import uuid
from datetime import datetime
from pathlib import Path

CONVERSATIONS_DIRNAME = "conversations"
LEGACY_HISTORY_FILENAME = "chat_history.json"
TITLE_MAX_LEN = 20


def _conversations_dir(dir_path: Path) -> Path:
    conv_dir = dir_path / CONVERSATIONS_DIRNAME
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def _derive_title(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip().replace("\n", " ")
            return text[:TITLE_MAX_LEN] + ("..." if len(text) > TITLE_MAX_LEN else "")
    return "新对话"


def _migrate_legacy_history(dir_path: Path) -> None:
    """把旧版单文件 chat_history.json 迁移成按对话分文件存储，只在第一次调用时生效。"""
    legacy_path = dir_path / LEGACY_HISTORY_FILENAME
    if not legacy_path.exists():
        return
    with open(legacy_path, encoding="utf-8") as f:
        messages = json.load(f)
    if messages:
        save_conversation(dir_path, new_conversation_id(), messages)
    legacy_path.rename(legacy_path.with_suffix(".json.migrated"))


def list_conversations(dir_path: Path) -> list[dict]:
    """返回所有对话的元信息（id/title/updated_at），按更新时间倒序，不含 messages。"""
    _migrate_legacy_history(dir_path)
    conversations = []
    for path in _conversations_dir(dir_path).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        conversations.append(
            {"id": data["id"], "title": data["title"], "updated_at": data["updated_at"]}
        )
    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return conversations


def load_conversation(dir_path: Path, conv_id: str) -> list[dict]:
    path = _conversations_dir(dir_path) / f"{conv_id}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["messages"]


def save_conversation(dir_path: Path, conv_id: str, messages: list[dict]) -> None:
    path = _conversations_dir(dir_path) / f"{conv_id}.json"
    # 摘要是另一条路径写进来的，这里整体重写文件时要保住它，否则每轮问答都会把它清掉
    summary, summary_upto = load_summary(dir_path, conv_id)
    data = {
        "id": conv_id,
        "title": _derive_title(messages),
        "updated_at": datetime.now().isoformat(),
        "summary": summary,
        "summary_upto": summary_upto,
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_conversation(dir_path: Path, conv_id: str) -> None:
    path = _conversations_dir(dir_path) / f"{conv_id}.json"
    if path.exists():
        path.unlink()


def recent_turns(messages: list[dict], n_turns: int = 3) -> list[dict]:
    """取最近 n_turns 轮对话，用于喂给 LLM 做多轮记忆。

    助手消息会附上它当时引用的文档和页码。用户经常用"刚才那页第 3 步"这类说法追问，
    而 sources 原本在这一步被整个丢掉，模型无从知道"那页"指的是哪一页。
    附的是页码不是原文，占用很小。
    """
    turns = []
    for message in messages[-n_turns * 2 :]:
        content = message["content"]
        sources = message.get("sources")
        if message["role"] == "assistant" and sources:
            cited = dict.fromkeys(f"《{s['source']}》第{s['page']}页" for s in sources)
            content = f"{content}\n\n（本次回答依据：{'、'.join(cited)}）"
        turns.append({"role": message["role"], "content": content})
    return turns


def session_history(messages: list[dict], summary: str, n_turns: int = 3) -> list[dict]:
    """拼出喂给 LLM 的对话历史：会话摘要 + 最近 n_turns 轮。

    摘要以一问一答的形式垫在最前面，而不是塞进 system prompt——查询改写和回答用的是
    两个不同的 system prompt，走 history 这条路两边都能拿到，不必给它们各改一次签名。
    补一句助手应答是为了保持 user/assistant 交替，避免出现连续两条 user 消息。
    """
    turns = recent_turns(messages, n_turns=n_turns)
    if not summary:
        return turns
    return [
        {"role": "user", "content": f"（以下是本次对话前面已经确立的要点，供你理解后续追问）\n{summary}"},
        {"role": "assistant", "content": "已记录这些要点。"},
        *turns,
    ]


def load_summary(dir_path: Path, conv_id: str) -> tuple[str, int]:
    """读会话摘要，返回 (摘要, 已被摘要覆盖到第几条消息)。"""
    path = _conversations_dir(dir_path) / f"{conv_id}.json"
    if not path.exists():
        return "", 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("summary", ""), data.get("summary_upto", 0)


def save_summary(dir_path: Path, conv_id: str, summary: str, upto: int) -> None:
    """把摘要写回对话文件。对话还没落盘时直接跳过，下一轮保存后自会重新生成。"""
    path = _conversations_dir(dir_path) / f"{conv_id}.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["summary"] = summary
    data["summary_upto"] = upto
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
