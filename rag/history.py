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
    data = {
        "id": conv_id,
        "title": _derive_title(messages),
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_conversation(dir_path: Path, conv_id: str) -> None:
    path = _conversations_dir(dir_path) / f"{conv_id}.json"
    if path.exists():
        path.unlink()


def recent_turns(messages: list[dict], n_turns: int = 3) -> list[dict]:
    """取最近 n_turns 轮对话（role+content），用于喂给 LLM 做多轮记忆，不含 sources。"""
    plain = [{"role": m["role"], "content": m["content"]} for m in messages]
    return plain[-n_turns * 2 :]
