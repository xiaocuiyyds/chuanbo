"""会话记忆的回归测试。

排障对话天然是多轮的，而喂给模型的窗口只有最近三轮。这里保证两件事：
滑出窗口的前提能通过摘要保留下来；摘要不会被每轮的保存操作冲掉。
"""

import json

from rag.history import (
    load_conversation,
    load_summary,
    recent_turns,
    save_conversation,
    save_summary,
    session_history,
)


def make_messages(n_turns: int, with_sources: bool = False) -> list[dict]:
    messages = []
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"问题{i}"})
        assistant = {"role": "assistant", "content": f"回答{i}"}
        if with_sources:
            assistant["sources"] = [{"source": "手册.pdf", "page": 10 + i}]
        messages.append(assistant)
    return messages


def test_recent_turns_keeps_only_the_window():
    turns = recent_turns(make_messages(10), n_turns=3)
    assert len(turns) == 6
    assert turns[0]["content"] == "问题7"


def test_recent_turns_annotates_cited_pages():
    """用户会用"刚才那页"追问，助手消息必须带上当时引用的页码。"""
    turns = recent_turns(make_messages(2, with_sources=True), n_turns=3)
    assistant = [t for t in turns if t["role"] == "assistant"][-1]
    assert "《手册.pdf》第11页" in assistant["content"]
    assert "回答1" in assistant["content"]


def test_recent_turns_deduplicates_cited_pages():
    messages = [
        {"role": "user", "content": "问"},
        {
            "role": "assistant",
            "content": "答",
            "sources": [
                {"source": "手册.pdf", "page": 5},
                {"source": "手册.pdf", "page": 5},
                {"source": "手册.pdf", "page": 6},
            ],
        },
    ]
    content = recent_turns(messages)[-1]["content"]
    assert content.count("第5页") == 1
    assert "第6页" in content


def test_recent_turns_without_sources_is_unchanged():
    turns = recent_turns(make_messages(1), n_turns=3)
    assert turns[-1]["content"] == "回答0"


def test_session_history_without_summary_is_just_the_window():
    messages = make_messages(5)
    assert session_history(messages, "", n_turns=3) == recent_turns(messages, n_turns=3)


def test_session_history_prepends_summary_and_keeps_roles_alternating():
    history = session_history(make_messages(5), "主机是 MAN B&W ME-B。", n_turns=3)
    assert "主机是 MAN B&W ME-B。" in history[0]["content"]
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    roles = [m["role"] for m in history]
    assert all(a != b for a, b in zip(roles, roles[1:])), "不能出现连续同角色的消息"


def test_session_history_still_contains_the_recent_window():
    history = session_history(make_messages(5), "要点。", n_turns=3)
    assert history[-1]["content"] == "回答4"
    assert sum(1 for m in history if m["role"] == "user") == 4  # 摘要那条 + 3 轮提问


def test_summary_round_trip(tmp_path):
    save_conversation(tmp_path, "c1", make_messages(2))
    save_summary(tmp_path, "c1", "已确认滑油设定值 0.4MPa。", 4)
    assert load_summary(tmp_path, "c1") == ("已确认滑油设定值 0.4MPa。", 4)


def test_saving_conversation_preserves_the_summary(tmp_path):
    """每轮问答都会重写对话文件，摘要不能被这一步清掉。"""
    save_conversation(tmp_path, "c1", make_messages(2))
    save_summary(tmp_path, "c1", "要点。", 4)

    save_conversation(tmp_path, "c1", make_messages(3))
    assert load_summary(tmp_path, "c1") == ("要点。", 4)
    assert len(load_conversation(tmp_path, "c1")) == 6


def test_summary_of_unknown_conversation_is_empty(tmp_path):
    assert load_summary(tmp_path, "没有这个对话") == ("", 0)


def test_save_summary_on_missing_conversation_is_a_noop(tmp_path):
    save_summary(tmp_path, "还没落盘", "要点。", 2)
    assert load_summary(tmp_path, "还没落盘") == ("", 0)


def test_conversation_file_shape(tmp_path):
    save_conversation(tmp_path, "c1", make_messages(1))
    data = json.loads((tmp_path / "conversations" / "c1.json").read_text(encoding="utf-8"))
    assert set(data) >= {"id", "title", "updated_at", "summary", "summary_upto", "messages"}
