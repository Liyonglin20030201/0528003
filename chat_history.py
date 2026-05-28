"""对话历史持久化模块

将聊天记录保存到本地 JSON 文件，支持多会话管理。
"""
import json
import time
from pathlib import Path

import config

HISTORY_DIR = config.BASE_DIR / "chat_histories"


def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def list_sessions() -> list[dict]:
    """列出所有保存的会话，按最近修改时间排序"""
    _ensure_dir()
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "id": f.stem,
                "title": data.get("title", "未命名对话"),
                "updated_at": data.get("updated_at", 0),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def load_session(session_id: str) -> list[dict]:
    """加载指定会话的消息列表"""
    _ensure_dir()
    file_path = HISTORY_DIR / f"{session_id}.json"
    if not file_path.exists():
        return []
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data.get("messages", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_session(session_id: str, messages: list[dict], title: str | None = None):
    """保存会话消息到文件"""
    _ensure_dir()
    if title is None:
        for msg in messages:
            if msg["role"] == "user":
                title = msg["content"][:30]
                break
        else:
            title = "未命名对话"

    data = {
        "title": title,
        "updated_at": time.time(),
        "messages": messages,
    }
    file_path = HISTORY_DIR / f"{session_id}.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rename_session(session_id: str, new_title: str):
    """重命名会话标题"""
    _ensure_dir()
    file_path = HISTORY_DIR / f"{session_id}.json"
    if not file_path.exists():
        return
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        data["title"] = new_title
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except (json.JSONDecodeError, OSError):
        pass


def export_session_as_text(session_id: str) -> str:
    """将会话导出为可读的文本格式"""
    _ensure_dir()
    file_path = HISTORY_DIR / f"{session_id}.json"
    if not file_path.exists():
        return ""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    title = data.get("title", "未命名对话")
    messages = data.get("messages", [])

    lines = [f"对话标题：{title}", "=" * 40, ""]
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"【{role}】")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def delete_session(session_id: str):
    """删除指定会话"""
    file_path = HISTORY_DIR / f"{session_id}.json"
    if file_path.exists():
        file_path.unlink()


def new_session_id() -> str:
    """生成新会话 ID"""
    return str(int(time.time() * 1000))
