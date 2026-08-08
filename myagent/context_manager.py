from __future__ import annotations

from .agent_config import AgentParams, Message
from tokenizer import tokenizer
from dataclasses import dataclass, field, asdict
import json
from datetime import datetime
from pathlib import Path

class ContextManager:
    """上下文管理器，用于在多轮对话中管理消息历史记录。

    配置了 storage_dir 时，会在会话开始时按时间戳生成独立的历史文件，
    把消息以 JSON 形式永久化保存到本地；下次会话自动加载最近一次的历史，
    实现跨会话的上下文持久化。
    """

    HISTORY_PREFIX = "history"

    def __init__(self, agent_params: AgentParams):
        self.agent_params = agent_params
        self.messages: list[Message] = agent_params.messages or []
        self.tokenizer = tokenizer
        self.storage_dir = (
            Path(agent_params.storage_dir).resolve()
            if agent_params.storage_dir
            else None
        )
        # 会话开始时按当前时间生成文件名，避免所有会话挤在同一个文件里。
        self.history_file = (
            self.storage_dir
            / f"{self.HISTORY_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}.json"
            if self.storage_dir
            else None
        )

        if self.storage_dir:
            # 确保存储文件夹存在，并加载最近一次会话的历史。
            self._ensure_storage_dir()
            self._load()

        # 本会话产生的消息（独立跟踪，避免裁剪历史导致索引错位）。
        self._session_messages: list[Message] = []

    def _ensure_storage_dir(self) -> None:
        """判断存储文件夹是否存在，不存在则创建。"""
        if not self.storage_dir.is_dir():
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _latest_history_file(self) -> Path | None:
        """返回存储目录下最近修改的历史文件；不存在则返回 None。"""
        if self.storage_dir is None:
            return None
        files = list(self.storage_dir.glob(f"{self.HISTORY_PREFIX}_*.json"))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def _load(self) -> None:
        """加载最近一次会话的历史；文件缺失或损坏时静默跳过。"""
        history_file = self._latest_history_file()
        if history_file is None:
            return
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
            self.messages = [
                Message(**item)
                for item in data
                if isinstance(item, dict)
            ]
        except (json.JSONDecodeError, TypeError, ValueError):
            # 文件损坏时从空历史开始，避免阻塞对话。
            self.messages = []
        # 上次会话的历史可能超出 token 上限，加载后从最旧的开始裁剪。
        self.messages = self._clip(self.messages)

    def save(self) -> None:
        """把本次会话新增且仍保留在上下文中的消息写入本会话历史文件。

        只保存本会话产生的消息，不会把加载进来的历史写入本文件；
        被 _clip 裁掉的本会话旧消息也不再保存。
        """
        if self.storage_dir is None:
            return
        self._ensure_storage_dir()
        # 只保留仍存在于上下文中的本会话消息。
        session_messages = [
            m for m in self._session_messages
            if any(m is x for x in self.messages)
        ]
        payload = [asdict(m) for m in session_messages]
        self.history_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_user_message(self, content: str) -> None:
        """添加用户消息，裁剪以满足 token 限制，然后增量保存。"""
        message = Message(role="user", content=content)
        self.messages.append(message)
        self._session_messages.append(message)
        # 先裁剪，再保存仍留在上下文中的本会话消息。
        self.messages = self._clip(self.messages)
        self.save()

    def __enter__(self):
        """进入上下文时，返回当前消息列表。"""
        return self.messages

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文时，将消息列表更新回 agent_params 并永久化保存。"""
        self.agent_params.messages = self.messages
        self.save()

    def _clip(self, messages: list[Message]) -> list[Message]:
        """裁剪消息列表，确保总 token 数不超过限制。"""

        total_tokens = self._count_message_tokens(messages)

        if total_tokens <= self.agent_params.token_limit:
            return messages

        clipped_messages = messages.copy()

        # 从最旧的 message 开始删除
        while clipped_messages:
            if total_tokens <= self.agent_params.token_limit:
                break

            removed_message = clipped_messages.pop(0)

            total_tokens -= self._count_message_tokens(
                [removed_message]
            )

        return clipped_messages

    def _count_message_tokens(self, messages: list[Message]) -> int:
        """计算消息列表的总 token 数。"""
        json_text = json.dumps(
            [asdict(m) for m in messages],
            ensure_ascii=False
        )

        return len(
            self.tokenizer.encode(json_text)
        )