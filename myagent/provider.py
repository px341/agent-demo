"""从 .env.llm（或 .env.example 模板）加载 LLM provider，并提供统一调用函数。

调用方（如 cli.py）只需调用 complete() / chat() 即可完成一次 LLM 请求。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# 优先读取本地的 .env.llm（真实配置/密钥），缺失时回退到 .env.example 模板。
ENV_FILES = (
    Path(__file__).resolve().parent.parent / ".env.llm",
    Path(__file__).resolve().parent.parent / ".env.example",
)
DEFAULT_PROVIDER = "deepseek"
PREFIX_ALIASES = {"claude": "ANTHROPIC", "grok": "XAI"}


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    key: str
    name: str
    protocol: str
    api_key_env: str
    api_key: str | None
    base_url: str
    model: str
    timeout: int


@dataclass(frozen=True, slots=True)
class ChatResult:
    """一次对话调用的结果，供调用方读取回答与所用配置。"""

    answer: str
    settings: ProviderSettings


def _env_file() -> Path:
    """按优先级找到可用的配置文件。"""
    for path in ENV_FILES:
        if path.is_file():
            return path
    raise RuntimeError("缺少配置文件，请先复制 .env.example 为 .env.llm 并填写。")


def _read_env(path: Path | None = None) -> dict[str, str]:
    """解析 KEY=VALUE 形式的 env 文件，支持 export 前缀与引号。"""
    env_file = path or _env_file()
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_provider(provider: str | None = None) -> ProviderSettings:
    """加载配置；进程环境变量优先，其次 .env 文件，未指定时默认 deepseek。"""

    values = _read_env()

    def get(name: str, default: str = "") -> str:
        return os.getenv(name, values.get(name, default))

    key = (provider or get("PROVIDER", DEFAULT_PROVIDER)).lower()
    prefix = PREFIX_ALIASES.get(key, key.upper())
    required = {
        "protocol": f"{prefix}_API_TYPE",
        "base_url": f"{prefix}_API_BASE",
        "model": f"{prefix}_MODEL",
    }
    missing = [env_name for env_name in required.values() if not get(env_name)]
    if missing:
        raise RuntimeError(f"配置文件缺少 {key} 相关配置：{', '.join(missing)}")

    timeout = get("LLM_TIMEOUT", "60")
    try:
        timeout_value = int(timeout)
        if timeout_value <= 0:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError("LLM_TIMEOUT 必须是大于 0 的整数") from exc

    api_key_env = f"{prefix}_API_KEY"
    return ProviderSettings(
        key=key,
        name=key.title(),
        protocol=get(required["protocol"]),
        api_key_env=api_key_env,
        api_key=get(api_key_env) or None,
        base_url=get(required["base_url"]),
        model=get(required["model"]),
        timeout=timeout_value,
    )


def create_client(provider: str | None = None):
    """创建配置中指定协议的统一客户端。"""

    from .clients import CLIENTS

    settings = load_provider(provider)
    try:
        client_type = CLIENTS[settings.protocol]
    except KeyError as exc:
        choices = ", ".join(CLIENTS)
        raise RuntimeError(
            f"不支持 API_TYPE {settings.protocol!r}，可选：{choices}"
        ) from exc
    return client_type(settings)


def complete(
    prompt: str,
    provider: str | None = None,
    max_new_tokens: int = 512,
) -> ChatResult:
    """调用 LLM 完成一次对话，供 CLI 等外部调用方直接使用。

    职责链：加载配置 → 创建客户端 → 校验 API Key → 发起请求。
    失败统一抛出 RuntimeError，由调用方决定如何展示。
    """

    client = create_client(provider)
    if not client.settings.api_key:
        raise RuntimeError(
            f"未配置 {client.settings.api_key_env}，请在 .env.llm 中设置。"
        )
    answer = client.complete(prompt, max_new_tokens=max_new_tokens)
    return ChatResult(answer=answer, settings=client.settings)


def chat(prompt: str, provider: str | None = None) -> str:
    """轻量便捷函数：只返回回答文本（不关心展示信息时使用）。"""

    return complete(prompt, provider=provider).answer


def list_providers() -> list[str]:
    """返回配置文件中声明过的 provider 键（保持出现顺序）。"""

    values = _read_env()
    keys: list[str] = []
    for key in values.get("PROVIDER", DEFAULT_PROVIDER).split(","):
        key = key.strip().lower()
        if key and key not in keys:
            keys.append(key)
    return keys
