"""不同模型 API 的 HTTP Client。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from http.client import RemoteDisconnected
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .provider import ProviderSettings


def _versioned_url(base_url: str, endpoint: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url + endpoint


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code >= 500 and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"API request failed with HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, RemoteDisconnected, TimeoutError) as exc:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"could not reach API: {url}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("API returned invalid JSON") from exc

    raise RuntimeError(f"API request failed: {url}")


class ModelClient:
    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self.model = settings.model

    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        raise NotImplementedError

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        """多轮消息数组调用：role/content 真正分离，system 单独携带。"""
        raise NotImplementedError

    def _require_api_key(self) -> str:
        if not self.settings.api_key:
            raise RuntimeError(f"{self.settings.name} API key is not configured")
        return self.settings.api_key


class OllamaModelClient(ModelClient):
    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        data = _post_json(
            self.settings.base_url.rstrip("/") + "/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                },
            },
            {},
            self.settings.timeout,
        )
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str(data.get("response", ""))

    def chat(self, messages, system=None, max_new_tokens=512) -> str:
        payload_messages = list(messages)
        if system:
            payload_messages = [{"role": "system", "content": system}, *payload_messages]
        data = _post_json(
            self.settings.base_url.rstrip("/") + "/api/chat",
            {
                "model": self.model,
                "messages": payload_messages,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                },
            },
            {},
            self.settings.timeout,
        )
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str(data.get("message", {}).get("content", ""))


class OpenAICompatibleModelClient(ModelClient):
    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_new_tokens,
        }

        data = _post_json(
            _versioned_url(self.settings.base_url, "/responses"),
            payload,
            {"Authorization": f"Bearer {self._require_api_key()}"},
            self.settings.timeout,
        )
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        if data.get("output_text"):
            return str(data["output_text"])
        for output in data.get("output", []):
            for content in output.get("content", []):
                if content.get("text"):
                    return str(content["text"])
        raise RuntimeError("OpenAI-compatible response contains no text")

    def chat(self, messages, system=None, max_new_tokens=512) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": max_new_tokens,
            "input": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
        }
        if system:
            payload["instructions"] = system

        data = _post_json(
            _versioned_url(self.settings.base_url, "/responses"),
            payload,
            {"Authorization": f"Bearer {self._require_api_key()}"},
            self.settings.timeout,
        )
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        if data.get("output_text"):
            return str(data["output_text"])
        for output in data.get("output", []):
            for content in output.get("content", []):
                if content.get("text"):
                    return str(content["text"])
        raise RuntimeError("OpenAI-compatible response contains no text")


class AnthropicCompatibleModelClient(ModelClient):
    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        data = _post_json(
            _versioned_url(self.settings.base_url, "/messages"),
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
            },
            {
                "x-api-key": self._require_api_key(),
                "anthropic-version": "2023-06-01",
            },
            self.settings.timeout,
        )
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        for content in data.get("content", []):
            if content.get("type") == "text":
                return str(content.get("text", ""))
        raise RuntimeError("Anthropic-compatible response contains no text")

    def chat(self, messages, system=None, max_new_tokens=512) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
        }
        if system:
            payload["system"] = system

        data = _post_json(
            _versioned_url(self.settings.base_url, "/messages"),
            payload,
            {
                "x-api-key": self._require_api_key(),
                "anthropic-version": "2023-06-01",
            },
            self.settings.timeout,
        )
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        for content in data.get("content", []):
            if content.get("type") == "text":
                return str(content.get("text", ""))
        raise RuntimeError("Anthropic-compatible response contains no text")


CLIENTS = {
    "responses": OpenAICompatibleModelClient,
    "messages": AnthropicCompatibleModelClient,
    "ollama": OllamaModelClient,
}
