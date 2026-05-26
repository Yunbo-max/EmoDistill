"""
OpenAI-compatible LLM wrapper supporting both DashScope (Aliyun) and OpenAI.

Pick the provider at runtime by setting `LLM_PROVIDER`:

    LLM_PROVIDER=dashscope   (default)  →  Qwen-Plus etc. via DashScope endpoint
    LLM_PROVIDER=openai                 →  gpt-4o-mini etc. via OpenAI endpoint

Both backends expose the same OpenAI ChatCompletions API, so this wrapper just
swaps base URL + key source. Multiple keys are rotated round-robin to spread
load and stay under per-key rate limits.

Drop-in compatible with the LLMWrapper interface used elsewhere:
    wrapper.invoke([HumanMessage(content=...)], temperature=...)
returns an object with a `.content` attribute.

Environment:
    LLM_PROVIDER             — "dashscope" (default) or "openai"
    # DashScope
    DASHSCOPE_API_KEYS       — comma-separated list (preferred)
    DASHSCOPE_API_KEY        — single key fallback
    DASHSCOPE_DEFAULT_MODEL  — default model when not specified (default: qwen-plus)
    # OpenAI
    OPENAI_API_KEYS          — comma-separated list (preferred)
    OPENAI_API_KEY           — single key fallback
    OPENAI_DEFAULT_MODEL     — default model when not specified (default: gpt-4o-mini)
    OPENAI_BASE_URL          — override (defaults to https://api.openai.com/v1)
"""

import os
import threading
from typing import List, Optional

from openai import OpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_BASE_URL_DEFAULT = "https://api.openai.com/v1"


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "dashscope").strip().lower() or "dashscope"


def _provider_base_url(provider: str) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL_DEFAULT).strip()
    return DASHSCOPE_BASE_URL


def _provider_default_model(provider: str) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    return os.environ.get("DASHSCOPE_DEFAULT_MODEL", "qwen-plus")


def _load_keys(provider: str) -> List[str]:
    """Load API keys from env (multi or single) for the active provider."""
    if provider == "openai":
        multi = os.environ.get("OPENAI_API_KEYS", "").strip()
        single = os.environ.get("OPENAI_API_KEY", "").strip()
    else:
        multi = os.environ.get("DASHSCOPE_API_KEYS", "").strip()
        single = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    if single:
        return [single]
    return []


class _KeyRotator:
    """Thread-safe round-robin over a fixed key pool."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("No API keys available")
        self.keys = list(keys)
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            k = self.keys[self._idx]
            self._idx = (self._idx + 1) % len(self.keys)
            return k

    def __len__(self) -> int:
        return len(self.keys)


# Provider-specific process-wide rotators
_ROTATORS: dict = {}


def get_rotator(provider: Optional[str] = None) -> _KeyRotator:
    p = provider or _provider()
    if p not in _ROTATORS:
        keys = _load_keys(p)
        if not keys:
            env_hint = "OPENAI_API_KEY(S)" if p == "openai" else "DASHSCOPE_API_KEY(S)"
            raise RuntimeError(
                f"No {p} keys found. Set {env_hint} (comma-separated) in your environment or .env."
            )
        _ROTATORS[p] = _KeyRotator(keys)
    return _ROTATORS[p]


class _MockMessage:
    """Mimic the LangChain AIMessage interface."""

    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


class DashScopeWrapper:
    """OpenAI-compatible LLM client. Auto-selects DashScope or OpenAI based on LLM_PROVIDER."""

    @property
    def DEFAULT_MODEL(self) -> str:
        return _provider_default_model(self.provider)

    def __init__(
        self,
        model: Optional[str] = None,
        role: str = "generic",
        max_tokens: int = 512,
        timeout: float = 60.0,
        provider: Optional[str] = None,
    ):
        self.provider = (provider or _provider()).strip().lower()
        self.model = model or _provider_default_model(self.provider)
        self.role = role
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = _provider_base_url(self.provider)
        self.rotator = get_rotator(self.provider)
        self.call_count = 0
        self.fail_count = 0

    # -------- Public interface --------

    def invoke(self, messages, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs):
        api_messages = self._to_openai_messages(messages)
        max_t = max_tokens if max_tokens is not None else self.max_tokens

        # Qwen3 thinking-series baselines (qwen3-*, qwq-*) require
        # `enable_thinking: false` for non-streaming calls. Only attach for DashScope.
        extra_body = {"enable_thinking": False} if self.provider == "dashscope" else None

        last_err: Optional[Exception] = None
        for attempt in range(len(self.rotator)):
            key = self.rotator.next()
            client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)
            try:
                self.call_count += 1
                kwargs_create = dict(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_t,
                    stream=False,
                )
                if extra_body is not None:
                    kwargs_create["extra_body"] = extra_body
                resp = client.chat.completions.create(**kwargs_create)
                content = resp.choices[0].message.content or ""
                return _MockMessage(content)
            except Exception as e:
                self.fail_count += 1
                last_err = e
                msg = str(e).lower()
                if any(m in msg for m in ("rate", "limit", "401", "403", "quota")):
                    continue
                break
        raise RuntimeError(
            f"{self.provider} call failed after {len(self.rotator)} key attempts: {last_err}"
        )

    def cleanup(self) -> None:
        pass

    # -------- Helpers --------

    @staticmethod
    def _to_openai_messages(messages) -> list:
        """Convert LangChain message list (or strings) to OpenAI chat format."""
        out = []
        if isinstance(messages, str):
            out.append({"role": "user", "content": messages})
            return out
        for m in messages:
            if hasattr(m, "content"):
                content = m.content
                mtype = getattr(m, "type", "human")
                if mtype == "ai":
                    role = "assistant"
                elif mtype == "system":
                    role = "system"
                else:
                    role = "user"
            else:
                content = str(m)
                role = "user"
            out.append({"role": role, "content": content})
        return out


# Public aliases for clarity in user code
LLMClient = DashScopeWrapper
