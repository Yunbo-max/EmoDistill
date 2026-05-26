"""
DashScope (Aliyun) LLM wrapper with round-robin key rotation.

DashScope exposes an OpenAI-compatible endpoint, so we can reuse the openai
SDK. Multiple API keys are rotated round-robin across requests to spread load
and stay under per-key QPS limits.

Drop-in compatible with the LLMWrapper interface used elsewhere in the codebase:
    wrapper.invoke([HumanMessage(content=...)], temperature=...)
returns an object with a `.content` attribute.

Usage:
    from EmoDistill.dashscope_wrapper import DashScopeWrapper
    llm = DashScopeWrapper(model="qwen-plus", role="creditor")
    resp = llm.invoke([HumanMessage(content="Hi")])
    print(resp.content)

Environment:
    DASHSCOPE_API_KEYS       — comma-separated list of keys (preferred)
    DASHSCOPE_API_KEY        — single key fallback
    DASHSCOPE_DEFAULT_MODEL  — default model name when not specified
"""

import os
import threading
from typing import List, Optional

from openai import OpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _load_keys() -> List[str]:
    """Load API keys from env (multi or single)."""
    multi = os.environ.get("DASHSCOPE_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if single:
        return [single]
    return []


class _KeyRotator:
    """Thread-safe round-robin over a fixed key pool."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("No DashScope API keys available")
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


# Process-wide rotator so all wrappers share the same round-robin state.
_GLOBAL_ROTATOR: Optional[_KeyRotator] = None


def get_rotator() -> _KeyRotator:
    global _GLOBAL_ROTATOR
    if _GLOBAL_ROTATOR is None:
        keys = _load_keys()
        if not keys:
            raise RuntimeError(
                "No DashScope keys found. Set DASHSCOPE_API_KEYS (comma-separated) or DASHSCOPE_API_KEY."
            )
        _GLOBAL_ROTATOR = _KeyRotator(keys)
    return _GLOBAL_ROTATOR


class _MockMessage:
    """Mimic the LangChain AIMessage interface."""

    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


class DashScopeWrapper:
    """OpenAI-compatible DashScope client with key rotation."""

    DEFAULT_MODEL = os.environ.get("DASHSCOPE_DEFAULT_MODEL", "qwen-plus")

    def __init__(
        self,
        model: Optional[str] = None,
        role: str = "generic",
        max_tokens: int = 512,
        timeout: float = 60.0,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.role = role
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.rotator = get_rotator()
        self.call_count = 0
        self.fail_count = 0

    # -------- Public interface --------

    def invoke(self, messages, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs):
        api_messages = self._to_openai_messages(messages)
        max_t = max_tokens if max_tokens is not None else self.max_tokens

        # Qwen3 thinking-series baselines (qwen3-*, qwq-*) require
        # `enable_thinking: false` for non-streaming calls. Always send it; for
        # non-thinking baselines DashScope ignores the extra field.
        extra_body = {"enable_thinking": False}

        # Try each key once on transient failure (so we can recover from a single
        # bad key without aborting the call).
        last_err: Optional[Exception] = None
        for attempt in range(len(self.rotator)):
            key = self.rotator.next()
            client = OpenAI(api_key=key, base_url=DASHSCOPE_BASE_URL, timeout=self.timeout)
            try:
                self.call_count += 1
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_t,
                    stream=False,
                    extra_body=extra_body,
                )
                content = resp.choices[0].message.content or ""
                return _MockMessage(content)
            except Exception as e:
                self.fail_count += 1
                last_err = e
                # On rate limit or auth error, rotate to next key and retry
                msg = str(e).lower()
                if any(m in msg for m in ("rate", "limit", "401", "403", "quota")):
                    continue
                # Other errors: re-raise immediately
                break
        # All retries exhausted
        raise RuntimeError(f"DashScope call failed after {len(self.rotator)} key attempts: {last_err}")

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

        if not any(msg["role"] == "system" for msg in out):
            out.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in negotiation and communication.",
                },
            )
        return out

    def stats(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "keys": len(self.rotator),
            "calls": self.call_count,
            "failures": self.fail_count,
        }
