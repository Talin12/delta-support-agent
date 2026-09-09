"""LLM client for Gemini / Groq / OpenAI / Anthropic, with an on-disk cache.

Responses are cached by hash of (model, prompt, params), so reruns are free and
deterministic. LLM_OFFLINE=1 replays the cache and never hits the network, which
is how the results reproduce without an API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "cache" / "llm_cache.sqlite"

load_dotenv(REPO_ROOT / ".env")


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    """Raised when LLM_OFFLINE=1 and a prompt is not in the committed cache."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    api_key: str
    api_keys: tuple[str, ...] = ()

    def keys(self) -> tuple[str, ...]:
        return self.api_keys or (self.api_key,)


def _resolve_provider(model_override: str = "") -> Provider | None:
    """Pick a provider from whichever API key is present in the environment.

    model_override lets one call site run on a different model from the rest. The
    judge uses this: scoring generated replies with the same model that wrote them
    invites self-preference bias, where a model rates its own output generously.
    """
    candidates = [
        ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.6-flash"),
        ("groq", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
        ("openai", "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
        ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-5"),
    ]
    for name, key_var, model_var, default_model in candidates:
        # Google meters quota per PROJECT per model, so several keys from different
        # projects give genuinely independent budgets rather than a shared one.
        raw = os.environ.get(f"{key_var}S", "") or os.environ.get(key_var, "")
        keys = tuple(k.strip() for k in raw.split(",") if k.strip())
        if keys:
            model = model_override or os.environ.get(model_var, default_model).strip()
            return Provider(name, model, keys[0], keys)
    return None


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "  key TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT, created REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                (key, model, prompt, response, time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            models = [
                r[0] for r in self._conn.execute("SELECT DISTINCT model FROM responses")
            ]
        return {"entries": n, "models": models}


_cache = _Cache(CACHE_PATH)


class _RateLimiter:
    """Per-model token bucket.

    Google's free tier meters requests per minute *per project per model*, so each
    model carries its own independent budget. Running the three pipeline stages on
    three different models therefore multiplies available throughput, and the
    limiter has to track each one separately rather than throttling globally.
    """

    def __init__(self, per_minute: int = 14, window: float = 60.0):
        self.per_minute = per_minute
        self.window = window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def acquire(self, model: str) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                hits = [t for t in self._hits.get(model, []) if now - t < self.window]
                if len(hits) < self.per_minute:
                    hits.append(now)
                    self._hits[model] = hits
                    return
                sleep_for = self.window - (now - hits[0]) + 0.05
                self._hits[model] = hits
            time.sleep(max(sleep_for, 0.1))


_limiter = _RateLimiter(per_minute=int(os.environ.get("LLM_RPM", "14")))

# Round-robin cursor for spreading requests across API keys.
_rr_index = 0
_rr_lock = threading.Lock()


def cache_stats() -> dict[str, Any]:
    return _cache.stats()


def _cache_key(model: str, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    blob = json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class PermanentLLMError(LLMError):
    """A 4xx the server will keep rejecting: bad model name, bad key, malformed body."""


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _post_with_retry(url: str, *, headers: dict, payload: dict, attempts: int = 7) -> dict:
    # Free-tier models hit "high demand" 503 spikes that routinely outlast a short
    # backoff. 7 attempts from 3s doubles out to ~3 minutes of patience, which is
    # cheaper than a failed run.
    delay = 3.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code in RETRYABLE_STATUS:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code >= 400:
                # Retrying a bad model name or a bad key just wastes five backoffs
                # and buries the actual message.
                raise PermanentLLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        except PermanentLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry transport and rate errors
            last = exc
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
    raise LLMError(f"request failed after {attempts} attempts: {last}")


def _call_gemini(p: Provider, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{p.model}:generateContent"
    )
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    data = _post_with_retry(
        url, headers={"x-goog-api-key": p.api_key, "Content-Type": "application/json"}, payload=payload
    )
    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini response: {json.dumps(data)[:400]}") from exc

    parts = candidate.get("content", {}).get("parts")
    if not parts:
        # Gemini 3.x models always think, and thinking is billed against
        # maxOutputTokens BEFORE any text is emitted. Too small a budget yields a
        # candidate with no parts at all rather than a truncated string.
        finish = candidate.get("finishReason", "UNKNOWN")
        thoughts = data.get("usageMetadata", {}).get("thoughtsTokenCount", 0)
        raise LLMError(
            f"Gemini returned no output (finishReason={finish}, thinking consumed "
            f"{thoughts} tokens of a {max_tokens} budget). Raise max_tokens."
        )
    # Reasoning parts carry thought=True; only real output has usable text.
    return "".join(part.get("text", "") for part in parts if not part.get("thought"))


def _call_openai_compatible(
    p: Provider, base_url: str, system: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    data = _post_with_retry(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"},
        payload={
            "model": p.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response: {json.dumps(data)[:500]}") from exc


def _call_anthropic(p: Provider, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload: dict[str, Any] = {
        "model": p.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    data = _post_with_retry(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": p.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    try:
        return "".join(block.get("text", "") for block in data["content"])
    except (KeyError, TypeError) as exc:
        raise LLMError(f"unexpected Anthropic response: {json.dumps(data)[:500]}") from exc


def complete(
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model_tag: str = "",
    model_override: str = "",
) -> str:
    """Return a completion, served from cache when the same request was seen before.

    model_tag lets a caller force a distinct cache entry for the same prompt text
    (used to keep judge calls separate from generation calls).
    """
    offline = os.environ.get("LLM_OFFLINE", "0") == "1"
    provider = _resolve_provider(model_override)
    model_id = f"{provider.name}:{provider.model}" if provider else "offline"
    if model_tag:
        model_id = f"{model_id}#{model_tag}"

    key = _cache_key(model_id, system, prompt, temperature, max_tokens)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    if offline:
        raise CacheMiss(
            "LLM_OFFLINE=1 but this prompt is not in the committed cache. "
            "Set a provider API key in .env to generate it."
        )
    if provider is None:
        raise LLMError(
            "No LLM API key found. Copy .env.example to .env and set GEMINI_API_KEY "
            "(free, no credit card: https://aistudio.google.com/apikey)."
        )

    # Round-robin the STARTING key rather than always beginning at key one.
    # Rotating only on failure sends every request to the first key and queues the
    # whole run behind that key's rate limit while the others sit idle; starting at
    # a different offset each call spreads load across all projects, and the
    # remaining keys still act as failover when one is exhausted.
    all_keys = provider.keys()
    with _rr_lock:
        global _rr_index
        start = _rr_index % len(all_keys)
        _rr_index += 1
    ordered = all_keys[start:] + all_keys[:start]

    last_error: Exception | None = None
    for api_key in ordered:
        attempt = Provider(provider.name, provider.model, api_key)
        # Quota is per project per model, so throttle each pairing separately.
        _limiter.acquire(f"{provider.model}:{api_key[-6:]}")
        try:
            if provider.name == "gemini":
                text = _call_gemini(attempt, system, prompt, temperature, max_tokens)
            elif provider.name == "groq":
                text = _call_openai_compatible(
                    attempt, "https://api.groq.com/openai/v1", system, prompt, temperature, max_tokens
                )
            elif provider.name == "openai":
                text = _call_openai_compatible(
                    attempt, "https://api.openai.com/v1", system, prompt, temperature, max_tokens
                )
            elif provider.name == "anthropic":
                text = _call_anthropic(attempt, system, prompt, temperature, max_tokens)
            else:
                raise LLMError(f"unknown provider {provider.name}")
        except LLMError as exc:
            last_error = exc
            continue
        _cache.put(key, model_id, prompt, text)
        return text

    raise LLMError(f"all {len(provider.keys())} key(s) failed: {last_error}")


def complete_json(
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model_tag: str = "",
    model_override: str = "",
) -> dict[str, Any]:
    """Completion parsed as JSON, tolerating markdown fences and prose padding."""
    raw = complete(
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        model_tag=model_tag,
        model_override=model_override,
    )
    return parse_json_block(raw)


def parse_json_block(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def describe() -> str:
    provider = _resolve_provider()
    offline = os.environ.get("LLM_OFFLINE", "0") == "1"
    stats = cache_stats()
    if offline:
        return f"LLM: offline replay ({stats['entries']} cached responses)"
    if provider is None:
        return f"LLM: no key configured ({stats['entries']} cached responses available)"
    return f"LLM: {provider.name}/{provider.model} ({stats['entries']} cached responses)"


if __name__ == "__main__":
    print(describe())
