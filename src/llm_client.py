"""
LLM Client
===========
Thin wrapper around the Groq SDK with SSL fix (TU Dresden proxy),
automatic retries on rate-limit errors, and JSON response parsing.

Supports two models:
  - Agent model (8b): llama-3.1-8b-instant
  - Judge/Moderator model: meta-llama/llama-4-scout-17b-16e-instruct
    (replaced llama-3.3-70b-versatile, deprecated 2026-08-16;
     pilot comparison: 100% valid, 83-89% intra-agreement, 1.26s latency)

If litellm is needed later for multi-provider support, this module
can be swapped without changing the rest of the codebase.

See also:
  - pilots/feasibility/ — feasibility tests used this same approach
  - mds/compute_cost_estimate.md — benchmark results
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from groq import Groq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff
MAX_RATE_LIMIT_WAIT = 600  # 10 minutes max wait for daily limits


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Parsed LLM API response."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_s: float
    parsed_json: Optional[dict] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Groq API client with SSL bypass and retry logic.

    Usage:
        client = LLMClient()  # reads GROQ_API_KEY from env
        resp = client.complete(
            model="llama-3.1-8b-instant",
            system_prompt="You are...",
            user_prompt="Argue that...",
            max_tokens=200,
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        verify_ssl: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        # SSL bypass for TU Dresden / corporate proxy
        http_client = httpx.Client(verify=verify_ssl) if not verify_ssl else None
        self._client = Groq(
            api_key=self.api_key,
            http_client=http_client,
        )

    def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        parse_json: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request with retry logic.

        Args:
            model: Model identifier (e.g. AGENT_MODEL, JUDGE_MODEL).
            system_prompt: System message content.
            user_prompt: User message content.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            parse_json: If True, attempt to parse response as JSON.

        Returns:
            LLMResponse with text, token counts, and optionally parsed JSON.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.time()
                api_kwargs = {}
                if parse_json:
                    api_kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **api_kwargs,
                )
                latency = time.time() - t0

                text = resp.choices[0].message.content.strip()
                result = LLMResponse(
                    text=text,
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                    model=model,
                    latency_s=round(latency, 3),
                )

                if parse_json:
                    result.parsed_json = self._parse_json(text)

                return result

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate_limit" in error_str or "429" in error_str
                is_server_error = "500" in error_str or "503" in error_str

                if (is_rate_limit or is_server_error) and attempt < self.max_retries:
                    # Try to parse Groq's suggested wait time
                    parsed_wait = self._parse_retry_after(str(e))
                    if parsed_wait and parsed_wait <= MAX_RATE_LIMIT_WAIT:
                        delay = parsed_wait + 1  # +1s buffer
                    else:
                        delay = self.retry_base_delay * (2 ** (attempt - 1))

                    logger.warning(
                        f"LLM call failed (attempt {attempt}/{self.max_retries}): "
                        f"{type(e).__name__}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {attempt} attempts: {e}")
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("LLM call exhausted all retries")

    @staticmethod
    def _parse_retry_after(error_msg: str) -> Optional[float]:
        """Parse wait time from Groq error message.

        Handles formats like:
            'Please try again in 22m27.84s'
            'Please try again in 4.5s'
            'Please try again in 1m0s'
        """
        match = re.search(
            r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s",
            error_msg,
        )
        if match:
            minutes = int(match.group(1) or 0)
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        return None

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Best-effort JSON parsing from LLM output.

        Handles: raw JSON, markdown-fenced JSON, JSON embedded in text,
        thinking-model <think>...</think> prefix blocks (Qwen3, etc.),
        and JSON truncated by max_tokens cutoff.
        """
        cleaned = text.strip()

        # Strip <think>...</think> blocks produced by reasoning models
        # (Qwen3, DeepSeek-R1, etc.). Must happen BEFORE fence/brace search
        # because thinking content may itself contain braces.
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        # Strip markdown code fences (handles both complete and truncated)
        fence_match = re.search(r"```\w*\s*\n(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        elif cleaned.startswith("```"):
            # Opening fence without closing (truncated response)
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1:].strip()

        # 1) Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2) Try to extract JSON object from surrounding text
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

        # 3) Try to repair truncated JSON (max_tokens cutoff)
        if start != -1:
            repaired = LLMClient._repair_truncated_json(cleaned[start:])
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
        return None

    @staticmethod
    def _repair_truncated_json(fragment: str) -> Optional[str]:
        """Attempt to repair JSON truncated by max_tokens cutoff.

        Strategy: close any unclosed string, then try progressively
        trimming at comma break-points (from least to most aggressive)
        until balancing braces/brackets yields parseable JSON.
        """
        # Close any unclosed string literal
        in_str = False
        esc = False
        for c in fragment:
            if esc:
                esc = False
            elif c == "\\" and in_str:
                esc = True
            elif c == '"':
                in_str = not in_str
        if in_str:
            fragment += '"'

        # Find all comma positions outside strings (potential trim points)
        break_points: list[int] = []
        in_str = False
        esc = False
        for i, c in enumerate(fragment):
            if esc:
                esc = False
            elif c == "\\" and in_str:
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str and c == ",":
                break_points.append(i)

        # Try candidates: full fragment first, then trim at each comma R-to-L
        candidates = [fragment] + [fragment[:bp] for bp in reversed(break_points)]

        for candidate in candidates:
            trimmed = candidate.rstrip().rstrip(",").rstrip(":")
            if not trimmed:
                continue

            # Count unbalanced braces/brackets
            in_str = False
            esc = False
            brace = 0
            bracket = 0
            for c in trimmed:
                if esc:
                    esc = False
                elif c == "\\" and in_str:
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == "{": brace += 1
                    elif c == "}": brace -= 1
                    elif c == "[": bracket += 1
                    elif c == "]": bracket -= 1

            if brace <= 0 and bracket <= 0:
                continue  # already balanced or over-closed

            balanced = trimmed + "]" * max(0, bracket) + "}" * max(0, brace)
            try:
                json.loads(balanced)
                return balanced
            except json.JSONDecodeError:
                continue

        return None
