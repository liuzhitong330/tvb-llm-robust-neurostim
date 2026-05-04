"""JSON parsing helpers for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_fence(text: str) -> str:
    """Remove common markdown code fences without changing inner JSON."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def parse_json_response(text: str, *, context: str = "response") -> Any:
    """Parse JSON returned by an LLM with bounded repair for common failures."""

    candidate = strip_markdown_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    extracted = extract_first_json_value(candidate)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            repaired = repair_truncated_json(extracted)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from {context}: {candidate[:500]}")


def extract_first_json_value(text: str) -> str | None:
    """Return the first top-level JSON object or array embedded in text."""

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack: list[str] = []
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            expected = "{" if char == "}" else "["
            if stack[-1] != expected:
                return None
            stack.pop()
            if not stack and char == closing:
                return text[start : idx + 1]

    return text[start:]


def repair_truncated_json(text: str) -> str:
    """Close unclosed JSON containers after trimming partial trailing content."""

    text = text.rstrip().rstrip(",")
    last_complete = max(text.rfind("}"), text.rfind("]"))
    if last_complete >= 0:
        text = text[: last_complete + 1]

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()

    if in_string:
        text += '"'
    for char in reversed(stack):
        text += "}" if char == "{" else "]"
    return text
