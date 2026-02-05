from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional


@dataclass(frozen=True)
class HarmonyMessage:
    role: str
    content: str
    channel: Optional[str] = None
    to: Optional[str] = None


def default_system_message(
    *,
    knowledge_cutoff: str = "2024-06",
    current_date: Optional[str] = None,
    reasoning: str = "medium",
    channels: Iterable[str] = ("analysis", "commentary", "final"),
    tools_required: bool = False,
) -> str:
    if current_date is None:
        current_date = date.today().isoformat()
    lines = [
        "You are ChatGPT, a large language model trained by OpenAI.",
        f"Knowledge cutoff: {knowledge_cutoff}",
        f"Current date: {current_date}",
        "",
        f"Reasoning: {reasoning}",
        "",
        f"# Valid channels: {', '.join(channels)}. Channel must be included for every message.",
    ]
    if tools_required:
        lines.append("Calls to these tools must go to the commentary channel: 'functions'.")
    return "\n".join(lines)


def render_harmony(
    messages: Iterable[HarmonyMessage],
    *,
    add_assistant_start: bool = True,
    assistant_channel: Optional[str] = None,
) -> str:
    parts: list[str] = []
    for msg in messages:
        header = msg.role
        if msg.to:
            header += f" to={msg.to}"
        if msg.channel:
            header += f"<|channel|>{msg.channel}"
        parts.append(f"<|start|>{header}<|message|>{msg.content}<|end|>")
    if add_assistant_start:
        header = "assistant"
        if assistant_channel:
            header += f"<|channel|>{assistant_channel}"
        parts.append(f"<|start|>{header}")
    return "".join(parts)


def load_conversation_json(path: str) -> list[HarmonyMessage]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Conversation JSON must be a list of message objects.")

    messages: list[HarmonyMessage] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Message at index {idx} must be an object.")
        role = item.get("role")
        content = item.get("content")
        if not role or content is None:
            raise ValueError(f"Message at index {idx} requires 'role' and 'content'.")
        messages.append(
            HarmonyMessage(
                role=str(role),
                content=str(content),
                channel=item.get("channel"),
                to=item.get("to"),
            )
        )
    return messages
