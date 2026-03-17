"""
hooks/post_tool_use.py

PostToolUse hook — auto-captures ALL meaningful tool outcomes.
Uses smart_capture.py for intelligent scoring: saves minimum, keeps maximum value.

LATENCY: Returns in <5ms. DB save is fire-and-forget background thread.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook_utils import get_project_context, read_stdin, save
from memory_graph_utils import detect_component_from_text
from smart_capture import should_save


def main():
    data = read_stdin()
    project_id, session_id = get_project_context(data)
    if not project_id:
        return

    tool_name  = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    tool_output = str(data.get("tool_output", ""))

    file_path = ""
    command   = ""
    text      = ""
    pin       = False

    if tool_name in ("Write", "Edit", "MultiEdit"):
        file_path = tool_input.get("file_path", "")
        content_preview = str(tool_input.get("content", ""))[:300]
        text = f"File {tool_name.lower()}: {file_path}"
        if content_preview:
            text += f" — {content_preview}"

    elif tool_name == "Bash":
        command = tool_input.get("command", "")
        text = f"Ran: {command[:150]}"
        if tool_output.strip():
            text += f" → {tool_output[:200]}"

    elif tool_name in ("Read", "Glob"):
        return

    elif tool_name in ("WebSearch", "web_search"):
        query = tool_input.get("query", "")
        text = f"Researched: {query}"

    else:
        return

    if not text.strip():
        return

    ok, score, reason = should_save(
        text=text,
        tool_name=tool_name,
        file_path=file_path,
        command=command,
        output=tool_output,
    )

    if not ok:
        return

    component = detect_component_from_text(text, trigger=file_path or command)
    pin = score >= 0.85

    save(
        text=text,
        project_id=project_id,
        session_id=session_id,
        pin=pin,
        trigger=file_path or command or tool_name,
        outcome=f"{tool_name.lower()}_{component}",
        component=component,
    )


if __name__ == "__main__":
    main()
