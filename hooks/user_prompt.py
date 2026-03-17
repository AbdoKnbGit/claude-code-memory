"""
UserPrompt hook — captures user decisions and choices.
Only saves when the user makes an explicit decision or choice.

OPTIMIZED: save() is now fire-and-forget via hook_utils.
This hook returns INSTANTLY — actual save happens in background.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook_utils import get_project_context, read_stdin, save

DECISION_SIGNALS = [
    "let's go with", "i chose", "use ", "switch to",
    "instead of", "we'll use", "go with", "pick ",
    "prefer", "decided", "let's use", "change to",
    "replace with", "remove ", "keep ", "always use",
    "never use", "don't use", "stop using",
    "that failed", "not working", "it broke", "didn't work",
    "fix that", "try again", "revert", "undo",
    "on prend", "j'ai choisi", "utilise ", "passe à",
    "au lieu de", "on va utiliser", "on garde",
    "on supprime", "toujours utiliser", "ne jamais",
    "ça a échoué", "ne marche pas", "ça a cassé",
]


def main():
    data = read_stdin()
    project_id, session_id = get_project_context(data)
    if not project_id:
        return

    user_message = data.get("message", "")
    if not user_message or len(user_message) < 10:
        return

    message_lower = user_message.lower()

    MEMORY_KEYWORDS = [
        "memory_manage", "memory_forget", "memory_delete", "memory_clear",
        "forget entry", "forget id", "delete entry", "delete id",
        "remove entry", "remove id", "unpin entry", "unpin id",
        "/forget", "/delete", "/clear", "/reduce",
        "forget that memory", "delete that memory", "remove that memory",
        "stop saving", "stop remembering", "don't remember",
    ]
    if any(kw in message_lower for kw in MEMORY_KEYWORDS):
        return

    if not any(signal in message_lower for signal in DECISION_SIGNALS):
        return

    text = f"User decision: {user_message[:300]}"

    save(
        text=text,
        project_id=project_id,
        session_id=session_id,
        pin=False,
        trigger=user_message[:100],
        outcome="user_decision",
    )


if __name__ == "__main__":
    main()
