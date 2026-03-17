"""
SessionStart hook — DISABLED for performance.

The proxy (api/routes.py) already injects memory context into the system
prompt on every /v1/messages call.  This hook was doing a SECOND redundant
ChromaDB search + embedding call, doubling latency and compute cost.

Memory injection now happens exclusively in the proxy layer.
"""

import json
import sys


def main():
    try:
        sys.stdin.read()
    except Exception:
        pass
    print(json.dumps({"additionalContext": ""}))


if __name__ == "__main__":
    main()
