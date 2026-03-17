
import json

class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    def model_dump(self):
        return {"role": self.role, "content": self.content}

def _estimate_msg_tokens(msg) -> int:
    """Rough token estimate for a message — recursively includes tool results. Caches result."""
    if hasattr(msg, "_estimated_tokens"):
        return msg._estimated_tokens
    if isinstance(msg, dict) and "_estimated_tokens" in msg:
        return msg["_estimated_tokens"]

    if hasattr(msg, "model_dump"):
        d = msg.model_dump()
    elif isinstance(msg, dict):
        d = msg
    else:
        return max(1, len(str(msg)) // 3)

    role = d.get("role", "user")
    content = d.get("content", "")

    _CHARS_PER_TOKEN = 3

    def _count_recursive(obj) -> int:
        if isinstance(obj, str):
            return len(obj) // _CHARS_PER_TOKEN
        if isinstance(obj, list):
            return sum(_count_recursive(item) for item in obj)
        if isinstance(obj, dict):
            t = obj.get("type", "")
            if t == "text":
                return len(obj.get("text", "")) // _CHARS_PER_TOKEN
            if t == "tool_use":
                name_tok = len(obj.get("name", "")) // _CHARS_PER_TOKEN
                inp = obj.get("input", {})
                inp_tok = len(json.dumps(inp, default=str)) // _CHARS_PER_TOKEN if isinstance(inp, dict) else len(str(inp)) // _CHARS_PER_TOKEN
                return name_tok + inp_tok
            if t == "tool_result":
                inner = obj.get("content", "")
                return _count_recursive(inner)
            return len(obj.get("text", "")) // _CHARS_PER_TOKEN if "text" in obj else 0
        return 0

    total = max(1, _count_recursive(content))

    if hasattr(msg, "__dict__"):
        try:
            msg._estimated_tokens = total
        except (AttributeError, TypeError):
            pass
    elif isinstance(msg, dict):
        msg["_estimated_tokens"] = total

    return total

def test_token_cache():
    msg_obj = MockMessage("user", "Hello world")
    tokens1 = _estimate_msg_tokens(msg_obj)
    assert hasattr(msg_obj, "_estimated_tokens")
    assert msg_obj._estimated_tokens == tokens1
    
    msg_obj.content = "Something long"
    tokens2 = _estimate_msg_tokens(msg_obj)
    assert tokens2 == tokens1
    print("Object caching verified.")

    msg_dict = {"role": "user", "content": "Hello world"}
    tokens3 = _estimate_msg_tokens(msg_dict)
    assert "_estimated_tokens" in msg_dict
    assert msg_dict["_estimated_tokens"] == tokens3
    
    msg_dict["content"] = "Something long"
    tokens4 = _estimate_msg_tokens(msg_dict)
    assert tokens4 == tokens3
    print("Dict caching verified.")

if __name__ == "__main__":
    test_token_cache()
    print("All tests passed.")
