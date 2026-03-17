
import os
import re

def verify_session_cleanup():
    routes_path = "api/routes.py"
    if not os.path.exists(routes_path):
        print(f"Error: {routes_path} not found")
        return

    with open(routes_path, "r", encoding="utf-8") as f:
        content = f.read()

    dict_patterns = [
        r"^(_\w+):\s*(?:_OD|dict)\[str,",
        r"^(_\w+)\s*=\s*_OD\(\)",
    ]
    
    found_dicts = set()
    for line in content.splitlines():
        for pattern in dict_patterns:
            match = re.match(pattern, line)
            if match:
                name = match.group(1)
                if name not in ["_MODEL_CONTEXT_WINDOWS", "_MODEL_PRICING", "_memory_hash_cache", "_project_meta"]:
                    found_dicts.add(name)

    session_key_usage = set(re.findall(r'(_\w+)\[session_id\]', content))
    session_key_usage.update(re.findall(r'(_\w+)\.get\(session_id', content))
    session_key_usage.update(re.findall(r'(_\w+)\.pop\(session_id', content))
    
    session_dicts = {d for d in session_key_usage if d.startswith("_") and f"\n{d}" in content}
    
    all_potential_session_dicts = found_dicts.union(session_dicts)
    
    excluded = ["_REPR", "_RE", "_OD", "_init_project_meta_table", "_persist_project_meta", "_get_memory_hash", "_stop_keepalive"]
    all_potential_session_dicts = {d for d in all_potential_session_dicts if d not in excluded}

    stop_match = re.search(r"def _stop_keepalive\(session_id: str\):(.*?)def ", content, re.DOTALL)
    if not stop_match:
        stop_match = re.search(r"def _stop_keepalive\(session_id: str\):(.*)", content, re.DOTALL)
        
    if not stop_match:
        print("Error: _stop_keepalive not found")
        return

    stop_body = stop_match.group(1)
    
    popped_dicts = set(re.findall(r'(_\w+)\.pop\(session_id', stop_body))
    
    missing = all_potential_session_dicts - popped_dicts - {"_keepalive_tasks", "_session_last_msg", "_session_last_user_msg"}
    
    print("Potential Session Dicts:", sorted(list(all_potential_session_dicts)))
    print("Popped in _stop_keepalive:", sorted(list(popped_dicts)))
    
    actual_missing = []
    for m in missing:
        if f"{m}[session_id]" in content or f"{m}.get(session_id)" in content:
            actual_missing.append(m)
            
    if actual_missing:
        print("MISSING from _stop_keepalive:", sorted(actual_missing))
        return False
    else:
        print("Verification SUCCESS: All detected session dicts are covered in _stop_keepalive.")
        return True

if __name__ == "__main__":
    if verify_session_cleanup():
        exit(0)
    else:
        exit(1)
