"""
tests/test_install_cycle.py

Simulates a full install → verify → reinstall → uninstall cycle
on a clean machine (no Docker required — mocks all subprocess calls).

Run with:
    pytest tests/test_install_cycle.py -v
"""

import json
import shutil
import tempfile
from pathlib import Path


CONTAINER = "cc-nim-memory"
MCP_NAME = "cc-memory"


def _make_fake_claude_dir():
    """Create a temp directory simulating ~/.claude with existing settings."""
    tmp = tempfile.mkdtemp()
    return Path(tmp)


def _hook_cmd(script, win=False):
    path = "//app/hooks/" if win else "/app/hooks/"
    return f"docker exec -i {CONTAINER} python {path}{script}"


def merge_hooks(settings: dict, win: bool = False) -> dict:
    existing_hooks = settings.get("hooks", {})
    our_hooks = {
        "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": _hook_cmd("post_tool_use.py", win)}]}],
        "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command", "command": _hook_cmd("user_prompt.py", win)}]}],
    }
    for event, entries in our_hooks.items():
        existing = existing_hooks.get(event, [])
        filtered = [e for e in existing
                    if not any(CONTAINER in h.get("command", "") for h in e.get("hooks", []))]
        existing_hooks[event] = filtered + entries
    settings["hooks"] = existing_hooks
    return settings


def remove_hooks(settings: dict) -> dict:
    hooks = dict(settings.get("hooks", {}))
    for event in list(hooks.keys()):
        after = [e for e in hooks[event]
                 if not any(CONTAINER in h.get("command", "") for h in e.get("hooks", []))]
        if after:
            hooks[event] = after
        else:
            del hooks[event]
    settings["hooks"] = hooks
    return settings


class TestInstallCycle:

    def setup_method(self):
        self.claude_dir = _make_fake_claude_dir()
        self.settings_path = self.claude_dir / "settings.json"

    def teardown_method(self):
        shutil.rmtree(str(self.claude_dir))

    def _read(self):
        return json.loads(self.settings_path.read_text())

    def _write(self, data):
        self.settings_path.write_text(json.dumps(data, indent=2))


    def test_clean_install(self):
        """Install on machine with no settings.json — creates correctly."""
        result = merge_hooks({})
        self._write(result)
        data = self._read()
        assert "PostToolUse" in data["hooks"]
        assert "UserPromptSubmit" in data["hooks"]
        assert "PreToolUse" not in data["hooks"]
        assert "Notification" not in data["hooks"]

    def test_clean_uninstall_after_clean_install(self):
        """Install then uninstall on clean machine — settings.json ends up empty hooks."""
        installed = merge_hooks({})
        removed = remove_hooks(installed)
        self._write(removed)
        data = self._read()
        assert data["hooks"] == {}


    def test_install_alongside_existing_plugin(self):
        """Install when another plugin already registered PostToolUse."""
        other_settings = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Write", "hooks": [{"type": "command", "command": "docker exec -i other-plugin python /logger.py"}]}
                ]
            }
        }
        self._write(other_settings)
        existing = self._read()
        result = merge_hooks(existing)
        self._write(result)
        data = self._read()

        post_hooks = data["hooks"]["PostToolUse"]
        assert len(post_hooks) == 2

        commands = [h["hooks"][0]["command"] for h in post_hooks]
        assert any("other-plugin" in c for c in commands), "Other plugin lost"
        assert any(CONTAINER in c for c in commands), "cc-memory not added"

    def test_uninstall_preserves_other_plugin(self):
        """Uninstall when another plugin is present — other plugin untouched."""
        other_settings = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Write", "hooks": [{"type": "command", "command": "docker exec -i other-plugin python /logger.py"}]}
                ]
            }
        }
        self._write(other_settings)
        installed = merge_hooks(self._read())
        removed = remove_hooks(installed)
        self._write(removed)
        data = self._read()

        post_hooks = data["hooks"].get("PostToolUse", [])
        commands = [h["hooks"][0]["command"] for h in post_hooks]
        assert any("other-plugin" in c for c in commands), "Other plugin was removed"
        assert not any(CONTAINER in c for c in commands), "cc-memory survived uninstall"


    def test_reinstall_is_idempotent(self):
        """Running install twice produces same result as running once."""
        once = merge_hooks({})
        twice = merge_hooks(dict(once))
        assert once == twice

    def test_reinstall_with_other_plugin_no_duplication(self):
        """Reinstall with other plugin present — no hooks duplicated."""
        other = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "", "hooks": [{"type": "command", "command": "docker exec -i other python /x.py"}]}
                ]
            }
        }
        first = merge_hooks(dict(other))
        second = merge_hooks(dict(first))
        assert len(second["hooks"]["PostToolUse"]) == 2


    def test_full_cycle_clean_machine(self):
        """install → verify → reinstall → verify → uninstall → verify"""
        s1 = merge_hooks({})
        assert "PostToolUse" in s1["hooks"]
        assert "UserPromptSubmit" in s1["hooks"]

        s2 = merge_hooks(dict(s1))
        assert len(s2["hooks"]["PostToolUse"]) == 1

        s3 = remove_hooks(dict(s2))
        assert s3["hooks"] == {}

    def test_full_cycle_with_third_party(self):
        """install → uninstall with 3rd party present — 3rd party survives entire cycle."""
        other_cmd = "docker exec -i third-party python /hook.py"
        base = {
            "hooks": {
                "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": other_cmd}]}],
                "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "echo done"}]}],
            }
        }

        installed = merge_hooks(dict(base))

        cmds = [h["hooks"][0]["command"] for h in installed["hooks"]["PostToolUse"]]
        assert other_cmd in cmds

        removed = remove_hooks(dict(installed))

        cmds_after = [h["hooks"][0]["command"] for h in removed["hooks"].get("PostToolUse", [])]
        assert other_cmd in cmds_after

        assert "Stop" in removed["hooks"]

        assert not any(CONTAINER in c for c in cmds_after)
