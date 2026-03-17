"""
tests/test_hooks_merge.py

Tests for settings.json hook merge/uninstall logic.
Verifies cc-memory is a good citizen — never destroys other plugins' hooks.

Run with:
    pytest tests/test_hooks_merge.py -v
"""


CONTAINER = "cc-nim-memory"


def _hook_entry(script: str, win: bool = False) -> dict:
    path = "//app/hooks/" if win else "/app/hooks/"
    return {
        "type": "command",
        "command": f"docker exec -i {CONTAINER} python {path}{script}"
    }


def apply_merge(existing_settings: dict, win: bool = False) -> dict:
    """Replicates setup_hooks() merge logic from setup.py."""
    settings = dict(existing_settings)
    existing_hooks = settings.get("hooks", {})

    our_hooks = {
        "PostToolUse":      [{"matcher": "", "hooks": [_hook_entry("post_tool_use.py", win)]}],
        "UserPromptSubmit": [{"matcher": "", "hooks": [_hook_entry("user_prompt.py", win)]}],
    }

    for event, entries in our_hooks.items():
        existing = existing_hooks.get(event, [])
        filtered = [e for e in existing
                    if not any(CONTAINER in h.get("command", "")
                               for h in e.get("hooks", []))]
        existing_hooks[event] = filtered + entries

    settings["hooks"] = existing_hooks
    return settings


def apply_uninstall(existing_settings: dict) -> dict:
    """Replicates uninstall.py hook removal logic."""
    settings = dict(existing_settings)
    hooks = dict(settings.get("hooks", {}))
    for event in list(hooks.keys()):
        before = hooks[event]
        after = [e for e in before
                 if not any(CONTAINER in h.get("command", "")
                            for h in e.get("hooks", []))]
        if after:
            hooks[event] = after
        else:
            del hooks[event]
    settings["hooks"] = hooks
    return settings


class TestMergeInstall:

    def test_fresh_install_empty_settings(self):
        """Fresh machine — no settings.json — creates both hooks."""
        result = apply_merge({})
        hooks = result["hooks"]
        assert "PostToolUse" in hooks
        assert "UserPromptSubmit" in hooks
        assert len(hooks["PostToolUse"]) == 1
        assert len(hooks["UserPromptSubmit"]) == 1

    def test_does_not_register_pretooluse(self):
        """PreToolUse must NEVER be registered — token bomb."""
        result = apply_merge({})
        assert "PreToolUse" not in result["hooks"]

    def test_does_not_register_notification(self):
        """Notification must NEVER be registered — wrong event type."""
        result = apply_merge({})
        assert "Notification" not in result["hooks"]

    def test_preserves_third_party_post_tool_use(self):
        """Another plugin's PostToolUse hook must survive install."""
        other_plugin_hook = {
            "matcher": "Write",
            "hooks": [{"type": "command", "command": "docker exec -i other-plugin python /app/logger.py"}]
        }
        existing = {
            "hooks": {
                "PostToolUse": [other_plugin_hook]
            }
        }
        result = apply_merge(existing)
        post_hooks = result["hooks"]["PostToolUse"]
        assert len(post_hooks) == 2
        commands = [h["hooks"][0]["command"] for h in post_hooks]
        assert any("other-plugin" in c for c in commands), "Other plugin hook was lost"
        assert any(CONTAINER in c for c in commands), "cc-memory hook missing"

    def test_preserves_third_party_user_prompt_submit(self):
        """Another plugin's UserPromptSubmit hook must survive install."""
        other_hook = {
            "matcher": "",
            "hooks": [{"type": "command", "command": "docker exec -i analytics python /track.py"}]
        }
        existing = {"hooks": {"UserPromptSubmit": [other_hook]}}
        result = apply_merge(existing)
        hooks = result["hooks"]["UserPromptSubmit"]
        assert len(hooks) == 2
        commands = [h["hooks"][0]["command"] for h in hooks]
        assert any("analytics" in c for c in commands)
        assert any(CONTAINER in c for c in commands)

    def test_preserves_unrelated_hook_events(self):
        """Hook events cc-memory doesn't use must be completely untouched."""
        existing = {
            "hooks": {
                "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "echo done"}]}],
                "SubagentStop": [{"matcher": "", "hooks": [{"type": "command", "command": "echo sub"}]}],
            }
        }
        result = apply_merge(existing)
        assert "Stop" in result["hooks"]
        assert "SubagentStop" in result["hooks"]
        assert result["hooks"]["Stop"] == existing["hooks"]["Stop"]

    def test_idempotent_reinstall(self):
        """Installing twice must not duplicate hooks."""
        first = apply_merge({})
        second = apply_merge(first)
        assert len(second["hooks"]["PostToolUse"]) == 1
        assert len(second["hooks"]["UserPromptSubmit"]) == 1

    def test_idempotent_reinstall_with_third_party(self):
        """Reinstall with third-party hooks present must not duplicate anything."""
        other = {
            "matcher": "",
            "hooks": [{"type": "command", "command": "docker exec -i other python /x.py"}]
        }
        existing = {"hooks": {"PostToolUse": [other]}}
        first = apply_merge(existing)
        second = apply_merge(first)
        assert len(second["hooks"]["PostToolUse"]) == 2

    def test_win_path_uses_double_slash(self):
        """Windows hooks must use //app/ prefix to prevent Git Bash mangling."""
        result = apply_merge({}, win=True)
        cmd = result["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
        assert "//app/" in cmd, f"Expected //app/ prefix, got: {cmd}"

    def test_linux_path_uses_single_slash(self):
        """Linux/Mac hooks must use /app/ prefix."""
        result = apply_merge({}, win=False)
        cmd = result["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
        assert "/app/" in cmd
        assert "//app/" not in cmd


class TestUninstall:

    def test_removes_cc_memory_post_tool_use(self):
        """Uninstall removes cc-memory's PostToolUse hook."""
        installed = apply_merge({})
        result = apply_uninstall(installed)
        assert "PostToolUse" not in result["hooks"]

    def test_removes_cc_memory_user_prompt_submit(self):
        """Uninstall removes cc-memory's UserPromptSubmit hook."""
        installed = apply_merge({})
        result = apply_uninstall(installed)
        assert "UserPromptSubmit" not in result["hooks"]

    def test_preserves_third_party_on_uninstall(self):
        """Uninstall must not touch other plugins' hooks."""
        other = {
            "matcher": "",
            "hooks": [{"type": "command", "command": "docker exec -i other-plugin python /x.py"}]
        }
        existing = {"hooks": {"PostToolUse": [other]}}
        installed = apply_merge(existing)
        result = apply_uninstall(installed)

        post_hooks = result["hooks"].get("PostToolUse", [])
        commands = [h["hooks"][0]["command"] for h in post_hooks]
        assert not any(CONTAINER in c for c in commands), "cc-memory hook survived uninstall"

        assert any("other-plugin" in c for c in commands), "Other plugin hook was removed"

    def test_preserves_unrelated_events_on_uninstall(self):
        """Events cc-memory never registered must survive uninstall completely."""
        stop_hook = {"matcher": "", "hooks": [{"type": "command", "command": "echo stop"}]}
        installed = apply_merge({"hooks": {"Stop": [stop_hook]}})
        result = apply_uninstall(installed)
        assert "Stop" in result["hooks"]
        assert result["hooks"]["Stop"] == [stop_hook]

    def test_uninstall_on_clean_machine(self):
        """Uninstall on a machine where cc-memory was never installed — no crash."""
        result = apply_uninstall({})
        assert result == {"hooks": {}}

    def test_uninstall_idempotent(self):
        """Running uninstall twice must not crash or change anything."""
        installed = apply_merge({})
        first = apply_uninstall(installed)
        second = apply_uninstall(first)
        assert first == second

    def test_empty_event_key_cleaned_up(self):
        """After removing last entry from an event, the key itself is removed."""
        installed = apply_merge({})
        result = apply_uninstall(installed)
        for event, entries in result.get("hooks", {}).items():
            assert len(entries) > 0, f"Empty hook event key left: {event}"


class TestClaudeMdMerge:

    def _do_merge(self, our_content: str, existing_content: str) -> str:
        """Replicates setup_claude_md() prepend logic."""
        merged = our_content.rstrip() + "\n\n---\n# (existing content preserved below)\n\n" + existing_content
        return merged

    def test_prepend_preserves_existing_content(self):
        existing = "# My custom rules\nDo X before Y."
        our = "# CC-Memory Rules\nBoot sequence here."
        merged = self._do_merge(our, existing)
        assert "My custom rules" in merged
        assert "CC-Memory Rules" in merged

    def test_cc_memory_content_comes_first(self):
        existing = "# Existing"
        our = "# CC-Memory"
        merged = self._do_merge(our, existing)
        assert merged.index("CC-Memory") < merged.index("Existing")

    def test_separator_present(self):
        merged = self._do_merge("# Ours", "# Theirs")
        assert "existing content preserved below" in merged

    def test_no_content_lost(self):
        our = "BOOT SEQUENCE"
        existing = "CUSTOM USER RULES\nLINE TWO\nLINE THREE"
        merged = self._do_merge(our, existing)
        for line in existing.splitlines():
            assert line in merged, f"Line lost: {line}"
