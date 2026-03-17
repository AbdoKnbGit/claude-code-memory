"""
tests/conftest.py — shared fixtures
"""
import pytest


@pytest.fixture
def empty_settings():
    return {}


@pytest.fixture
def settings_with_third_party():
    return {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Write",
                    "hooks": [{"type": "command", "command": "docker exec -i other-plugin python /logger.py"}]
                }
            ],
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [{"type": "command", "command": "echo session-ended"}]
                }
            ]
        }
    }


@pytest.fixture
def settings_with_two_plugins():
    return {
        "hooks": {
            "PostToolUse": [
                {"matcher": "", "hooks": [{"type": "command", "command": "docker exec -i plugin-a python /a.py"}]},
                {"matcher": "", "hooks": [{"type": "command", "command": "docker exec -i plugin-b python /b.py"}]},
            ],
            "UserPromptSubmit": [
                {"matcher": "", "hooks": [{"type": "command", "command": "docker exec -i plugin-a python /prompt.py"}]},
            ]
        }
    }
