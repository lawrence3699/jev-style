"""server.json (MCP Registry listing) stays in step with the package and the README."""
import json
from pathlib import Path

import pytest

import jev_style

ROOT = Path(__file__).resolve().parents[1]
SERVER_JSON = ROOT / "server.json"

pytestmark = pytest.mark.skipif(not SERVER_JSON.exists(), reason="server.json is only in the git checkout")


def test_versions_match_package():
    d = json.loads(SERVER_JSON.read_text())
    assert d["version"] == jev_style.__version__
    assert [p["version"] for p in d["packages"]] == [jev_style.__version__]
    assert d["packages"][0]["identifier"] == "jev-style"


def test_readme_carries_the_registry_name():
    d = json.loads(SERVER_JSON.read_text())
    assert f"<!-- mcp-name: {d['name']} -->" in (ROOT / "README.md").read_text()
    assert len(d["description"]) <= 100
