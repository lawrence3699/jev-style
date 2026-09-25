import asyncio

import pytest

pytest.importorskip("mcp")

from jev_style.mcp_server import InProcessBackend, build_server  # noqa: E402


def test_tools_listed_and_callable():
    srv = build_server(InProcessBackend(fake=True))
    tools = {t.name for t in asyncio.run(srv.list_tools())}
    assert {"decide", "noul", "choice", "score", "model_info"} <= tools
    res = asyncio.run(srv.call_tool("choice", {"state": "charged twice", "question": "Team?",
                                               "options": ["billing", "tech"]}))
    assert not getattr(res, "isError", False)
