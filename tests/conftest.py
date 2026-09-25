import pytest
from fastapi.testclient import TestClient

from jev_style.server import build_app


@pytest.fixture(scope="session")
def app():
    return build_app(fake=True)


@pytest.fixture()
def client(app):
    return TestClient(app)
