import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import codelens.main as main_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the app at a throwaway index file so tests never touch the real
    # index.pkl that a developer might have built locally.
    fake_index_path = str(tmp_path / "test_index.pkl")
    monkeypatch.setattr(main_module, "DEFAULT_INDEX_PATH", fake_index_path)
    monkeypatch.setattr(main_module, "_index", main_module.SearchIndex())
    monkeypatch.setattr(main_module, "_index_loaded_from_disk", False)
    return TestClient(main_module.app)


@pytest.fixture
def toy_repo_path(tmp_path):
    repo = tmp_path / "toy_repo"
    repo.mkdir()
    (repo / "greetings.py").write_text(
        'def say_hello(name):\n    """Return a friendly greeting for name."""\n    return f"hello {name}"\n'
    )
    return str(repo)


def test_health_before_index_built(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_built"] is False


def test_search_before_index_returns_400(client):
    resp = client.post("/search", json={"query": "hello", "top_k": 3})
    assert resp.status_code == 400


def test_index_then_search_round_trip(client, toy_repo_path):
    resp = client.post("/index", json={"path": toy_repo_path})
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_chunks"] >= 1
    assert body["num_files"] == 1

    resp2 = client.get("/health")
    assert resp2.json()["index_built"] is True

    resp3 = client.post("/search", json={"query": "friendly greeting", "top_k": 3})
    assert resp3.status_code == 200
    results = resp3.json()["results"]
    assert len(results) >= 1
    assert results[0]["qualname"] == "say_hello"


def test_index_nonexistent_path_returns_400(client):
    resp = client.post("/index", json={"path": "/definitely/not/a/real/path"})
    assert resp.status_code == 400


def test_search_top_k_is_respected(client, toy_repo_path):
    client.post("/index", json={"path": toy_repo_path})
    resp = client.post("/search", json={"query": "hello", "top_k": 1})
    assert len(resp.json()["results"]) == 1


def test_stats_endpoint_after_indexing(client, toy_repo_path):
    client.post("/index", json={"path": toy_repo_path})
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "vocabulary_size" in body
    assert "svd_components" in body
