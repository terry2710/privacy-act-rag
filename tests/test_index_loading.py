from core import rag_backend


def test_returns_none_without_building_when_nothing_is_cached(tmp_path, monkeypatch):
    def unexpected_build(url):
        raise AssertionError("loading a cached index must never build one")

    monkeypatch.setattr(rag_backend, "INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(rag_backend, "S3_BUCKET", "")
    monkeypatch.setattr(rag_backend, "PyPDFLoader", unexpected_build)
    monkeypatch.setattr(rag_backend, "_make_embeddings", lambda: object())
    assert rag_backend.prv_load_cached_index() is None


def test_loads_local_cache(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    for name in rag_backend.INDEX_FILES:
        (index_dir / name).write_bytes(b"x")
    monkeypatch.setattr(rag_backend, "INDEX_DIR", str(index_dir))
    monkeypatch.setattr(rag_backend, "_make_embeddings", lambda: object())
    monkeypatch.setattr(rag_backend.FAISS, "load_local", lambda *args, **kwargs: "INDEX")
    assert rag_backend.prv_load_cached_index() == "INDEX"
