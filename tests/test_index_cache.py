"""Index cache resolution: local disk, S3 restore, and the fallback to a rebuild. No AWS.

The deployed app crashed inside faiss.read_index() because a failed S3 download had already
created an empty faiss_index/, and the loader treated the bare directory as a valid cache
instead of rebuilding. These tests pin the recovery behaviour.
"""
import os

import pytest

import rag_backend


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Point rag_backend at a scratch index directory."""
    target = tmp_path / "faiss_index"
    monkeypatch.setattr(rag_backend, "INDEX_DIR", str(target))
    monkeypatch.setattr(rag_backend, "S3_BUCKET", "test-bucket")
    return target


def write_index(directory, faiss_bytes=b"FAISSDATA", pkl_bytes=b"PKLDATA"):
    os.makedirs(directory, exist_ok=True)
    (directory / "index.faiss").write_bytes(faiss_bytes)
    (directory / "index.pkl").write_bytes(pkl_bytes)


class FakeS3:
    """Stands in for the boto3 S3 client; can fail on a chosen file."""

    def __init__(self, fail_on=None, payloads=None):
        self.fail_on = fail_on
        self.payloads = payloads or {"index.faiss": b"FROM-S3-FAISS", "index.pkl": b"FROM-S3-PKL"}
        self.downloaded = []

    def download_file(self, bucket, key, destination):
        name = key.rsplit("/", 1)[-1]
        self.downloaded.append(name)
        if name == self.fail_on:
            raise RuntimeError("AccessDenied")
        with open(destination, "wb") as f:
            f.write(self.payloads[name])


@pytest.fixture
def s3(monkeypatch):
    def install(**kwargs):
        client = FakeS3(**kwargs)
        monkeypatch.setattr(rag_backend, "_s3_client", lambda: client)
        return client
    return install


# ------------------------------------------------------------ cache detection

def test_complete_cache_is_recognised(index_dir):
    write_index(index_dir)
    assert rag_backend._index_is_cached() is True


def test_missing_directory_is_not_a_cache(index_dir):
    assert rag_backend._index_is_cached() is False


def test_empty_directory_is_not_a_cache(index_dir):
    """The exact state a failed download used to leave behind."""
    os.makedirs(index_dir)
    assert rag_backend._index_is_cached() is False


def test_partial_cache_is_not_a_cache(index_dir):
    os.makedirs(index_dir)
    (index_dir / "index.faiss").write_bytes(b"FAISSDATA")
    assert rag_backend._index_is_cached() is False, "index.pkl is missing"


def test_zero_byte_file_is_not_a_cache(index_dir):
    write_index(index_dir, faiss_bytes=b"")
    assert rag_backend._index_is_cached() is False


# --------------------------------------------------------------- S3 restore

def test_successful_download_populates_the_cache(index_dir, s3):
    client = s3()
    assert rag_backend._download_index_from_s3(lambda *a: None) is True
    assert rag_backend._index_is_cached() is True
    assert (index_dir / "index.faiss").read_bytes() == b"FROM-S3-FAISS"
    assert client.downloaded == ["index.faiss", "index.pkl"]


def test_failed_download_leaves_no_usable_cache(index_dir, s3, capsys):
    """The regression: a failure must not leave anything the loader would pick up."""
    s3(fail_on="index.pkl")
    assert rag_backend._download_index_from_s3(lambda *a: None) is False
    assert rag_backend._index_is_cached() is False, \
        "a half-finished download must not look like a cache"
    assert not os.path.exists(str(index_dir) + ".partial"), "staging directory was left behind"
    assert "could not restore the index" in capsys.readouterr().err, \
        "the reason must reach stderr, not only the Streamlit status line"


def test_failed_download_preserves_an_existing_cache(index_dir, s3):
    """A restore attempt must not destroy a cache that already works."""
    write_index(index_dir, faiss_bytes=b"ORIGINAL")
    s3(fail_on="index.faiss")

    assert rag_backend._download_index_from_s3(lambda *a: None) is False
    assert rag_backend._index_is_cached() is True
    assert (index_dir / "index.faiss").read_bytes() == b"ORIGINAL"


def test_download_is_skipped_without_a_bucket(index_dir, monkeypatch, s3):
    client = s3()
    monkeypatch.setattr(rag_backend, "S3_BUCKET", "")
    assert rag_backend._download_index_from_s3(lambda *a: None) is False
    assert client.downloaded == []
    assert not os.path.exists(index_dir), "no directory should be created when S3 is disabled"


def test_stale_staging_directory_is_cleared(index_dir, s3):
    staging = str(index_dir) + ".partial"
    os.makedirs(staging)
    with open(os.path.join(staging, "index.faiss"), "wb") as f:
        f.write(b"JUNK")
    s3()

    assert rag_backend._download_index_from_s3(lambda *a: None) is True
    assert (index_dir / "index.faiss").read_bytes() == b"FROM-S3-FAISS"


# ------------------------------------------------------- prv_index behaviour

def test_prv_index_rebuilds_when_the_download_fails(index_dir, s3, monkeypatch):
    """Rather than crashing in faiss.read_index(), it must fall through to a rebuild."""
    s3(fail_on="index.faiss")
    monkeypatch.setattr(rag_backend, "BedrockEmbeddings", lambda **kw: object())

    def unexpected_load(*a, **kw):
        raise AssertionError("load_local was called without a valid cache")

    monkeypatch.setattr(rag_backend.FAISS, "load_local", unexpected_load)

    sentinel = object()
    monkeypatch.setattr(rag_backend, "PyPDFLoader", lambda url: _raise_rebuild(sentinel))

    with pytest.raises(_RebuildReached):
        rag_backend.prv_index()


class _RebuildReached(Exception):
    pass


def _raise_rebuild(_sentinel):
    raise _RebuildReached()


def test_prv_index_loads_a_valid_cache(index_dir, monkeypatch):
    write_index(index_dir)
    monkeypatch.setattr(rag_backend, "BedrockEmbeddings", lambda **kw: object())
    loaded = {}

    def fake_load(path, embeddings, **kw):
        loaded["path"] = path
        return "INDEX"

    monkeypatch.setattr(rag_backend.FAISS, "load_local", fake_load)
    assert rag_backend.prv_index() == "INDEX"
    assert loaded["path"] == str(index_dir)
