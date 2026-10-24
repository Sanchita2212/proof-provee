import pytest
import io
import zipfile
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_empty_zip_rejection():
    # Test uploading an empty zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        pass
    buf.seek(0)
    files = {"file": ("empty.zip", buf.getvalue(), "application/zip")}
    data = {"collection_name": "empty_collection"}
    resp = client.post("/index", data=data, files=files)
    assert resp.status_code == 400

def test_corrupted_zip_rejection():
    files = {"file": ("corrupt.zip", b"not a zip file header content", "application/zip")}
    data = {"collection_name": "corrupt_collection"}
    resp = client.post("/index", data=data, files=files)
    assert resp.status_code == 400

def test_path_traversal_sanitization():
    # Attempt directory traversal via collection_name
    q_resp = client.post("/query", json={
        "collection_name": "../../windows/system32",
        "query": "test"
    })
    # Should safely fail as collection not found, not crash or traverse
    assert q_resp.status_code in (400, 404)

def test_empty_query_rejection():
    resp = client.post("/query", json={
        "collection_name": "pytest_test_collection",
        "query": "   "
    })
    assert resp.status_code == 400
