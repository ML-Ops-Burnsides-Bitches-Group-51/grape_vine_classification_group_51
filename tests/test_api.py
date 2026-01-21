# tests/integrationtests/test_apis.py
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grape_vine_classification.api import app  # <- adjust if your module path differs


def make_test_image_bytes(fmt="PNG", size=(64, 64)) -> bytes:
    """Create a small in-memory image to upload as if it was a user file."""
    img = Image.new("RGB", size)
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="module")
def client():
    # If you use FastAPI lifespan events, use the context manager.
    # The DTU guide calls this out explicitly. :contentReference[oaicite:3]{index=3}
    with TestClient(app) as c:
        yield c


# def test_read_root(client):
#     r = client.get("/")
#     assert r.status_code == 200

#     # If your root returns something else, update this assertion.
#     # Example:
#     # assert r.json() == {"message": "Welcome ..."}
#     assert isinstance(r.json(), dict)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_happy_path(client):
    img_bytes = make_test_image_bytes()

    files = [("files", ("test.png", img_bytes, "image/png"))]

    r = client.post("/predict", files=files)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 1

    first = data["results"][0]
    assert "predicted_label" in first
    assert "confidence" in first
    assert "filename" in first
    assert first["filename"] == "test.png"

    assert 0.0 <= float(first["confidence"]) <= 1.0
    assert isinstance(first.get("top_k", []), list)

    # If you added filename to the response model earlier:
    # assert data["filename"] == "test.png"

    # If you renamed top_k to something else, assert that instead:
    # assert "top_k" in data
    # assert isinstance(data["top_k"], list)


def test_predict_rejects_non_image(client):
    files = {"file": ("not_image.txt", b"hello", "text/plain")}
    r = client.post("/predict", files=files)

    # Depending on your implementation, this might be 400 or 422.
    assert r.status_code in (400, 415, 422)
