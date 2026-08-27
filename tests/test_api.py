import numpy as np
import pytest
from fastapi.testclient import TestClient

from diagramchess.board import BoardMatrix
from diagramchess.pipeline import ingest
from diagramchess.review.app import create_app

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def client(workspace, demo_pdf):
    ingest(workspace, demo_pdf)
    return TestClient(create_app(workspace)), workspace


def test_stats_reports_the_workspace(client):
    api, _ = client
    body = api.get("/api/stats").json()
    assert body["stats"]["diagrams"] > 0
    assert body["model"] is None
    assert len(body["labels"]) == 13


def test_queue_orders_by_doubt(client):
    api, workspace = client
    rows = workspace.diagrams()
    workspace.set_prediction(int(rows[0]["id"]), ["."] * 64, [0.99] * 64, "f", "white", "w", None)
    workspace.set_prediction(int(rows[1]["id"]), ["."] * 64, [0.2] * 64, "f", "white", "w", None)
    body = api.get("/api/queue?order=uncertain").json()
    ids = [d["id"] for d in body["diagrams"] if d["min_confidence"] is not None]
    assert ids[0] == int(rows[1]["id"])


def test_diagram_returns_sixty_four_cells_with_their_crops(client):
    api, workspace = client
    diagram_id = int(workspace.diagrams()[0]["id"])
    body = api.get(f"/api/diagram/{diagram_id}").json()
    assert len(body["cells"]) == 64
    assert body["cells"][0]["image"].startswith("data:image/png;base64,")
    assert body["cells"][0]["row"] == 0 and body["cells"][0]["col"] == 0
    assert body["cells"][63]["row"] == 7 and body["cells"][63]["col"] == 7


def test_unknown_diagram_is_a_404(client):
    api, _ = client
    assert api.get("/api/diagram/999999").status_code == 404


def test_crop_and_page_images_are_served(client):
    api, workspace = client
    diagram_id = int(workspace.diagrams()[0]["id"])
    crop = api.get(f"/api/crop/{diagram_id}")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/png"
    page = api.get(f"/api/page/{diagram_id}")
    assert page.status_code == 200 and page.headers["content-type"] == "image/png"


def test_saving_a_review_records_labels_and_returns_the_position(client):
    api, workspace = client
    diagram_id = int(workspace.diagrams()[0]["id"])
    workspace.set_prediction(diagram_id, ["."] * 64, [0.5] * 64, "f", "white", "w", None)
    labels = BoardMatrix.from_fen(START).flat()

    body = api.post(f"/api/diagram/{diagram_id}/save",
                    json={"labels": labels, "orientation": "white", "side_to_move": "w"}).json()
    assert body["fen"] == START
    assert body["lichess"].startswith("https://lichess.org/analysis/")
    assert body["problems"] == []
    assert body["corrections"] == 32          # every piece square disagreed with '.'
    assert workspace.diagram(diagram_id)["status"] == "verified"
    assert len(workspace.verified_squares()) == 64


def test_saving_rejects_a_malformed_board(client):
    api, workspace = client
    diagram_id = int(workspace.diagrams()[0]["id"])
    assert api.post(f"/api/diagram/{diagram_id}/save", json={"labels": ["."] * 63}).status_code == 400
    assert api.post(f"/api/diagram/{diagram_id}/save",
                    json={"labels": ["Z"] + ["."] * 63}).status_code == 400
    assert api.post(f"/api/diagram/{diagram_id}/save",
                    json={"labels": ["."] * 64, "orientation": "sideways"}).status_code == 400
    assert api.post(f"/api/diagram/{diagram_id}/save",
                    json={"labels": ["."] * 64, "side_to_move": "x"}).status_code == 400


def test_rejecting_a_detection(client):
    api, workspace = client
    diagram_id = int(workspace.diagrams()[0]["id"])
    assert api.post(f"/api/diagram/{diagram_id}/status", json={"status": "rejected"}).status_code == 200
    assert workspace.diagram(diagram_id)["status"] == "rejected"
    assert api.post(f"/api/diagram/{diagram_id}/status", json={"status": "nonsense"}).status_code == 400


def test_reread_needs_a_model(client):
    api, workspace = client
    diagram_id = int(workspace.diagrams()[0]["id"])
    response = api.post(f"/api/diagram/{diagram_id}/reread", json={})
    assert response.status_code == 400
    assert "no model" in response.json()["detail"]


def test_fen_endpoint_converts_and_warns():
    from diagramchess.store import Workspace
    import tempfile

    api = TestClient(create_app(Workspace(tempfile.mkdtemp())))
    labels = BoardMatrix.from_fen(START).flat()
    body = api.post("/api/fen", json={"labels": labels, "side_to_move": "b"}).json()
    assert body["fen"].endswith("b KQkq - 0 1")
    assert body["counts"]["P"] == 8
    assert body["problems"] == []

    body = api.post("/api/fen", json={"labels": ["."] * 64}).json()
    assert any("king" in p for p in body["problems"])

    assert api.post("/api/fen", json={"labels": ["."] * 10}).status_code == 400


def test_the_pages_are_served():
    from diagramchess.store import Workspace
    import tempfile

    api = TestClient(create_app(Workspace(tempfile.mkdtemp())))
    for path in ("/", "/review"):
        response = api.get(path)
        assert response.status_code == 200
        assert "diagramchess" in response.text
