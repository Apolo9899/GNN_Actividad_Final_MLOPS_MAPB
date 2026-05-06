"""
Tests de la API REST con FastAPI TestClient.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "checkpoint_final.pt",
)
MODEL_EXISTS = os.path.isfile(MODEL_PATH)
os.environ["MODEL_PATH"] = MODEL_PATH

from fastapi.testclient import TestClient
from src.inference_api import app


@pytest.fixture(scope="module")
def client():
    """Cliente que activa el lifespan (carga el modelo al arrancar)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_endpoint(client):
    """El endpoint /health debe devolver status 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info_endpoint(client):
    """El endpoint /model/info debe devolver metadatos."""
    resp = client.get("/model/info")
    if MODEL_EXISTS:
        assert resp.status_code == 200
        data = resp.json()
        assert "architecture" in data
        assert "hidden_channels" in data
    else:
        assert resp.status_code in (200, 503)


def test_predict_valid_input(client):
    """Predicción con entrada válida debe devolver 200 y lipophilicity float."""
    if not MODEL_EXISTS:
        pytest.skip("Modelo no disponible para test de predicción.")

    payload = {
        "nodes": [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 6,
        "edges": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 4], [4, 5]],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "lipophilicity" in data
    assert isinstance(data["lipophilicity"], float)
    assert "inference_time_ms" in data


def test_predict_wrong_feature_count(client):
    """Entrada con número incorrecto de features debe devolver 422."""
    payload = {
        "nodes": [[0.0, 1.0, 0.0]],  # 3 features en lugar de 9
        "edges": [],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_edge(client):
    """Arista con índice fuera de rango debe devolver 422."""
    payload = {
        "nodes": [[0.0] * 9, [0.0] * 9],
        "edges": [[0, 99]],  # nodo 99 no existe
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_nodes(client):
    """Lista de nodos vacía debe devolver 422."""
    payload = {"nodes": [], "edges": []}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_no_edges(client):
    """Molécula con nodos pero sin aristas debe responder (grafo desconectado)."""
    if not MODEL_EXISTS:
        pytest.skip("Modelo no disponible.")
    payload = {
        "nodes": [[0.0] * 9] * 3,
        "edges": [],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
