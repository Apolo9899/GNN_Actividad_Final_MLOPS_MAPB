"""
API REST con FastAPI para predicción de lipofilicidad molecular.

Lanzar localmente:
    uvicorn src.inference_api:app --reload --port 8000

Endpoints:
    GET  /health           -> estado del servicio
    POST /predict          -> predicción desde features/edges en JSON
    GET  /model/info       -> metadatos del modelo cargado
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VirtualNodeGIN
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(ROOT_DIR, "models", "checkpoint_final.pt")
)

NUM_FEATURES = 9
HIDDEN_CHANNELS = 256
N_LAYERS = 5
DROPOUT = 0.2

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="GNN Lipophilicity API",
    description=(
        "Predice la lipofilicidad de moléculas representadas como grafos. "
        "Proyecto MLOps — Máster Deep Learning UPM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_model = None
_device = get_device()
_model_load_time = None


def load_model():
    global _model, _model_load_time
    if not os.path.isfile(MODEL_PATH):
        logger.error(f"No se encontró el modelo en: {MODEL_PATH}")
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")

    _model = VirtualNodeGIN(
        num_features=NUM_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(_device)
    state = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    _model.load_state_dict(state)
    _model.eval()
    _model_load_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Modelo cargado desde {MODEL_PATH} en {_device}")



class MoleculeInput(BaseModel):
    """
    Representación de una molécula como grafo.

    - nodes: lista de vectores de características por nodo (9 valores cada uno).
    - edges: lista de pares [origen, destino] para cada arista.
    """

    nodes: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Vectores de características de nodos (cada uno con 9 valores).",
        json_schema_extra={"example": [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 5},
    )
    edges: List[List[int]] = Field(
        ...,
        description="Lista de aristas como pares [src, dst].",
        json_schema_extra={"example": [[0, 1], [1, 0], [1, 2], [2, 1]]},
    )


class PredictionResponse(BaseModel):
    lipophilicity: float
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    architecture: str
    hidden_channels: int
    n_layers: int
    num_features: int
    checkpoint: str
    loaded_at: str


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=_device,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Sistema"])
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return ModelInfoResponse(
        architecture="VirtualNodeGIN (JK + Multi-scale Pooling + Nodo Virtual)",
        hidden_channels=HIDDEN_CHANNELS,
        n_layers=N_LAYERS,
        num_features=NUM_FEATURES,
        checkpoint=os.path.basename(MODEL_PATH),
        loaded_at=_model_load_time or "desconocido",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(molecule: MoleculeInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    for i, node in enumerate(molecule.nodes):
        if len(node) != NUM_FEATURES:
            raise HTTPException(
                status_code=422,
                detail=f"Nodo {i} tiene {len(node)} features; se esperan {NUM_FEATURES}.",
            )

    n_nodes = len(molecule.nodes)
    for src, dst in molecule.edges:
        if src < 0 or dst < 0 or src >= n_nodes or dst >= n_nodes:
            raise HTTPException(
                status_code=422,
                detail=f"Arista inválida [{src}, {dst}] para {n_nodes} nodos.",
            )

    try:
        t0 = time.perf_counter()
        x = torch.tensor(molecule.nodes, dtype=torch.float)
        if molecule.edges:
            edge_index = torch.tensor(molecule.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, batch=batch).to(_device)

        with torch.no_grad():
            pred = _model(data).item()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Predicción: {pred:.4f} | {elapsed_ms:.1f} ms | nodos={n_nodes}")
        return PredictionResponse(lipophilicity=pred, inference_time_ms=round(elapsed_ms, 2))

    except Exception as exc:
        logger.error(f"Error en predicción: {exc}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(exc)}")
