"""
Tests unitarios e de integración para los modelos GNN.
"""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AdvancedGINGraph, GATGraph, GCNGraph, GINGraph, VirtualNodeGIN
from torch_geometric.data import Batch, Data


NUM_FEATURES = 9


def _make_batch(n_graphs=4, n_nodes=10, n_edges=20):
    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, NUM_FEATURES)
        src = torch.randint(0, n_nodes, (n_edges,))
        dst = torch.randint(0, n_nodes, (n_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize(
    "ModelClass,kwargs",
    [
        (GCNGraph, {"num_features": NUM_FEATURES}),
        (GINGraph, {"num_features": NUM_FEATURES}),
        (GATGraph, {"num_features": NUM_FEATURES}),
        (AdvancedGINGraph, {"num_features": NUM_FEATURES}),
        (VirtualNodeGIN, {"num_features": NUM_FEATURES}),
    ],
)
def test_output_shape(ModelClass, kwargs):
    """El modelo debe devolver un tensor de shape (batch_size, 1)."""
    model = ModelClass(**kwargs)
    model.eval()
    batch = _make_batch(n_graphs=4)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4, 1), f"Shape incorrecto: {out.shape}"


@pytest.mark.parametrize(
    "ModelClass",
    [GCNGraph, GINGraph, GATGraph, AdvancedGINGraph, VirtualNodeGIN],
)
def test_output_dtype(ModelClass):
    """La salida debe ser float."""
    model = ModelClass(num_features=NUM_FEATURES)
    model.eval()
    batch = _make_batch()
    with torch.no_grad():
        out = model(batch)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
    "ModelClass",
    [GCNGraph, GINGraph, GATGraph, AdvancedGINGraph, VirtualNodeGIN],
)
def test_gradients_not_none(ModelClass):
    """La cabeza de regresión debe tener gradientes tras backward."""
    model = ModelClass(num_features=NUM_FEATURES)
    model.train()
    batch = _make_batch()
    batch.y = torch.randn(4, 1)
    out = model(batch)
    loss = torch.nn.MSELoss()(out.view(-1), batch.y.view(-1))
    loss.backward()
    # Verificar que la cabeza final de regresión tiene gradientes
    # (excluimos vn_mlps que pueden no recibir gradiente en el último paso)
    head_params = [(n, p) for n, p in model.named_parameters()
                   if ("mlp_head" in n or "cls" in n or
                       (("mlp" in n) and "vn_mlps" not in n and "convs" not in n))
                   and p.requires_grad]
    assert len(head_params) > 0, "No se encontraron parámetros de la cabeza"
    for name, p in head_params:
        assert p.grad is not None, f"Gradiente None en {name}"


def test_advanced_gin_overfit_small_batch():
    """El modelo debe poder memorizar un lote muy pequeño (test de sanidad)."""
    torch.manual_seed(0)
    model = AdvancedGINGraph(num_features=NUM_FEATURES, hidden_channels=64, n_layers=3)
    batch = _make_batch(n_graphs=4, n_nodes=8, n_edges=10)
    batch.y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(batch).view(-1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(batch).view(-1), batch.y).item()
    assert final_loss < 0.5, f"El modelo no puede memorizar el lote pequeño (loss={final_loss:.4f})"


def test_single_graph_inference():
    """La API debe poder hacer inferencia con un único grafo (batch_size=1)."""
    model = AdvancedGINGraph(num_features=NUM_FEATURES)
    model.eval()
    x = torch.randn(5, NUM_FEATURES)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    with torch.no_grad():
        out = model(data)
    assert out.shape == (1, 1)
