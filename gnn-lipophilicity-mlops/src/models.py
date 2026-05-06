"""
Arquitecturas GNN para predicción de lipofilicidad molecular.
"""
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, Module, ReLU, Sequential
from torch_geometric.nn import (
    GATConv,
    GCN,
    GINConv,
    MLP,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GCNGraph(Module):
    """GCN con global_add_pool y MLP clasificador."""

    def __init__(self, num_features, num_classes=1, hidden_channels=64, n_layers=3):
        super().__init__()
        self.gcn = GCN(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            num_layers=n_layers,
            out_channels=hidden_channels,
            dropout=0.5,
        )
        self.cls = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_layers=2,
            dropout=0.5,
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcn(x.float(), edge_index)
        x = global_add_pool(x, batch)
        return self.cls(x)


class GINGraph(Module):
    """GIN con BatchNorm y LR scheduling."""

    def __init__(self, num_features, hidden_channels=128, n_layers=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        mlp_in = Sequential(
            Linear(num_features, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp_in, train_eps=True))
        self.bns.append(BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 1):
            mlp_h = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp_h, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            BatchNorm1d(hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class GATGraph(Module):
    """Graph Attention Network con BatchNorm."""

    def __init__(
        self, num_features, hidden_channels=128, n_layers=3, heads=4, dropout=0.3
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(
            GATConv(
                num_features, hidden_channels // heads, heads=heads, dropout=dropout, concat=True
            )
        )
        self.bns.append(BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.bns.append(BatchNorm1d(hidden_channels))

        self.convs.append(
            GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout, concat=False)
        )
        self.bns.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            BatchNorm1d(hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.mlp(x)


class AdvancedGINGraph(Module):
    """GIN con Jumping Knowledge + Multi-scale Pooling + Residuales."""

    def __init__(self, num_features, hidden_channels=256, n_layers=5, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers

        self.input_proj = Sequential(
            Linear(num_features, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
        )

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for _ in range(n_layers):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels * 2),
                BatchNorm1d(hidden_channels * 2),
                ReLU(),
                Linear(hidden_channels * 2, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))

        jk_dim = n_layers * hidden_channels * 3
        self.mlp_head = Sequential(
            Linear(jk_dim, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            BatchNorm1d(hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.input_proj(x)
        layer_reps = []

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new + x
            rep = torch.cat(
                [global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)],
                dim=1,
            )
            layer_reps.append(rep)

        out = torch.cat(layer_reps, dim=1)
        return self.mlp_head(out)


class VirtualNodeGIN(Module):
    """GIN con Nodo Virtual + JK + Multi-scale Pooling + Residuales."""

    def __init__(self, num_features, hidden_channels=256, n_layers=5, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers

        self.input_proj = Sequential(
            Linear(num_features, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
        )

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.vn_mlps = torch.nn.ModuleList()

        for _ in range(n_layers):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels * 2),
                BatchNorm1d(hidden_channels * 2),
                ReLU(),
                Linear(hidden_channels * 2, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))
            self.vn_mlps.append(
                Sequential(
                    Linear(hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    ReLU(),
                )
            )

        jk_dim = n_layers * hidden_channels * 3
        self.mlp_head = Sequential(
            Linear(jk_dim, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            BatchNorm1d(hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        n_graphs = int(batch.max().item()) + 1
        x = self.input_proj(x)
        vn_emb = torch.zeros(n_graphs, x.size(1), device=x.device)
        layer_reps = []

        for conv, bn, vn_mlp in zip(self.convs, self.bns, self.vn_mlps):
            x = x + vn_emb[batch]
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new + x
            vn_emb = vn_emb + vn_mlp(global_add_pool(x, batch))
            vn_emb = F.dropout(vn_emb, p=self.dropout, training=self.training)
            layer_reps.append(
                torch.cat(
                    [
                        global_add_pool(x, batch),
                        global_mean_pool(x, batch),
                        global_max_pool(x, batch),
                    ],
                    dim=1,
                )
            )

        out = torch.cat(layer_reps, dim=1)
        return self.mlp_head(out)


MODEL_REGISTRY = {
    "gcn": GCNGraph,
    "gin": GINGraph,
    "gat": GATGraph,
    "gin_jk": AdvancedGINGraph,
    "gin_vn": VirtualNodeGIN,
}
