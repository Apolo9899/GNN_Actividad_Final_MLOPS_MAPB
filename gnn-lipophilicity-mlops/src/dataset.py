"""
Dataset de moléculas para predicción de lipofilicidad.
"""
import os

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


class MoleculeDataset(InMemoryDataset):
    """
    Dataset de moléculas cargado desde archivos GML locales.

    Parámetros
    ----------
    root : str
        Directorio raíz donde se guardan los archivos procesados.
    gml_dir : str
        Ruta a la carpeta con los archivos GML.
    """

    NUM_FEATURES = 9

    def __init__(self, root, gml_dir, transform=None, pre_transform=None):
        self.gml_dir = gml_dir
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.train_idx = torch.load(self.processed_paths[1], weights_only=False)
        self.test_idx = torch.load(self.processed_paths[2], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt", "train_idx.pt", "test_idx.pt"]

    @property
    def num_features(self):
        return self.NUM_FEATURES

    def download(self):
        pass

    def process(self):
        print(f"Procesando grafos desde {self.gml_dir} ...")
        data_list, train_idx, test_idx = [], [], []

        gml_files = [f for f in os.listdir(self.gml_dir) if f.endswith(".gml")]
        for fname in gml_files:
            path = os.path.join(self.gml_dir, fname)
            G = nx.read_gml(path)
            data = from_networkx(G)
            data.x = data.x.float()
            data.nid = torch.tensor([G.graph["molecule_id"]], dtype=torch.long)

            if G.graph["set"] == "training":
                data.y = torch.tensor([G.graph["y"]], dtype=torch.float)
                train_idx.append(G.graph["molecule_id"])
            else:
                data.y = torch.tensor([float("nan")], dtype=torch.float)
                test_idx.append(G.graph["molecule_id"])

            data_list.append(data)

        data_list.sort(key=lambda d: d.nid.item())
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
        torch.save(train_idx, self.processed_paths[1])
        torch.save(test_idx, self.processed_paths[2])
        print(f"  -> {len(train_idx)} train, {len(test_idx)} test")
