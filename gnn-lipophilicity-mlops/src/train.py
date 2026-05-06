"""
Script de entrenamiento con integración de Weights & Biases.

Uso:
    python src/train.py --model gin_jk --epochs 600 --lr 0.001 --wandb
"""
import argparse
import os
import sys

import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import MoleculeDataset
from src.models import MODEL_REGISTRY, AdvancedGINGraph
from src.utils import evaluate, get_device, set_seed, train_model


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "nx_graphs")
CACHE_DIR = os.path.join(ROOT_DIR, "data", "processed_cache")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento GNN lipofilicidad")
    parser.add_argument("--model", type=str, default="gin_jk", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--wandb", action="store_true", help="Activar W&B logging")
    parser.add_argument("--wandb_project", type=str, default="gnn-lipophilicity")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Dispositivo: {device}")

    dataset = MoleculeDataset(root=CACHE_DIR, gml_dir=DATA_DIR)
    print(
        f"Dataset: {len(dataset)} moléculas | "
        f"Train: {len(dataset.train_idx)} | Test: {len(dataset.test_idx)}"
    )

    train_idx, val_idx = train_test_split(
        dataset.train_idx, test_size=args.val_split, random_state=args.seed
    )
    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=512, shuffle=False)

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(
        num_features=dataset.num_features,
        hidden_channels=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {args.model} | Parámetros: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300, eta_min=1e-5
    )
    criterion = torch.nn.MSELoss()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config={
                "model": args.model,
                "hidden_channels": args.hidden,
                "n_layers": args.layers,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "n_params": n_params,
                "device": device,
            },
        )

    checkpoint_path = args.checkpoint or os.path.join(
        MODELS_DIR, f"checkpoint_{args.model}.pt"
    )
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_losses, val_mses, best_mse, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=scheduler,
        n_epochs=args.epochs,
        patience=args.patience,
        device=device,
        verbose=True,
        checkpoint_path=checkpoint_path,
        wandb_run=wandb_run,
    )

    _, y_true, y_pred = evaluate(model, val_loader, device)
    print(f"\nMejor MSE de validación: {best_mse:.4f} (epoch {best_epoch})")
    print(f"Checkpoint guardado en: {checkpoint_path}")

    if wandb_run is not None:
        wandb_run.summary["best_val_mse"] = best_mse
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()

    return best_mse


if __name__ == "__main__":
    main()
