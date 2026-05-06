"""
Registra en W&B los resultados ya calculados en el notebook del torneo GNN.

Uso:
    python src/log_experiments_wandb.py --project gnn-lipophilicity
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = [
    {
        "name": "Exp1_GCN_Baseline",
        "config": {
            "model": "GCNGraph",
            "hidden_channels": 64,
            "n_layers": 3,
            "dropout": 0.5,
            "lr": 0.001,
            "batch_size": 512,
            "loss": "HuberLoss",
            "scheduler": "none",
            "n_params": 13313,
        },
        "best_val_mse": 0.7505,
        "best_epoch": 498,
        # Curva de validación MSE por época (cada 50 épocas)
        "val_mse_curve": [
            (50, 1.4715), (100, 1.3973), (150, 1.3543), (200, 1.1888),
            (250, 0.9867), (300, 0.9309), (350, 0.8143), (400, 0.7975),
            (450, 0.7737), (500, 0.7505),
        ],
        "train_loss_curve": [
            (50, 0.6403), (100, 0.6087), (150, 0.5769), (200, 0.5383),
            (250, 0.4890), (300, 0.4420), (350, 0.4329), (400, 0.4115),
            (450, 0.3964), (500, 0.3863),
        ],
    },
    {
        "name": "Exp2_GIN_BatchNorm",
        "config": {
            "model": "GINGraph",
            "hidden_channels": 128,
            "n_layers": 4,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 256,
            "loss": "MSELoss",
            "scheduler": "ReduceLROnPlateau",
            "n_params": 127365,
        },
        "best_val_mse": 0.4162,
        "best_epoch": 283,
        "val_mse_curve": [
            (50, 0.6602), (100, 0.4976), (150, 0.4643), (200, 0.4267),
            (250, 0.4288), (300, 0.4267), (350, 0.4269), (358, 0.4162),
        ],
        "train_loss_curve": [
            (50, 0.6815), (100, 0.4641), (150, 0.4042), (200, 0.3372),
            (250, 0.3436), (300, 0.3027), (350, 0.3191), (358, 0.3191),
        ],
    },
    {
        "name": "Exp3_GAT",
        "config": {
            "model": "GATGraph",
            "hidden_channels": 128,
            "n_layers": 3,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 256,
            "loss": "MSELoss",
            "scheduler": "ReduceLROnPlateau",
            "n_params": 44289,
        },
        "best_val_mse": 0.6392,
        "best_epoch": 341,
        "val_mse_curve": [
            (50, 1.1602), (100, 0.9651), (150, 0.8299), (200, 0.6996),
            (250, 0.6599), (300, 0.6639), (350, 0.6579), (400, 0.6449),
            (416, 0.6392),
        ],
        "train_loss_curve": [
            (50, 1.3880), (100, 1.1737), (150, 1.0386), (200, 0.9279),
            (250, 0.8831), (300, 0.8849), (350, 0.8980), (400, 0.8573),
            (416, 0.8573),
        ],
    },
    {
        "name": "Exp4_GIN_Avanzado_JK",
        "config": {
            "model": "AdvancedGINGraph",
            "hidden_channels": 256,
            "n_layers": 5,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 128,
            "loss": "MSELoss",
            "scheduler": "CosineAnnealingLR",
            "jk_connections": True,
            "multiscale_pooling": True,
            "residual_connections": True,
            "n_params": 2673926,
        },
        "best_val_mse": 0.3030,
        "best_epoch": 263,
        "val_mse_curve": [],  # curva no disponible en detalle
        "train_loss_curve": [],
    },
    {
        "name": "Exp5_KFold_Ensemble",
        "config": {
            "model": "AdvancedGINGraph_KFold",
            "hidden_channels": 256,
            "n_layers": 5,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 128,
            "loss": "MSELoss",
            "scheduler": "CosineAnnealingLR",
            "k_folds": 5,
            "ensemble": True,
            "n_params": 2673926,
        },
        "best_val_mse": 0.3055,
        "best_val_mse_std": 0.0108,
        "best_epoch": 0,
        "val_mse_curve": [],
        "train_loss_curve": [],
    },
    {
        "name": "Exp6_VirtualNodeGIN",
        "config": {
            "model": "VirtualNodeGIN",
            "hidden_channels": 256,
            "n_layers": 5,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 128,
            "loss": "MSELoss",
            "scheduler": "CosineAnnealingLR",
            "virtual_node": True,
            "jk_connections": True,
            "multiscale_pooling": True,
            "n_params": 2673926,
        },
        "best_val_mse": 0.3412,
        "best_epoch": 0,
        "val_mse_curve": [],
        "train_loss_curve": [],
    },
    {
        "name": "Exp8_GraphMAE_FineTune",
        "config": {
            "model": "GraphMAE_FineTune",
            "hidden_channels": 256,
            "n_layers": 5,
            "dropout": 0.2,
            "pretrain_epochs": 300,
            "mask_rate": 0.30,
            "finetune_lr_encoder": 5e-5,
            "finetune_lr_head": 5e-4,
            "pretraining": "GraphMAE",
            "n_params": 2673926,
        },
        "best_val_mse": 0.3377,
        "best_epoch": 0,
        "val_mse_curve": [],
        "train_loss_curve": [],
    },
]


def log_all_experiments(project: str, entity: str = None):
    import wandb

    print(f"Registrando {len(EXPERIMENTS)} experimentos en W&B proyecto: {project}\n")

    for exp in EXPERIMENTS:
        run = wandb.init(
            project=project,
            entity=entity,
            name=exp["name"],
            config=exp["config"],
            tags=["gnn-lipophilicity", "torneo-2025-26"],
        )

        # Loggear curvas si están disponibles
        for epoch, val_mse in exp.get("val_mse_curve", []):
            train_loss = next(
                (l for e, l in exp.get("train_loss_curve", []) if e == epoch), None
            )
            log_dict = {"val_mse": val_mse, "epoch": epoch}
            if train_loss is not None:
                log_dict["train_loss"] = train_loss
            wandb.log(log_dict, step=epoch)

        # Resumen final
        summary = {"best_val_mse": exp["best_val_mse"]}
        if "best_val_mse_std" in exp:
            summary["best_val_mse_std"] = exp["best_val_mse_std"]
        if exp.get("best_epoch"):
            summary["best_epoch"] = exp["best_epoch"]

        for k, v in summary.items():
            wandb.run.summary[k] = v

        print(f"  ✓ {exp['name']} — MSE: {exp['best_val_mse']:.4f}")
        wandb.finish()

    print("\nTodos los experimentos registrados.")
    print(f"Ve a https://wandb.ai/{entity or 'TU_USUARIO'}/{project}/table para ver la tabla comparativa.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="gnn-lipophilicity")
    parser.add_argument("--entity", type=str, default=None, help="Tu usuario de W&B")
    args = parser.parse_args()
    log_all_experiments(args.project, args.entity)
