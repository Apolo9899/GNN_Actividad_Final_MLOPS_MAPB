"""
Funciones auxiliares para entrenamiento, evaluación y reproducibilidad.
"""
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss, total_graphs = 0.0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data).view(-1)
        y_true = data.y.view(-1)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, device: str):
    """Evalúa el modelo. Devuelve (mse, y_true, y_pred)."""
    model.eval()
    y_pred_list, y_true_list = [], []
    for data in loader:
        data = data.to(device)
        out = model(data).view(-1)
        y_pred_list.extend(out.cpu().tolist())
        y_true_list.extend(data.y.view(-1).cpu().tolist())
    mse = mean_squared_error(y_true_list, y_pred_list)
    return mse, y_true_list, y_pred_list


@torch.no_grad()
def predict(model, loader, device: str):
    """Genera predicciones sobre un dataloader (sin etiquetas)."""
    model.eval()
    model.to(device)
    nids, preds = [], []
    for data in loader:
        data = data.to(device)
        out = model(data).view(-1).cpu().tolist()
        preds.extend(out)
        nids.extend(data.nid.cpu().tolist())
    return nids, preds


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    n_epochs: int = 500,
    patience: int = 50,
    device: str = "cpu",
    verbose: bool = True,
    checkpoint_path: str = None,
    wandb_run=None,
):
    """
    Entrenamiento con early stopping y logging opcional a W&B.

    Devuelve (train_losses, val_mses, best_val_mse, best_epoch).
    """
    best_val_mse = float("inf")
    best_epoch = 0
    patience_cnt = 0
    train_losses, val_mses = [], []
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, _, _ = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_mses.append(val_mse)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mse)
            else:
                scheduler.step()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train_loss": train_loss,
                    "val_mse": val_mse,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        else:
            patience_cnt += 1

        if verbose and epoch % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} "
                f"| Best: {best_val_mse:.4f} (ep {best_epoch}) | LR: {lr:.6f}"
            )

        if patience_cnt >= patience:
            if verbose:
                print(f"  Early stopping en epoch {epoch} (mejor: {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_mses, best_val_mse, best_epoch
