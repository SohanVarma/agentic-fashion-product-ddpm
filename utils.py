from pathlib import Path
import csv
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images.detach().cpu(), nrow=nrow)
    save_image(grid, path)
    return path


def save_loss_csv(losses, path="outputs/training_loss.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for i, loss in enumerate(losses):
            writer.writerow([i, loss])


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


def plot_loss(loss_csv="outputs/training_loss.csv"):
    import pandas as pd
    path = Path(loss_csv)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots()
    ax.plot(df["step"], df["loss"])
    ax.set_title("Diffusion Training Loss")
    ax.set_xlabel("Logged Step")
    ax.set_ylabel("MSE Loss")
    return fig
