"""Milestone 1 smoke-test model.

This is NOT the full DDPM. It is a small conditional denoising model to prove
that the training loop, logging, and reproducibility work correctly.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = TinyDenoiser()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X = torch.randn(256, 64)
    noise = torch.randn_like(X) * 0.1
    noisy_X = X + noise

    losses = []

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(noisy_X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    out_file = RESULTS_DIR / "milestone1_metrics.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, l in enumerate(losses, 1):
            writer.writerow([i, l])

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
