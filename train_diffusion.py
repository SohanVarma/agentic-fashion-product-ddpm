import argparse
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import FashionProductsDataset
from model import ConditionalUNet
from diffusion import Diffusion
from utils import save_loss_csv, save_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--max_images", type=int, default=12000)
    parser.add_argument("--label_col", type=str, default="masterCategory")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = FashionProductsDataset(image_size=args.image_size, label_col=args.label_col, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = ConditionalUNet(num_classes=len(dataset.class_names), base_channels=args.base_channels).to(args.device)
    diffusion = Diffusion(timesteps=args.timesteps, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    losses = []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Diffusion epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            # Classifier-free guidance training: randomly drop labels to NULL.
            drop_mask = torch.rand_like(y.float()) < args.cfg_dropout
            y_train = y.clone()
            y_train[drop_mask] = model.null_class

            t = diffusion.sample_timesteps(x.size(0))
            x_t, noise = diffusion.noise_images(x, t)
            pred_noise = model(x_t, t, y_train)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if global_step % 25 == 0:
                losses.append(loss.item())
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # save a quick sample after every epoch
        with torch.no_grad():
            labels = torch.arange(min(8, len(dataset.class_names)), device=args.device)
            sample = diffusion.sample(model, n=len(labels), labels=labels, image_size=args.image_size, guidance_scale=2.5)
            save_grid(sample, f"outputs/sample_epoch_{epoch+1}.png", nrow=4)

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "class_names": dataset.class_names,
        "image_size": args.image_size,
        "timesteps": args.timesteps,
        "label_col": args.label_col,
        "base_channels": args.base_channels,
    }, "checkpoints/conditional_fashion_product_ddpm.pt")
    save_loss_csv(losses, "outputs/training_loss.csv")
    print("Saved checkpoints/conditional_fashion_product_ddpm.pt")
    print("Saved outputs/training_loss.csv")


if __name__ == "__main__":
    main()
