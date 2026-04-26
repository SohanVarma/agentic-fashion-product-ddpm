import argparse
import torch
from model import ConditionalUNet
from diffusion import Diffusion
from utils import save_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/conditional_fashion_product_ddpm.pt")
    parser.add_argument("--class_name", default="Apparel")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    class_names = ckpt["class_names"]
    if args.class_name not in class_names:
        raise ValueError(f"Unknown class {args.class_name}. Choices: {class_names}")
    image_size = ckpt.get("image_size", 64)
    timesteps = ckpt.get("timesteps", 100)
    base_channels = ckpt.get("base_channels", 64)
    model = ConditionalUNet(num_classes=len(class_names), base_channels=base_channels).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    diffusion = Diffusion(timesteps=timesteps, device=args.device)
    label_idx = class_names.index(args.class_name)
    labels = torch.full((args.num_images,), label_idx, dtype=torch.long, device=args.device)
    images = diffusion.sample(model, args.num_images, labels, image_size=image_size, guidance_scale=args.guidance_scale)
    out = save_grid(images, f"outputs/generated_{args.class_name}.png", nrow=4)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
