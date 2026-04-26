import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_utils import FashionProductsDataset
from classifier import FashionClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=12000)
    parser.add_argument("--label_col", type=str, default="masterCategory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = FashionProductsDataset(image_size=args.image_size, label_col=args.label_col, max_images=args.max_images)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = FashionClassifier(num_classes=len(dataset.class_names)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Classifier epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}, val_acc={acc:.3f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "class_names": dataset.class_names,
        "image_size": args.image_size,
        "label_col": args.label_col,
    }, "checkpoints/fashion_product_classifier.pt")
    print("Saved checkpoints/fashion_product_classifier.pt")


if __name__ == "__main__":
    main()
