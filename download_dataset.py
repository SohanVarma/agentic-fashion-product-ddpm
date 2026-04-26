"""Download Kaggle Fashion Product Images Small dataset.

Option 1:
    python download_dataset.py

If Kaggle download fails, manually download from:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
Then place styles.csv and images/ inside data/fashion_products/
"""
from pathlib import Path
import shutil
import sys

TARGET = Path("data/fashion_products")


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        print("Downloading dataset using kagglehub...")
        path = Path(kagglehub.dataset_download("paramaggarwal/fashion-product-images-small"))
        print(f"Downloaded to: {path}")
    except Exception as exc:
        print("Could not download with kagglehub.")
        print("Reason:", exc)
        print("Manual fallback:")
        print("1. Open https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
        print("2. Download and unzip it")
        print("3. Put styles.csv and images/ inside data/fashion_products/")
        sys.exit(1)

    # KaggleHub may return a folder containing styles.csv/images or a nested folder.
    candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
    source = None
    for c in candidates:
        if (c / "styles.csv").exists() and (c / "images").exists():
            source = c
            break

    if source is None:
        print("Downloaded folder did not contain styles.csv and images/ in an expected layout.")
        print("Please manually copy them to data/fashion_products/")
        sys.exit(1)

    print(f"Preparing dataset from: {source}")
    shutil.copy2(source / "styles.csv", TARGET / "styles.csv")

    dst_images = TARGET / "images"
    if dst_images.exists():
        shutil.rmtree(dst_images)
    shutil.copytree(source / "images", dst_images)

    print("Dataset ready at data/fashion_products/")
    print("Expected files:")
    print("- data/fashion_products/styles.csv")
    print("- data/fashion_products/images/*.jpg")


if __name__ == "__main__":
    main()
