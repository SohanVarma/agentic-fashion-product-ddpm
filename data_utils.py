from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_DATA_DIR = Path("data/fashion_products")


def load_styles(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csv_path = data_dir / "styles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run python download_dataset.py or manually place the dataset in data/fashion_products/."
        )
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    if "id" not in df.columns:
        raise ValueError("styles.csv must contain an 'id' column")
    return df


def build_class_list(df: pd.DataFrame, label_col: str = "masterCategory", top_k: int = 8) -> List[str]:
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col!r} not found in styles.csv. Available: {list(df.columns)}")
    values = df[label_col].dropna().astype(str)
    counts = values.value_counts()
    return list(counts.head(top_k).index)


class FashionProductsDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        image_size: int = 64,
        label_col: str = "masterCategory",
        class_names: Optional[List[str]] = None,
        max_images: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.df = load_styles(self.data_dir)
        self.label_col = label_col
        if class_names is None:
            class_names = build_class_list(self.df, label_col=label_col, top_k=8)
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        rows = []
        for _, row in self.df.iterrows():
            label = str(row.get(label_col, ""))
            if label not in self.class_to_idx:
                continue
            img_path = self.image_dir / f"{row['id']}.jpg"
            if img_path.exists():
                rows.append((img_path, self.class_to_idx[label], label))
            if max_images and len(rows) >= max_images:
                break

        if not rows:
            raise RuntimeError(
                "No usable images found. Check data/fashion_products/styles.csv and data/fashion_products/images/."
            )
        self.samples: List[Tuple[Path, int, str]] = rows
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.preview_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label_idx, dtype=torch.long)

    def get_real_samples(self, label_idx: int, count: int = 8):
        imgs = []
        for path, y, _ in self.samples:
            if y == label_idx:
                img = Image.open(path).convert("RGB")
                imgs.append(self.preview_transform(img))
                if len(imgs) >= count:
                    break
        return torch.stack(imgs) if imgs else torch.empty(0)
