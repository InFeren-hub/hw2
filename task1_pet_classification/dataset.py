from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PetClassificationDataset(Dataset):
    """Oxford-IIIT Pet 分类数据集"""

    def __init__(self, data_dir="data", split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        list_name = "trainval.txt" if split == "train" else "test.txt"
        list_file = self.data_dir / "annotations" / list_name
        if not list_file.exists():
            raise FileNotFoundError(f"Missing dataset list: {list_file}")

        self.samples = []
        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0] + ".jpg"
                    label = int(parts[1]) - 1 
                    self.samples.append((image_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, label = self.samples[index]
        image = Image.open(self.data_dir / "images" / image_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
