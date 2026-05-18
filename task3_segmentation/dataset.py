import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from pathlib import Path


class PetSegDataset(Dataset):
    """trimap=0宠物 1边缘 2背景"""

    def __init__(self, data_dir="data", split="train", image_size=256):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        list_name = "trainval.txt" if split == "train" else "test.txt"
        with (self.data_dir / "annotations" / list_name).open("r", encoding="utf-8") as f:
            self.ids = [line.split()[0] for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = Image.open(self.data_dir / "images" / f"{name}.jpg").convert("RGB")
        mask = Image.open(self.data_dir / "annotations" / "trimaps" / f"{name}.png")

        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.Resampling.NEAREST)
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, (mask - 1).clamp(0, 2)
