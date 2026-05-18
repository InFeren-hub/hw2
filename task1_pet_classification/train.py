import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from task1_pet_classification.dataset import PetClassificationDataset
from task1_pet_classification.models import build_model, parameter_groups

EXPERIMENTS = {
    "scratch":    dict(pretrained=False, use_attention=False, lr_backbone=1e-3, lr_head=1e-3),
    "pretrained": dict(pretrained=True,  use_attention=False, lr_backbone=1e-5, lr_head=1e-3),
    "se":         dict(pretrained=True,  use_attention=True,  lr_backbone=1e-5, lr_head=1e-3),
    "lower_lr":   dict(pretrained=True,  use_attention=False, lr_backbone=5e-6, lr_head=5e-4),
    "higher_lr":  dict(pretrained=True,  use_attention=False, lr_backbone=5e-5, lr_head=5e-3),
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(data_dir: str, batch_size: int, seed: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = PetClassificationDataset(data_dir=data_dir, split="train", transform=train_tf)
    test_set = PetClassificationDataset(data_dir=data_dir, split="test", transform=eval_tf)
    # 固定数据加载顺序
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def train_one(args, exp_name: str, config: dict):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.seed, args.num_workers)

    model = build_model(
        num_classes=args.num_classes,
        pretrained=config["pretrained"],
        use_attention=config["use_attention"],
    ).to(device)

    optimizer = optim.Adam(
        parameter_groups(model, config["lr_backbone"], config["lr_head"]),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  
    criterion = nn.CrossEntropyLoss()

    best_acc, best_epoch, best_state = 0.0, 0, None  
    no_improve, rows = 0, []
    print(f"\n[{exp_name}] device={device}, pretrained={config['pretrained']}, attention={config['use_attention']}")
    print(f"lr_backbone={config['lr_backbone']}, lr_head={config['lr_head']}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)
        test_acc = evaluate(model, test_loader, device)
        rows.append([epoch, train_loss, train_acc, test_acc])
        print(f"epoch {epoch:02d}/{args.epochs} loss={train_loss:.4f} train_acc={train_acc:.2f} test_acc={test_acc:.2f}")

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            best_state = copy.deepcopy(model.state_dict())  
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

   
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "config": {**config, **vars(args), "exp_name": exp_name},
        },
        out_dir / f"{exp_name}.pth",
    )
    with (out_dir / f"{exp_name}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc"])
        writer.writerows(rows)
    return best_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--experiment", default="all", choices=["all", *EXPERIMENTS.keys()])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=37)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)  # 早停耐心值
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    results = [(name, train_one(args, name, EXPERIMENTS[name])) for name in names]
    print("\nSummary")
    for name, acc in results:
        print(f"{name:12s} best_test_acc={acc:.2f}%")


if __name__ == "__main__":
    main()
