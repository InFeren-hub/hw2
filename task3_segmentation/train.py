import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from task3_segmentation.dataset import PetSegDataset
from task3_segmentation.models import UNet, build_loss


def update_iou(stats, logits, masks, num_classes):
    preds = logits.argmax(1)
    for cls in range(num_classes):
        pred = preds == cls
        target = masks == cls
        stats["inter"][cls] += torch.logical_and(pred, target).sum().item()
        stats["union"][cls] += torch.logical_or(pred, target).sum().item()


def compute_miou(stats):
    ious = [i / u for i, u in zip(stats["inter"], stats["union"]) if u > 0]
    return sum(ious) / max(len(ious), 1)


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, total = 0.0, 0
    stats = {"inter": [0.0] * num_classes, "union": [0.0] * num_classes}
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()

    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * images.size(0)
            total += images.size(0)
            update_iou(stats, logits.detach(), masks, num_classes)
    return loss_sum / max(total, 1), compute_miou(stats)


def train_one(args, loss_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        PetSegDataset(args.data_dir, "train", args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PetSegDataset(args.data_dir, "test", args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(num_classes=args.num_classes, base=args.base_channels).to(device)
    criterion = build_loss(loss_name, args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, best_miou = [], 0.0
    print(f"\n[{loss_name}] device={device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou = run_epoch(model, train_loader, criterion, device, args.num_classes, optimizer)
        val_loss, val_miou = run_epoch(model, val_loader, criterion, device, args.num_classes)
        rows.append([epoch, train_loss, train_miou, val_loss, val_miou])
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_miou={train_miou:.4f} val_miou={val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {"model_state_dict": model.state_dict(), "best_val_miou": best_miou, "loss": loss_name, "args": vars(args)},
                out_dir / f"task3_unet_{loss_name}_best.pth",
            )

    with (out_dir / f"task3_unet_{loss_name}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_miou", "val_loss", "val_miou"])
        writer.writerows(rows)
    return best_miou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="experiments/task3")
    parser.add_argument("--loss", default="all", choices=["all", "ce", "dice", "ce_dice"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    losses = ["ce", "dice", "ce_dice"] if args.loss == "all" else [args.loss]
    results = [(name, train_one(args, name)) for name in losses]
    print("\nSummary")
    for name, miou in results:
        print(f"{name:7s} best_val_miou={miou:.4f}")


if __name__ == "__main__":
    main()
