import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS = [
    ("Exp1_Baseline_Scratch", "Scratch"),
    ("Exp2_Baseline_Pretrained", "ImageNet Pretrained"),
    ("Exp3_Attention_SEBlock", "Pretrained + SE"),
    ("Exp4_Hparam_LowerLR", "Lower LR"),
    ("Exp5_Hparam_HigherLR", "Higher LR"),
]


def load_csv(path: Path):
    """加载训练的 metrics CSV 文件"""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "test_acc": float(row["test_acc"]),
            }
            for row in csv.DictReader(f)
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="experiments")
    parser.add_argument("--out-dir", default="report/figures")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fields = [
        ("train_loss", "Train Loss"),
        ("train_acc", "Train Accuracy (%)"),
        ("test_acc", "Test Accuracy (%)"),
    ]
    for ax, (field, ylabel) in zip(axes, fields):
        for exp_name, label in EXPERIMENTS:
            rows = load_csv(metrics_dir / f"{exp_name}_metrics.csv")
            if not rows:
                continue
            ax.plot([r["epoch"] for r in rows], [r[field] for r in rows],
                    marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "task1_training_curves.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "task1_training_curves.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
