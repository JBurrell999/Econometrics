from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOCAL_CACHE_DIR = Path(".cache")
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((LOCAL_CACHE_DIR / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR.resolve()))


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False
    return get_ipython() is not None


try:
    import matplotlib

    if os.environ.get("MPLBACKEND") is None and not _running_in_ipython():
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image, ImageFile
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    raise SystemExit(
        "Missing dependency: "
        f"{missing}. Install the PHYS 303 HW4 stack first, for example:\n"
        "python3 -m pip install torch torchvision matplotlib numpy pillow scikit-learn"
    ) from exc

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cat-vs-dog classifier for PHYS 303 HW4."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/jjburrell/Downloads/kagglecatsanddogs_5340/PetImages"),
        help="Path containing Cat/ and Dog/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phys303_hw4_outputs"),
        help="Directory for model checkpoints, plots, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Device for training. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional cap for faster experiments.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise ValueError("CUDA was requested but is not available in this environment.")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("MPS was requested but is not available in this environment.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class PetDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool) -> None:
        self.samples = samples
        transform_steps: list[object] = [transforms.Resize((image_size, image_size))]
        if train:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                ]
            )
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = torch.tensor(sample.label, dtype=torch.float32)
        return tensor, label


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs).squeeze(1)


def discover_samples(
    data_root: Path, max_samples_per_class: int | None
) -> tuple[list[Sample], list[dict[str, str]], dict[str, int]]:
    samples: list[Sample] = []
    removed: list[dict[str, str]] = []
    mode_counts: dict[str, int] = {}
    label_map = {"Cat": 0, "Dog": 1}

    for class_name, label in label_map.items():
        class_dir = data_root / class_name
        class_files = sorted(class_dir.glob("*.jpg"))
        if max_samples_per_class is not None:
            class_files = class_files[:max_samples_per_class]
        for path in class_files:
            try:
                if path.stat().st_size == 0:
                    removed.append(
                        {"path": str(path), "label": class_name, "reason": "zero_bytes"}
                    )
                    continue
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    mode = image.mode
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                samples.append(Sample(path=path, label=label))
            except Exception as exc:  # pragma: no cover - defensive data cleaning
                removed.append(
                    {
                        "path": str(path),
                        "label": class_name,
                        "reason": type(exc).__name__,
                    }
                )
    return samples, removed, mode_counts


def split_samples(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    rng = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        split_idx = int(0.8 * len(label_samples))
        train_samples.extend(label_samples[:split_idx])
        val_samples.extend(label_samples[split_idx:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def make_loader(
    samples: list[Sample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PetDataset(samples=samples, image_size=image_size, train=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | list[float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels, dtype=np.float32)
    probs_np = np.asarray(all_probs, dtype=np.float32)
    preds_np = (probs_np >= 0.5).astype(np.int32)

    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "roc_auc": float(roc_auc_score(labels_np, probs_np)),
        "average_precision": float(average_precision_score(labels_np, probs_np)),
        "labels": labels_np.tolist(),
        "probs": probs_np.tolist(),
    }
    return metrics


def plot_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_epoch.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_accuracy"], marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_epoch.png", dpi=200)
    plt.close()


def plot_roc_and_pr(labels: Iterable[float], probs: Iterable[float], output_dir: Path) -> None:
    labels_np = np.asarray(list(labels), dtype=np.float32)
    probs_np = np.asarray(list(probs), dtype=np.float32)

    fpr, tpr, _ = roc_curve(labels_np, probs_np)
    precision, recall, _ = precision_recall_curve(labels_np, probs_np)
    auc = roc_auc_score(labels_np, probs_np)
    ap = average_precision_score(labels_np, probs_np)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color="tab:red")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()


def save_removed_files(removed: list[dict[str, str]], output_dir: Path) -> None:
    with (output_dir / "removed_images.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "reason"])
        writer.writeheader()
        writer.writerows(removed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    samples, removed, mode_counts = discover_samples(
        data_root=args.data_root, max_samples_per_class=args.max_samples_per_class
    )
    train_samples, val_samples = split_samples(samples, seed=args.seed)
    save_removed_files(removed, args.output_dir)

    train_loader = make_loader(
        train_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(
        {
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "total_valid_images": len(samples),
            "removed_images": len(removed),
            "train_size": len(train_samples),
            "validation_size": len(val_samples),
        }
    )

    model = SmallCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "val_average_precision": [],
    }

    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_roc_auc"].append(float(val_metrics["roc_auc"]))
        history["val_average_precision"].append(float(val_metrics["average_precision"]))

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} | "
            f"val_ap={val_metrics['average_precision']:.4f}"
        )

        if float(val_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    final_metrics = evaluate(
        model=model, loader=val_loader, criterion=criterion, device=device
    )
    torch.save(best_state, args.output_dir / "best_model.pt")
    plot_curves(history=history, output_dir=args.output_dir)
    plot_roc_and_pr(
        labels=final_metrics["labels"], probs=final_metrics["probs"], output_dir=args.output_dir
    )

    metrics_payload = {
        "config": {
            "data_root": str(args.data_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "max_samples_per_class": args.max_samples_per_class,
        },
        "dataset": {
            "total_valid_images": len(samples),
            "removed_images": len(removed),
            "train_size": len(train_samples),
            "validation_size": len(val_samples),
            "mode_counts_before_rgb_conversion": mode_counts,
        },
        "history": history,
        "final_validation_metrics": {
            "loss": float(final_metrics["loss"]),
            "accuracy": float(final_metrics["accuracy"]),
            "roc_auc": float(final_metrics["roc_auc"]),
            "average_precision": float(final_metrics["average_precision"]),
        },
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print("Saved outputs to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
