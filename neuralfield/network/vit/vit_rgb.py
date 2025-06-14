"""
vit_rgb_segmentation.py – Обучение модели SegFormer (ViT-based) **только на RGB** входных данных
для многоклассовой сегментации на Agriculture-Vision или аналогичных датасетах.

📌 Отличие от agri_segformer.py:
    • Вход: только 3 канала (R, G, B)
    • Упрощённый датасет
    • Все ссылки на NDVI, NIR и 5-канальную сборку удалены

📁 Структура данных:
<split>/
  rgb/     --> изображения RGB (jpg/png)
  masks/   --> однослойные маски классов (0…N-1, 255 — игнор)

▶ Пример запуска:
python vit_rgb_segmentation.py \
  --train-dir D:/Agriculture-Vision-2021/train \
  --val-dir   D:/Agriculture-Vision-2021/val \
  --num-classes 9 \
  --checkpoint-dir runs/vit_rgb
"""

import argparse
import csv
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class RGBDataset(Dataset):
    def __init__(self, rgb_dir: Path, mask_dir: Path, size=512, augment=False):
        self.rgb_paths = sorted(rgb_dir.glob("*"))
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment
        self.tf = self.build_tf()

    def build_tf(self):
        tf = [A.Resize(self.size, self.size)]
        if self.augment:
            tf += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.3),
            ]
        tf += [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2(transpose_mask=True)]
        return A.Compose(tf)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        img_path = self.rgb_paths[idx]
        mask_path = self.mask_dir / img_path.with_suffix(".png").name

        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        transformed = self.tf(image=rgb, mask=mask)
        return transformed["image"].float(), transformed["mask"].long()


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    preds = logits.argmax(1)
    valid = target != 255
    correct = (preds[valid] == target[valid]).sum()
    total = valid.sum()
    pix_acc = (correct / total).item() if total > 0 else 0.0
    ious = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (target == cls) & valid).sum().item()
        union = ((preds == cls) | (target == cls)) & valid
        union = union.sum().item()
        if union > 0:
            ious.append(inter / union)
    miou = float(np.mean(ious)) if ious else 0.0
    return pix_acc, miou


def train_epoch(model, loader, opt, device, num_classes):
    model.train()
    tot_loss = tot_acc = tot_iou = 0.0
    for img, mask in tqdm(loader, desc="Train", leave=False):
        img, mask = img.to(device), mask.to(device)
        opt.zero_grad()
        out = model(pixel_values=img, labels=mask)
        loss = out.loss
        loss.backward()
        opt.step()
        acc, miou = compute_metrics(out.logits.detach(), mask, num_classes)
        n = img.size(0)
        tot_loss += loss.item() * n
        tot_acc += acc * n
        tot_iou += miou * n
    n_samples = len(loader.dataset)
    return tot_loss / n_samples, tot_acc / n_samples, tot_iou / n_samples


@torch.no_grad()
def eval_epoch(model, loader, device, num_classes):
    model.eval()
    tot_loss = tot_acc = tot_iou = 0.0
    for img, mask in tqdm(loader, desc="Val", leave=False):
        img, mask = img.to(device), mask.to(device)
        out = model(pixel_values=img, labels=mask)
        loss = out.loss
        acc, miou = compute_metrics(out.logits, mask, num_classes)
        n = img.size(0)
        tot_loss += loss.item() * n
        tot_acc += acc * n
        tot_iou += miou * n
    n_samples = len(loader.dataset)
    return tot_loss / n_samples, tot_acc / n_samples, tot_iou / n_samples


def save_ckpt(model, path: Path, epoch: int, miou: float):
    torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou}, path)


def log_header(path: Path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_loss", "val_acc", "val_miou"])


def log_row(path: Path, row: List):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def main():
    parser = argparse.ArgumentParser(description="SegFormer ViT training on RGB only")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--checkpoint-dir", default="runs/vit_rgb")
    parser.add_argument("--model-size", choices=["b0", "b1", "b2", "b3"], default="b2")
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    train_rgb = Path(args.train_dir) / "rgb"
    train_mask = Path(args.train_dir) / "masks"
    val_rgb = Path(args.val_dir) / "rgb"
    val_mask = Path(args.val_dir) / "masks"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = {i: f"class_{i}" for i in range(args.num_classes)}
    cfg = SegformerConfig.from_pretrained(
        f"nvidia/mit_{args.model_size}",
        num_channels=3,
        num_labels=args.num_classes,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_index=255,
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/mit_{args.model_size}", config=cfg, ignore_mismatched_sizes=True
    ).to(device)

    train_ds = RGBDataset(train_rgb, train_mask, size=args.img_size, augment=True)
    val_ds = RGBDataset(val_rgb, val_mask, size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train.csv"; log_header(log_path)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc, tr_iou = train_epoch(model, train_loader, opt, device, args.num_classes)
        val_loss, val_acc, val_iou = eval_epoch(model, val_loader, device, args.num_classes)
        sched.step(val_loss)
        log_row(log_path, [epoch, tr_loss, val_loss, val_acc, val_iou])
        print(f"loss={tr_loss:.3f}  val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  val_mIoU={val_iou:.3f}")

        if val_iou > best_miou:
            best_miou = val_iou
            save_ckpt(model, ckpt_dir / "best_model.pt", epoch, val_iou)
            print(f"✔ Saved best mIoU {val_iou:.3f} at epoch {epoch}")

    print(f"Training finished. Best mIoU = {best_miou:.3f}")


if __name__ == "__main__":
    main()
