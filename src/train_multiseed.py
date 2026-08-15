"""
Multi-seed training harness.

Runs the training pipeline across multiple random seeds for statistical rigor.
Produces per-seed checkpoints and aggregated results (mean ± std).

Usage:
    python src/train_multiseed.py --model_name fastvit_t8 --data_dir data/train
    python src/train_multiseed.py --model_name fastvit_t8 --data_dir data/train --seeds 42 123 456
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import get_dataloaders, mixup_data, mixup_criterion
from models import get_model


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_seed(args, seed):
    """Train a single model with a specific random seed. Returns validation accuracy."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Training {args.model_name} | Seed {seed}")
    print(f"{'='*60}")

    # Set up data with this seed's split
    train_loader, val_loader, _, class_to_idx = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, random_state=seed
    )
    num_classes = len(class_to_idx)

    # Load model
    model = get_model(args.model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    best_val_acc = 0.0
    save_path = os.path.join(args.save_dir, f"{args.model_name}_seed{seed}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if args.mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup_alpha)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.autocast(device_type='cpu'):
                outputs = model(images)
                if args.mixup_alpha > 0:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total

        print(f"  Seed {seed} | Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    return best_val_acc, save_path


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training for statistical rigor")
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
                        help="Random seeds for multi-run training (default: 5 seeds)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results = []
    for seed in args.seeds:
        val_acc, ckpt_path = train_single_seed(args, seed)
        results.append({
            "model": args.model_name,
            "seed": seed,
            "best_val_acc": val_acc,
            "checkpoint": ckpt_path
        })

    # Aggregate and report
    accs = [r["best_val_acc"] for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED RESULTS: {args.model_name}")
    print(f"{'='*60}")
    for r in results:
        print(f"  Seed {r['seed']}: {r['best_val_acc']:.2f}%")
    print(f"  Mean +/- Std: {mean_acc:.2f} +/- {std_acc:.2f}%")
    print(f"{'='*60}")

    # Save per-seed results
    df = pd.DataFrame(results)
    results_file = f"results/multiseed_{args.model_name}.csv"
    df.to_csv(results_file, index=False)
    print(f"Per-seed results saved to {results_file}")

    # Append summary to an aggregate file
    summary_file = "results/multiseed_summary.csv"
    summary_data = {
        "Model": [args.model_name],
        "Mean_Acc": [mean_acc],
        "Std_Acc": [std_acc],
        "Num_Seeds": [len(args.seeds)],
        "Seeds": [str(args.seeds)]
    }
    df_summary = pd.DataFrame(summary_data)
    if os.path.exists(summary_file):
        df_existing = pd.read_csv(summary_file)
        df_existing = pd.concat([df_existing[df_existing['Model'] != args.model_name], df_summary], ignore_index=True)
        df_existing.to_csv(summary_file, index=False)
    else:
        df_summary.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
