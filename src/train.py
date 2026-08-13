import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders, mixup_data, mixup_criterion
from models import get_model

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up data
    print(f"Loading data from {args.data_dir}...")
    try:
        train_loader, val_loader, _, class_to_idx = get_dataloaders(args.data_dir, batch_size=args.batch_size)
        num_classes = len(class_to_idx)
    except FileNotFoundError:
        print(f"Warning: Data directory {args.data_dir} not found. Ensure you download the dataset first.")
        num_classes = 27

    # Load model
    model = get_model(args.model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize Mixed Precision Scaler to save VRAM
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(train_loader, desc="Training") if 'train_loader' in locals() else []
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if args.mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup_alpha)

            optimizer.zero_grad()
            
            # Use AMP for forward pass to save VRAM
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.autocast(device_type='cpu'):
                outputs = model(images)
                
                if args.mixup_alpha > 0:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            # Backward pass with Scaler
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            
            if 'pbar' in locals() and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=loss.item())

        if running_loss > 0:
            scheduler.step()

        if 'val_loader' in locals():
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc = 100 * correct / total
            
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        else:
            print("Skipping validation due to missing dataloader.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--model_name", type=str, default="mobilenetv2_100")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    train(args)
