# src/train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Import custom modules
from data.dataset import ZebrafishAugmentedDataset
from models.architectures import ConvBiLSTM
from utils.visualization import plot_confusion_matrix, save_classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Zebrafish Behavior Classification Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing .npy classes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=150, help="Window size (frames)")
    return parser.parse_args()

def prepare_data_paths(root_dir):
    """Scan directory and split into Train, Val, and Test sets."""
    all_paths, all_labels = [], []
    class_names = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(root_dir, cls)
        for f in os.listdir(cls_path):
            if f.endswith('.npy'):
                all_paths.append(os.path.join(cls_path, f))
                all_labels.append(class_to_idx[cls])

    # Calculate class weights to handle imbalance
    weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)

    # 75/15/10 Split
    train_p, temp_p, train_l, temp_l = train_test_split(all_paths, all_labels, test_size=0.25, stratify=all_labels, random_state=42)
    val_p, test_p, val_l, test_l = train_test_split(temp_p, temp_l, test_size=0.4, stratify=temp_l, random_state=42)

    return (train_p, val_p, test_p), (train_l, val_l, test_l), class_names, class_weights

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training started on device: {device}")

    # 1. Load data paths
    paths, labels, class_names, class_weights = prepare_data_paths(args.data_dir)
    class_weights = class_weights.to(device)

    # 2. Setup Dataloaders
    train_ds = ZebrafishAugmentedDataset(paths[0], labels[0], window_size=args.window_size, training=True)
    val_ds = ZebrafishAugmentedDataset(paths[1], labels[1], window_size=args.window_size, training=False)
    test_ds = ZebrafishAugmentedDataset(paths[2], labels[2], window_size=args.window_size, training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 3. Initialize Model, Criterion, and Optimizer
    # ResNet50 + Delta Trick results in 4096 input dimensions
    model = ConvBiLSTM(input_dim=4096, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # 4. Training Loop
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_acc += (out.argmax(1) == y).sum().item() / len(y)

        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_acc += (out.argmax(1) == y).sum().item() / len(y)
        
        val_acc /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    # 5. Final Evaluation on Test Set
    print("\n📊 Evaluating on Test Set...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_trues.extend(y.cpu().numpy())

    # 6. Generate Reports and Visualization
    plot_confusion_matrix(all_trues, all_preds, class_names, "test_confusion_matrix.png")
    save_classification_report(all_trues, all_preds, class_names, "test_report.csv")
    print("✅ Training and Evaluation complete!")

if __name__ == "__main__":
    main()