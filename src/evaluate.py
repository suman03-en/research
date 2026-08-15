import argparse
import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import get_dataloaders
from models import get_model
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support
from fvcore.nn import FlopCountAnalysis, parameter_count

def topk_accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {args.model_name} on {device}")

    try:
        _, _, test_loader, class_to_idx = get_dataloaders(args.data_dir, batch_size=1, is_test=True, classes_dir="data/train")
        num_classes = len(class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    except FileNotFoundError:
        print(f"Warning: Data directory {args.data_dir} not found. Using dummy dataloader...")
        test_loader = []
        num_classes = 27
        idx_to_class = {}

    model = get_model(args.model_name, num_classes=num_classes, pretrained=False)
    
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    
    model = model.to(device)
    model.eval()

    # 1. Measure FLOPs and Parameters
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    params = sum(p.numel() for p in model.parameters())
    
    # 2. CPU Latency (force CPU for this test)
    model_cpu = model.to('cpu')
    dummy_input_cpu = torch.randn(1, 3, 224, 224).to('cpu')
    
    # Warmup
    for _ in range(10):
        _ = model_cpu(dummy_input_cpu)
        
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            start_time = time.perf_counter()
            _ = model_cpu(dummy_input_cpu)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # ms
            
    avg_latency = sum(latencies) / len(latencies)
    
    # 3. Accuracy, Top-5 Accuracy, and F1 Score
    y_true = []
    y_pred = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    model = model.to(device)
    if test_loader:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels_dev = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                # Top-k accuracy
                batch_top1, batch_top5 = topk_accuracy(outputs, labels_dev, topk=(1, min(5, num_classes)))
                top1_correct += batch_top1
                top5_correct += batch_top5
                total += labels.size(0)
                
        acc = accuracy_score(y_true, y_pred) * 100
        top5_acc = (top5_correct / total) * 100
        macro_f1 = f1_score(y_true, y_pred, average='macro')
    else:
        acc, top5_acc, macro_f1 = 0.0, 0.0, 0.0

    print(f"--- Results for {args.model_name} ---")
    print(f"Parameters (M): {params / 1e6:.2f}")
    print(f"FLOPs (G): {total_flops / 1e9:.2f}")
    print(f"CPU Latency (ms): {avg_latency:.2f}")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    results_file = "results/benchmark_metrics.csv"
    
    new_data = {
        "Model": [args.model_name],
        "Top1_Accuracy": [acc],
        "Top5_Accuracy": [top5_acc],
        "Macro_F1": [macro_f1],
        "Parameters_M": [params / 1e6],
        "FLOPs_G": [total_flops / 1e9],
        "CPU_Latency_ms": [avg_latency]
    }
    df_new = pd.DataFrame(new_data)
    
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        # update or append
        df = pd.concat([df[df['Model'] != args.model_name], df_new], ignore_index=True)
    else:
        df = df_new
        
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Per-class metrics
    if test_loader and y_true and idx_to_class:
        class_names = [idx_to_class[i] for i in range(num_classes)]
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )
        
        per_class_data = {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_per_class,
            "Support": support
        }
        df_per_class = pd.DataFrame(per_class_data)
        per_class_file = f"results/per_class_metrics_{args.model_name}.csv"
        df_per_class.to_csv(per_class_file, index=False)
        print(f"Per-class metrics saved to {per_class_file}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        # Save predictions for confusion matrix generation
        preds_file = f"results/predictions_{args.model_name}.npz"
        np.savez(preds_file, y_true=np.array(y_true), y_pred=np.array(y_pred), 
                 class_names=np.array(class_names))
        print(f"Predictions saved to {preds_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/test")
    parser.add_argument("--weights", type=str, default="")
    args = parser.parse_args()
    
    evaluate(args)
