import argparse
import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from dataset import get_dataloaders
from models import get_model
from sklearn.metrics import f1_score, accuracy_score
from fvcore.nn import FlopCountAnalysis, parameter_count

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {args.model_name} on {device}")

    try:
        _, _, test_loader, class_to_idx = get_dataloaders(args.data_dir, batch_size=1, is_test=True, classes_dir="data/train")
        num_classes = len(class_to_idx)
    except FileNotFoundError:
        print(f"Warning: Data directory {args.data_dir} not found. Using dummy dataloader...")
        test_loader = []
        num_classes = 27

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
    
    # 3. Accuracy and F1 Score (only if dataloader exists)
    y_true = []
    y_pred = []
    
    model = model.to(device)
    if test_loader:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        acc = accuracy_score(y_true, y_pred) * 100
        macro_f1 = f1_score(y_true, y_pred, average='macro')
    else:
        acc, macro_f1 = 0.0, 0.0

    print(f"--- Results for {args.model_name} ---")
    print(f"Parameters (M): {params / 1e6:.2f}")
    print(f"FLOPs (G): {total_flops / 1e9:.2f}")
    print(f"CPU Latency (ms): {avg_latency:.2f}")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    results_file = "results/benchmark_metrics.csv"
    
    new_data = {
        "Model": [args.model_name],
        "Top1_Accuracy": [acc],
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/test")
    parser.add_argument("--weights", type=str, default="")
    args = parser.parse_args()
    
    evaluate(args)
