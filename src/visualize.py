import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_bubble_chart(csv_path="results/benchmark_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Please run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    # Create bubble chart
    # X: Latency, Y: Accuracy, Size: Parameters
    scatter = sns.scatterplot(
        data=df, 
        x="CPU_Latency_ms", 
        y="Top1_Accuracy", 
        size="Parameters_M",
        sizes=(100, 1500),
        hue="Model",
        palette="deep",
        alpha=0.7,
        edgecolor="k"
    )
    
    # Annotate points
    for i, row in df.iterrows():
        plt.text(row["CPU_Latency_ms"] + 0.5, row["Top1_Accuracy"] + 0.5, 
                 row["Model"], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plt.title("Accuracy vs. Inference Latency (Bubble Size = Parameter Count)", fontsize=16)
    plt.xlabel("CPU Inference Latency (ms)", fontsize=14)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=14)
    
    # Adjust legend
    h, l = scatter.get_legend_handles_labels()
    plt.legend(h[1:len(df)+1], l[1:len(df)+1], title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/accuracy_vs_latency.png", dpi=300)
    print("Saved bubble chart to results/plots/accuracy_vs_latency.png")

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix for Best Model", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/confusion_matrix.png", dpi=300)
    print("Saved confusion matrix to results/plots/confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['bubble', 'cm'], default='bubble')
    args = parser.parse_args()
    
    if args.type == 'bubble':
        plot_bubble_chart()
    else:
        print("To plot a confusion matrix, please pass true and predicted labels directly in your script.")
