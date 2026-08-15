import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Clean model name mapping: timm model ID -> paper-friendly name
MODEL_NAMES = {
    "fastvit_t8": "FastViT-T8",
    "edgenext_small": "EdgeNeXt-S",
    "resnet50": "ResNet50",
    "mobilenetv4_conv_small": "MobileNetV4-CS",
}

def clean_model_name(name):
    """Convert timm model IDs to clean paper-friendly names."""
    return MODEL_NAMES.get(name, name)


def plot_bubble_chart(csv_path="results/benchmark_metrics.csv", save_dir="results/plots"):
    """Generate accuracy vs. latency bubble chart with clean formatting."""
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Please run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)
    df["Model_Clean"] = df["Model"].apply(clean_model_name)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_theme(style="whitegrid")

    # Color palette
    palette = {"FastViT-T8": "#2196F3", "EdgeNeXt-S": "#4CAF50", 
               "ResNet50": "#FF9800", "MobileNetV4-CS": "#9C27B0"}

    # Draw bubbles
    for _, row in df.iterrows():
        name = row["Model_Clean"]
        ax.scatter(
            row["CPU_Latency_ms"], row["Top1_Accuracy"],
            s=row["Parameters_M"] * 60,  # Scale bubble size
            color=palette.get(name, "#999"),
            alpha=0.7, edgecolor="k", linewidth=1.0,
            zorder=3
        )

    # Annotate points with smart placement to avoid overlaps
    offsets = {
        "FastViT-T8": (-8, -2.0),
        "EdgeNeXt-S": (2.0, -1.0),
        "ResNet50": (1.5, 0.5),
        "MobileNetV4-CS": (1.5, 1.0),
    }
    for _, row in df.iterrows():
        name = row["Model_Clean"]
        dx, dy = offsets.get(name, (1.5, 0.5))
        ax.annotate(
            name,
            xy=(row["CPU_Latency_ms"], row["Top1_Accuracy"]),
            xytext=(row["CPU_Latency_ms"] + dx, row["Top1_Accuracy"] + dy),
            fontsize=11, fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
            zorder=4
        )

    ax.set_title("Accuracy vs. CPU Inference Latency", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("CPU Inference Latency (ms)", fontsize=13)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=13)
    ax.tick_params(labelsize=11)

    # Add parameter count legend (manual bubble size legend)
    for param_val in [3, 10, 24]:
        ax.scatter([], [], s=param_val * 60, c="gray", alpha=0.5, edgecolor="k",
                   label=f"{param_val}M params")
    ax.legend(title="Parameter Count", loc="lower right", fontsize=10, title_fontsize=11,
              framealpha=0.9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "accuracy_vs_latency.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bubble chart to {save_path}")
    return save_path


def plot_metrics_comparison(csv_path="results/benchmark_metrics.csv", save_dir="results/plots"):
    """Generate grouped bar chart comparing accuracy and Macro F1 across models."""
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Please run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)
    df["Model_Clean"] = df["Model"].apply(clean_model_name)

    # Prepare data for grouped bar chart
    models = df["Model_Clean"].tolist()
    accuracy = df["Top1_Accuracy"].tolist()
    macro_f1_pct = [f * 100 for f in df["Macro_F1"].tolist()]  # Scale to percentage

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, accuracy, width, label="Top-1 Accuracy (%)", 
                   color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, macro_f1_pct, width, label="Macro F1 (×100)", 
                   color="#FF9800", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Use broken y-axis starting from 55 to show differences clearly
    ax.set_ylim(55, 75)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Top-1 Accuracy and Macro F1 Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    # Add break marks at bottom to indicate non-zero start
    ax.spines['bottom'].set_visible(True)
    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax.plot((-d, +d), (0 - d, 0 + d), **kwargs)
    ax.plot((-d, +d), (0 + d, 0 + 3*d), **kwargs)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to {save_path}")
    return save_path


def plot_confusion_matrix(predictions_file, save_dir="results/plots"):
    """Generate confusion matrix heatmap from saved predictions."""
    if not os.path.exists(predictions_file):
        print(f"Predictions file {predictions_file} not found. Run evaluate.py first.")
        return

    data = np.load(predictions_file, allow_pickle=True)
    y_true = data['y_true']
    y_pred = data['y_pred']
    class_names = data['class_names']

    # Shorten class names for readability
    short_names = []
    for name in class_names:
        # e.g., "Tomato_leaf_bacterial_spot" -> "Tom. Bact. Spot"
        short = name.replace("_leaf", "").replace("_", " ")
        if len(short) > 20:
            parts = short.split()
            short = " ".join(p[:4] + "." if len(p) > 4 else p for p in parts)
        short_names.append(short)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='white')

    ax.set_title("Confusion Matrix — FastViT-T8 on PlantDoc", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=13)
    ax.set_ylabel("True Class", fontsize=13)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")
    return save_path


def copy_figures_to_paper(plots_dir="results/plots", paper_figures_dir="paper/figures"):
    """Copy generated plots to the paper figures directory."""
    import shutil
    os.makedirs(paper_figures_dir, exist_ok=True)
    for filename in ["accuracy_vs_latency.png", "metrics_comparison.png", "confusion_matrix.png"]:
        src = os.path.join(plots_dir, filename)
        dst = os.path.join(paper_figures_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {filename} to {paper_figures_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['bubble', 'metrics', 'cm', 'all'], default='all',
                        help="Type of plot to generate")
    parser.add_argument("--csv", type=str, default="results/benchmark_metrics.csv",
                        help="Path to benchmark metrics CSV")
    parser.add_argument("--predictions", type=str, default="results/predictions_fastvit_t8.npz",
                        help="Path to predictions .npz file (for confusion matrix)")
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="Copy generated figures to paper/figures/")
    args = parser.parse_args()

    if args.type in ('bubble', 'all'):
        plot_bubble_chart(args.csv)
    if args.type in ('metrics', 'all'):
        plot_metrics_comparison(args.csv)
    if args.type in ('cm', 'all'):
        plot_confusion_matrix(args.predictions)
    if args.copy_to_paper:
        copy_figures_to_paper()
