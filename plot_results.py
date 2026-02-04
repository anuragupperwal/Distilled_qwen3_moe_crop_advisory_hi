import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
log_file = "outputs/training_log.csv"
output_img = "outputs/training_plots.png"

def plot_training_metrics():
    if not Path(log_file).exists():
        print(f"❌ Error: Log file not found at {log_file}")
        return

    # Load Data
    df = pd.read_csv(log_file)
    print(f"Loaded {len(df)} steps from log.")

    # Set Style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Loss Convergence ---
    sns.lineplot(data=df, x="step", y="total_loss", ax=axes[0], label="Total Loss", color="#1f77b4")
    sns.lineplot(data=df, x="step", y="kl_loss", ax=axes[0], label="Distillation (KL) Loss", color="#ff7f0e", linestyle="--")
    axes[0].set_title("Training Convergence")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Training Steps")
    axes[0].legend()

    # --- Plot 2: CKA Alignment ---
    # Low value = High similarity to Teacher
    sns.lineplot(data=df, x="step", y="cka_feat_loss", ax=axes[1], color="#2ca02c")
    axes[1].set_title("Teacher-Student Alignment (CKA)")
    axes[1].set_ylabel("Feature Distance (1 - CKA)")
    axes[1].set_xlabel("Training Steps")

    # --- Plot 3: MoE Expert Load ---
    sns.lineplot(data=df, x="step", y="max_expert_load", ax=axes[2], color="#d62728")
    axes[2].set_title("Expert Specialization Health")
    axes[2].set_ylabel("Max Expert Load (%)")
    axes[2].set_xlabel("Training Steps")
    
    # Draw reference line for ideal load (assuming 8 experts, ~12.5% is perfectly balanced)
    axes[2].axhline(y=0.15, color='gray', linestyle='--', alpha=0.7, label="Balanced Threshold (~15%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"✅ Plots saved successfully to: {output_img}")

if __name__ == "__main__":
    plot_training_metrics()