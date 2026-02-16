import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cka_heatmap(matrix_path, save_path="expert_specialization.png"):
    # Load the matrix saved during the Audit in train_distill.py
    # Matrix shape should be (n_experts, n_experts)
    cka_matrix = torch.load(matrix_path, weights_only=True).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cka_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        xticklabels=[f"Exp {i}" for i in range(len(cka_matrix))],
        yticklabels=[f"Exp {i}" for i in range(len(cka_matrix))]
    )
    
    plt.title("Expert Output Diversity (1 - CKA Score)\nHigher = More Specialized Experts")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    plot_cka_heatmap("checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_E86460/step-500.pth")
    pass