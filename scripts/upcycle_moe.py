import os
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from litgpt.model import GPT
from litgpt.config import Config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def upcycle_checkpoint(
    dense_ckpt_path: Path,
    output_dir: Path,
    student_config_name: str = "Qwen3-0.6B-MoE"
):
    """
    Performs weight surgery to convert a dense model to an MoE model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") # on C PU to avoid OOM

    # Load Dense Weights
    logger.info(f"Loading dense weights from {dense_ckpt_path}")
    dense_state_dict = torch.load(dense_ckpt_path, map_location=device, weights_only=True)

    # Initialize Student Model with MoE Config
    logger.info(f"Initializing student model architecture: {student_config_name}")
    config = Config.from_name(student_config_name)
    student_model = GPT(config)
    student_state_dict = student_model.state_dict()

    # 3. Perform Weight Surgery
    new_state_dict = {}
    
    # Track which keys we've mapped for verification
    mapped_keys = set()

    for key, weight in dense_state_dict.items():
        # Handle MLP to MoE mapping
        # Dense: transformer.h.0.mlp.fc_1.weight
        # Student: transformer.h.0.mlp.experts.0.fc_1.weight, ... .experts.7.fc_1.weight
        if ".mlp." in key:
            for i in range(config.n_expert):
                # Construct the new expert key
                expert_key = key.replace(".mlp.", f".mlp.experts.{i}.")
                
                # Copy weight + Add tiny jitter to break symmetry
                jitter = torch.randn_like(weight) * 0.02
                new_state_dict[expert_key] = weight.clone() + jitter
                mapped_keys.add(expert_key)
        else:
            # Handle non-MLP layers (Attention, Norm, Embedding, Head)
            if key in student_state_dict:
                new_state_dict[key] = weight.clone()
                mapped_keys.add(key)

    # 4. Initialize the Router (Gate) weights
    # Routers are new, so we initialize them randomly using model's default init
    for key in student_state_dict.keys():
        if ".gate." in key and key not in new_state_dict:
            logger.info(f"Initializing new router weights: {key}")
            # Router weights are usually kept small
            new_state_dict[key] = torch.randn_like(student_state_dict[key]) * 0.02
            mapped_keys.add(key)

    # 5. Final Safety Check and Save
    missing_keys = set(student_state_dict.keys()) - mapped_keys
    if missing_keys:
        logger.warning(f"Weights missing for keys: {missing_keys}")
    
    save_path = output_dir / "lit_model.pth"
    torch.save(new_state_dict, save_path)
    logger.info(f"Successfully saved upcycled MoE model to {save_path}")


def verify_experts_and_plot(folder_path):
    device = torch.device("cpu")    
    
    ckpt_path = folder_path / "lit_model.pth"
    print(f"\n🔍 Verifying Upcycled Model: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"Error: File not found at {ckpt_path}")
        return
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # We manually extract expert weights from Layer 14 (Middle Layer)
    # Key format: transformer.h.14.mlp.experts.0.fc_1.weight
    layer_idx = 14
    n_experts = 8
    
    print(f"Analyzing Layer {layer_idx} Experts...")
    
    expert_weights = []
    for i in range(n_experts):
        key = f"transformer.h.{layer_idx}.mlp.experts.{i}.fc_1.weight"
        if key not in state_dict:
            print(f"Key not found: {key}")
            return
        expert_weights.append(state_dict[key].float()) # Convert to float32 for CKA calc

    # Compute Full Pairwise Similarity Matrix ---
    sim_matrix = torch.zeros((n_experts, n_experts))
    
    for i in range(n_experts):
        for j in range(n_experts):
            sim = torch.nn.functional.cosine_similarity(
                expert_weights[i].flatten().unsqueeze(0), 
                expert_weights[j].flatten().unsqueeze(0)
            ).item()
            sim_matrix[i, j] = sim

    #  TEXT REPORT 
    ref_expert_sims = sim_matrix[0] # Row 0
    print("\n Similarity Check (Expert 0 vs Others)")
    print("Goal: Values should be < 0.9999 (Green)")
    
    all_distinct = True
    for i in range(n_experts):
        sim = ref_expert_sims[i].item()
        status = "🔴 CLONE (FAIL)" if sim > 0.99999 and i != 0 else "🟢 DISTINCT (PASS)"
        if i == 0: status = "Self (1.0)"
        if "FAIL" in status: all_distinct = False
        print(f"Expert 0 vs Expert {i}: {sim:.6f} [{status}]")

    # PLOT HEATMAP 
    print("\n🎨 Generating Heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix.numpy(), 
        annot=True, 
        fmt=".4f", 
        cmap="YlGnBu", 
        xticklabels=[f"Exp {i}" for i in range(n_experts)],
        yticklabels=[f"Exp {i}" for i in range(n_experts)]
    )
    plt.title(f"Expert Weight Similarity (Layer {layer_idx})\n(1.0 = Identical, <0.999 = Distinct)")
    
    plot_path = folder_path / "expert_init_heatmap.png"
    plt.savefig(plot_path)
    print(f"✅ Heatmap saved to: {plot_path}")
    plt.close()
        

if __name__ == "__main__":
    # Example usage - Update paths as per your local setup
    DENSE_CKPT = Path("checkpoints/Qwen/Qwen3-0.6B/lit_model.pth")
    OUTPUT_DIR = Path("checkpoints/Qwen/Qwen3-0.6B-moe-init-noisy/")
    
    upcycle_checkpoint(DENSE_CKPT, OUTPUT_DIR)
    verify_experts_and_plot(OUTPUT_DIR)







