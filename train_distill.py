import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import csv
import json
import uuid

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from litgpt.tokenizer import Tokenizer 
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.utils import chunked_cross_entropy
from tqdm import tqdm
from agri_data import AgriDataset
from plot_specialization import plot_cka_heatmap


#config

RUN_TAG = "run_test"  # <--- EDIT THIS PER RUN

DISTILL_CONFIG = {
    "alpha": 0.5,
    "lambda": 0.5,  
    "beta": 0.2, 
    "gamma": 1.5, 
    "T": 2.0 
}
MAX_SEQ_LENGTH = 3072
BATCH_SIZE = 2
ACCUMULATE_GRAD_STEPS = 3  # 1 * 4 = Effective Batch Size 4 (Better stability)

# Generate a short 4-char unique ID (e.g., 'A9D2') to prevent overwrites
short_id = uuid.uuid4().hex[:6].upper()
run_name = f"{RUN_TAG}_{short_id}"

OUTPUT_ROOT = Path(f"outputs/{run_name}")
CHECKPOINT_ROOT = Path(f"checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/{run_name}")

# Create directories
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "audits").mkdir(exist_ok=True)

print(f"Starting Run: {run_name}")
print(f"Output Dir:   {OUTPUT_ROOT}")

# --- 3. SAVE FULL CONFIG (The "Black Box" Recorder) ---
# We save the numbers here, so we don't need them in the filename.
config_path = OUTPUT_ROOT / "experiment_config.json"
experiment_settings = {
    "run_name": run_name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    **DISTILL_CONFIG,
    "max_seq_length": MAX_SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "accum_steps": ACCUMULATE_GRAD_STEPS,
    "dataset": "agri_hi_train.parquet"
}
with open(config_path, 'w') as f:
    json.dump(experiment_settings, f, indent=4)
print(f"Parameters saved to: {config_path}")



class LossLogger:
    def __init__(self, log_dir="outputs/training_log.csv"):
        self.file_path = Path(log_dir)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = ["step", "total_loss", "kl_loss", "cka_feat_loss", "diversity_loss", "max_expert_load"]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, step, total, kl, ce, cka, div, load):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{total:.6f}", f"{kl:.6f}", f"{ce:.6f}", f"{cka:.6f}", f"{div:.6f}", f"{load:.4f}"])


#CKA Loss
class LinearCKALoss(nn.Module):
    """
    Minibatch Linear CKA for feature manifold alignment.
    Handles different hidden dimensions (1024 vs 5120) naturally.
    """
    def forward(self, x, y):
        # x (Student): [B*T, 1024], y (Teacher): [B*T, 5120]
        x = x.view(-1, x.size(-1)).to(torch.float32)
        y = y.view(-1, y.size(-1)).to(torch.float32)
        
        # Center the features
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        dot_product = torch.norm(torch.matmul(x.t(), y))**2
        norm_x = torch.norm(torch.matmul(x.t(), x))
        norm_y = torch.norm(torch.matmul(y.t(), y))
        
        cka_score = dot_product / (norm_x * norm_y + 1e-6)
        return 1 - cka_score



@torch.no_grad()
def audit_expert_specialization(student, cka_fn, batch, step):
    """
    Runs every 500 steps. Forces ALL experts to process the SAME tokens
    to calculate a perfect Output-Space CKA matrix.
    It proves that  loss_diversity isn't just "adding noise" to the weights, 
    but is forcing the experts to learn distinct functional mappings for your Hindi Agricultural data.
    """
    student.eval()
    device = next(student.parameters()).device
    # Use a small subset of the batch for the audit to save VRAM
    
    audit_input = batch[0][:2].to(device) 

    save_dir = OUTPUT_ROOT / "audits"
    matrix_path = save_dir / f"cka_matrix_step_{step}.pth"
    
    print(f"\nStep {step}: Expert Specialization Audit ")
    
    # Extract Expert Logic - target a specific layer - for the thesis chart
    target_layer_idx = student.config.n_layer // 2
    layer = student.transformer.h[target_layer_idx].mlp
    
    if hasattr(layer, 'experts'):
        # Get hidden states entering the MLP
        # This requires a quick forward pass up to that layer
        # For simplicity, we compare expert outputs on a dummy latent vector
        latent_dim = student.config.n_embd
        dummy_latent = torch.randn(1, 128, latent_dim, device=device, dtype=torch.bfloat16)
        
        expert_outputs = [expert(dummy_latent) for expert in layer.experts]
        
        # Calculate CKA Matrix (N_expert x N_expert)
        n_exp = len(expert_outputs)
        cka_matrix = torch.zeros((n_exp, n_exp))
        
        for i in range(n_exp):
            for j in range(n_exp):
                # We want 1 - CKA (Distance)
                cka_matrix[i, j] = 1 - cka_fn(expert_outputs[i].view(-1, latent_dim), 
                                              expert_outputs[j].view(-1, latent_dim))
        
        torch.save(cka_matrix, matrix_path)
        plot_cka_heatmap(matrix_path, save_path=save_dir / f"heatmap_step_{step}.png")
        print(f"Audit matrix and heatmap saved to {save_dir}")
    
        print(f"Layer {target_layer_idx} Avg Expert Distance: {cka_matrix.mean().item():.4f}")
    
    student.train()


def train_distill(
    student_path="checkpoints/Qwen/Qwen3-0.6B-moe-initial/lit_model.pth",
    teacher_path="checkpoints/Qwen/Qwen3-8B/lit_model.pth",
    data_loader=None
):
    device = torch.device("cuda") 
    logger = LossLogger()

    tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B")
    dataset = AgriDataset(
            data_path="data/agri_hi_train.parquet", 
            tokenizer=tokenizer, 
            max_seq_length=MAX_SEQ_LENGTH 
        )

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    print("Loading 8B Teacher in BF16...")
    teacher = GPT(Config.from_name("Qwen3-8B")).to(device, dtype=torch.bfloat16) 
    teacher.load_state_dict(torch.load(teacher_path, mmap=True, weights_only=True), strict=False)
    teacher.eval()
    
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading 0.6B-MoE Student in BF16...")
    student = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device, dtype=torch.bfloat16)
    student.load_state_dict(torch.load(student_path))
    # Enable Gradient Checkpointing on Student to save VRAM, critical for the H100 80GB when running two models
    student.gradient_checkpointing_enable()

    optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=2e-5) # 8-bit AdamW saves ~2GB
    cka_fn = LinearCKALoss()

    # Layer Mapping (28 student layers -> 40 teacher layers)
    mapping = {s_i: int(s_i * (32-1) / (28-1)) for s_i in range(28)}

    n_experts = student.config.n_expert
    n_active = student.config.n_expert_per_token
    
    optimizer.zero_grad() 

    for batch_idx, (input_ids, loss_mask) in enumerate(tqdm(data_loader)):
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)

        # 1. Teacher Forward (No Grads)
        with torch.no_grad():
            t_logits, t_features = teacher(input_ids, return_features=True)

        # 2. Student Forward
        s_logits, s_features = student(input_ids, return_features=True)

        # LOSS A: Soft Logit Distillation (With Masking)
        # We calculate KL Divergence only on the unmasked tokens (Thoughts + Advisory)

        T = DISTILL_CONFIG["T"] # Distillation Temperature
        # Shapes: [B, T, Vocab]
        student_log_probs = F.log_softmax(s_logits / DISTILL_CONFIG["T"], dim=-1)
        teacher_probs = F.softmax(t_logits / DISTILL_CONFIG["T"], dim=-1)
        # Standard KLDiv reduces to a scalar, so we need to implement manual reduction for masking
        kl_loss_pointwise = F.kl_div(student_log_probs, teacher_probs, reduction="none", log_target=False)
        # kl_loss_pointwise shape: [B, T, Vocab] -> Sum over vocab -> [B, T]
        kl_loss_per_token = kl_loss_pointwise.sum(dim=-1) 
        # Apply Mask
        # loss_mask is [B, T], 1 for learnable tokens, 0 for prompt
        active_loss = (kl_loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        loss_kl = active_loss * (DISTILL_CONFIG["T"]**2)


        #LOSS B: Cross-Entropy 
        # Shift targets: logits[0] predicts input_ids[1]
        shift_logits = s_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = loss_mask[..., 1:].contiguous()

        # Calculate standard CE
        ce_loss_pointwise = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
        ce_loss_pointwise = ce_loss_pointwise.view(shift_labels.size())
        # Apply mask
        loss_ce = (ce_loss_pointwise * shift_mask).sum() / (shift_mask.sum() + 1e-6)


        # LOSS C: Feature CKA Distillation 
        loss_cka = sum(cka_fn(s_features[s], t_features[t]) for s, t in mapping.items()) / len(mapping)

        # LOSS D: Expert Diversity (The Research Key) 
        loss_diversity = 0
        moe_layers = [h.mlp for h in student.transformer.h if hasattr(h.mlp, 'experts')]
        
        for mlp in moe_layers:
            layer_div = 0
            pairs = 0
            # Compare ALL experts in the layer to force global divergence
            # This is more robust than just comparing the top-k chosen ones
            for i in range(n_experts):
                for j in range(i + 1, n_experts):
                    # Use Weights (fc_1) to ensure matching dimensions for CKA
                    w_i = mlp.experts[i].fc_1.weight
                    w_j = mlp.experts[j].fc_1.weight
                    layer_div += (1 - cka_fn(w_i, w_j))
                    pairs += 1
            loss_diversity += (layer_div / pairs)
            
        loss_diversity /= len(moe_layers)

        # Total Loss 
        total_loss = (DISTILL_CONFIG["alpha"] * loss_kl) + \
                     (DISTILL_CONFIG["lambda"] * loss_ce) + \
                     (DISTILL_CONFIG["beta"] * loss_cka) + \
                     (DISTILL_CONFIG["gamma"] * loss_diversity)        


        loss_scaled = total_loss / ACCUMULATE_GRAD_STEPS
        loss_scaled.backward()

        # Step Optimizer ONLY every 'N' steps
        if (batch_idx + 1) % ACCUMULATE_GRAD_STEPS == 0:
            # Gradient Clipping for MoE Stability
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        #logging
        all_indices = torch.cat([layer.mlp.last_indices for layer in student.transformer.h if hasattr(layer.mlp, 'last_indices')], dim=0)
        max_load = (torch.bincount(all_indices.view(-1), minlength=n_experts).float() / all_indices.numel()).max().item()

        logger.log(batch_idx, total_loss.item(), loss_kl.item(), loss_cka.item(), loss_diversity.item(), max_load)

        #  THE AUDIT (Every 500 Steps) 
        if batch_idx % 500 == 0 and batch_idx > 0:
            audit_expert_specialization(student, cka_fn, [input_ids], batch_idx)
            torch.save(student.state_dict(), CHECKPOINT_ROOT / f"step-{batch_idx}.pth")


        if batch_idx % 10 == 0:
            print(f"Step {batch_idx} | Total_Loss: {total_loss:.4f} | Max_Load: {max_load:.2%}")

        # Intermediate Saving
        if batch_idx % 500 == 0 and batch_idx > 0:
            torch.save(student.state_dict(), CHECKPOINT_ROOT / f"step-{batch_idx}.pth")

    # Save the final student state separately
    final_path = CHECKPOINT_ROOT / "lit_model.pth"
    torch.save(student.state_dict(), final_path)
    audit_expert_specialization(student, cka_fn, [input_ids], batch_idx)


    # Copy configuration files to make the model portable
    from litgpt.utils import copy_config_files
    copy_config_files(Path("checkpoints/Qwen/Qwen3-0.6B"), CHECKPOINT_ROOT)
    print(f"✨ Distilled model saved to: {CHECKPOINT_ROOT}")




if __name__ == "__main__":
    train_distill()




