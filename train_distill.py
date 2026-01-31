import torch
import torch.nn as nn
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
import lightning as L

class DistillProjector(nn.Module):
    """Upscales Student 1024-dim to Teacher 5120-dim for comparison."""
    def __init__(self, s_dim=1024, t_dim=5120):
        super().__init__()
        self.proj = nn.Linear(s_dim, t_dim, bias=False)
        
    def forward(self, x):
        return self.proj(x)

def cka_loss(experts_outputs):
    """
    Computes a diversity penalty using a simplified CKA-like correlation.
    experts_outputs: List of tensors [batch, seq, dim] from each expert.
    """
    if len(experts_outputs) < 2: return 0.0
    
    # We want to minimize the correlation between experts
    # to prevent expert collapse.
    loss = 0.0
    for i in range(len(experts_outputs)):
        for j in range(i + 1, len(experts_outputs)):
            # Flatten to (tokens, dim)
            ei = experts_outputs[i].view(-1, experts_outputs[i].size(-1))
            ej = experts_outputs[j].view(-1, experts_outputs[j].size(-1))
            
            # Simple Cosine Similarity penalty (proxy for CKA diversity)
            sim = F.cosine_similarity(ei, ej, dim=-1).mean()
            loss += sim
    return loss / (len(experts_outputs) * (len(experts_outputs)-1) / 2)

class DistillTask(L.LightningModule):
    def __init__(self, student_path, teacher_path):
        super().__init__()
        # Load Architecture Configs
        self.s_config = Config.from_name("Qwen3-0.6B-MoE")
        self.t_config = Config.from_name("Qwen3-14B")
        
        # Load Student (Trainable)
        self.student = GPT(self.s_config)
        self.student.load_state_dict(torch.load(student_path))
        
        # Load Teacher (Frozen)
        self.teacher = GPT(self.t_config)
        self.teacher.load_state_dict(torch.load(teacher_path))
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Projection layers for intermediate states
        self.projections = nn.ModuleList([
            DistillProjector() for _ in range(self.s_config.n_layer)
        ])
        
        # Mapping indices (28 Student -> 40 Teacher)
        self.layer_map = [round(i * (39 / 27)) for i in range(28)]

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        
        # 1. Forward Teacher (Capture Hidden States)
        # We assume model.py has been modified to return hidden_states
        with torch.no_grad():
            t_logits, t_hiddens = self.teacher(input_ids, return_hiddens=True)
        
        # 2. Forward Student (Capture Hidden States + Expert info)
        s_logits, s_hiddens, expert_data = self.student(input_ids, return_hiddens=True)

        # 3. Intermediate Loss (Feature Distillation)
        feature_loss = 0.0
        for s_idx, t_idx in enumerate(self.layer_map):
            s_feat = self.projections[s_idx](s_hiddens[s_idx])
            t_feat = t_hiddens[t_idx]
            feature_loss += F.mse_loss(s_feat, t_feat)

        # 4. Expert Diversity Loss (CKA)
        diversity_loss = 0.0
        for layer_experts in expert_data: # List of expert outputs per layer
            diversity_loss += cka_loss(layer_experts)

        # 5. Output Distillation (KL Divergence)
        T = 2.0 # Temperature
        soft_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T**2)

        # 6. Thinking Loss (Hard SFT)
        hard_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), labels.view(-1))

        # Total Weighted Loss
        total_loss = (1.0 * hard_loss) + (0.5 * soft_loss) + (0.1 * feature_loss) + (0.01 * diversity_loss)
        
        self.log("train_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)