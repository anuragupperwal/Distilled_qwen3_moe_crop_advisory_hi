# --- Core Deep Learning & LitGPT Base ---
torch
lightning
safetensors
huggingface_hub
tqdm
pyyaml
typing-extensions

# --- Tokenization & Data ---
sentencepiece
tiktoken
transformers

# --- Knowledge Distillation & MoE ---
# Used for KL-Divergence and advanced loss functions
scipy
numpy

# --- CKA Analysis & Measurement ---
# Specialized for Centered Kernel Alignment implementation
torchmetrics
matplotlib

# --- Performance & Speed (Recommended) ---
# Faster checkpoint downloading
hf_transfer
# Faster attention kernels (if your GPU supports it)
flash-attn --no-build-isolation