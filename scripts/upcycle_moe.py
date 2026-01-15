import torch
import logging
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
                jitter = torch.randn_like(weight) * 1e-5
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

if __name__ == "__main__":
    # Example usage - Update paths as per your local setup
    DENSE_CKPT = Path("checkpoints/Qwen/Qwen3-0.6B/lit_model.pth")
    OUTPUT_DIR = Path("checkpoints/Qwen/Qwen3-0.6B-moe-initial/")
    
    upcycle_checkpoint(DENSE_CKPT, OUTPUT_DIR)