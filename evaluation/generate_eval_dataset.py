import torch
import pandas as pd
import os
import gc
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_PARQUET_PATH = "data/raw_train_final_merged_220221.parquet"
OUTPUT_CSV_PATH = "results/my_test_results.csv"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-init"

# Ensure the paths to your specific checkpoints are correct
MODELS = {
    "Base": {
        "config_name": "Qwen3-0.6B-MoE",
        "path": "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth"
    },
    "Distilled": {
        "config_name": "Qwen3-0.6B-MoE",
        "path": "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/02_03_run_test_3k_38EA08/step-3000.pth" # Update step if needed
    },
    "Teacher": {
        "config_name": "Qwen3-8B", 
        "path": "checkpoints/Qwen/Qwen3-8B/lit_model.pth"
    }
}

# ==========================================
# INFERENCE ENGINE (Handles Long Contexts)
# ==========================================
def generate_continuous(model, idx, max_new_tokens, eos_id, temperature=0.3, repetition_penalty=1.15):
    """Dynamic KV cache setup to safely handle 4k-5k+ token inputs."""
    B, T = idx.shape
    
    # Dynamically allocate KV cache for the exact prompt size + desired output
    model.set_kv_cache(batch_size=B, max_seq_length=T + max_new_tokens, device=DEVICE)
    
    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)
    logits = logits[:, -1, :]
    
    generated = []
    
    for i in range(max_new_tokens):
        if repetition_penalty > 1.0:
            context_len = 150
            start_idx = max(0, idx.size(1) - context_len)
            current_context = idx[:, start_idx:].long()
            score = torch.gather(logits, 1, current_context)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, current_context, score)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.tensor([T + i], device=DEVICE, dtype=torch.long)
        logits = model(idx_next, input_pos=input_pos)
        logits = logits[:, -1, :]

    model.clear_kv_cache() # Crucial for preventing VRAM spikes between rows
    
    return torch.cat(generated, dim=1) if len(generated) > 0 else torch.tensor([[]], device=DEVICE)

# ==========================================
# MAIN SCRIPT
# ==========================================
def main():
    print("🚀 Initializing Long-Context Evaluation Dataset Generator...")
    tokenizer = Tokenizer(TOKENIZER_DIR)
    eos_id = tokenizer.eos_id

    # ---------------------------------------------------------
    # STEP 1: Find 100 rows with 4000 - 5000 tokens
    # ---------------------------------------------------------
    print(f"📂 Loading {INPUT_PARQUET_PATH}...")
    df_raw = pd.read_parquet(INPUT_PARQUET_PATH)
    
    # Shuffle to ensure a random distribution of test data
    df_shuffled = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
    
    selected_rows = []
    print("🔍 Searching for 100 rows with 4,000 - 5,000 tokens...")
    
    for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled)):
        # Construct the exact prompt the model saw during training
        full_prompt = (
            f"<|system|>\n{row['system_instruction']}\n"
            f"<|user|>\n{row['prompt']}\n"
            f"<|thought|>\n"
        )
        
        token_count = len(tokenizer.encode(full_prompt))
        
        if 4000 <= token_count <= 5000:
            selected_rows.append({
                'custom_id': row['custom_id'],
                'Full_Prompt': full_prompt,
                'Token_Count': token_count,
                'Reference_Thought': row['thoughts'],
                'Reference_Advisory': row['advisory']
            })
            
        if len(selected_rows) >= 100:
            break
            
    df_eval = pd.DataFrame(selected_rows)
    print(f"✅ Found {len(df_eval)} rows. Average prompt tokens: {df_eval['Token_Count'].mean():.0f}")

    # ---------------------------------------------------------
    # STEP 2: Generate Predictions (Sequentially to protect VRAM)
    # ---------------------------------------------------------
    for model_key, config_data in MODELS.items():
        print(f"\n========================================")
        print(f"🧠 Loading Model: {model_key}")
        print(f"========================================")
        
        config = Config.from_name(config_data["config_name"])
        model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
        
        try:
            model.load_state_dict(torch.load(config_data["path"], map_location=DEVICE, weights_only=True), strict=False)
        except Exception as e:
            print(f"⚠️ Skipping {model_key} - Could not load weights: {e}")
            df_eval[f'{model_key}_Thought'] = "LOAD_FAILED"
            df_eval[f'{model_key}_Advisory'] = "LOAD_FAILED"
            continue
            
        model.eval()
        
        model_thoughts = []
        model_advisories = []
        
        print(f"⚙️ Generating 100 predictions for {model_key}...")
        for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
            input_ids = tokenizer.encode(row['Full_Prompt'], bos=False, eos=False).to(DEVICE)
            
            with torch.no_grad():
                output_ids = generate_continuous(
                    model, 
                    input_ids.unsqueeze(0), 
                    max_new_tokens=1500, 
                    eos_id=eos_id,
                    temperature=0.3 
                )
            
            raw_output = tokenizer.decode(output_ids[0])
            
            # --- PARSE THE OUTPUT (Separate Thoughts from Advisory) ---
            if "<|assistant|>" in raw_output:
                parts = raw_output.split("<|assistant|>")
                thought_text = parts[0].strip()
                advisory_text = parts[1].replace("<|end|>", "").replace("<|start|>", "").strip()
            else:
                # If a model (like the Base model) fails to use the transition tag,
                # we assume the whole output is the advisory and it failed to think.
                thought_text = "NO_THOUGHT_GENERATED"
                advisory_text = raw_output.replace("<|end|>", "").replace("<|start|>", "").strip()
                
            model_thoughts.append(thought_text)
            model_advisories.append(advisory_text)
            
        # Append lists to the DataFrame
        df_eval[f'{model_key}_Thought'] = model_thoughts
        df_eval[f'{model_key}_Advisory'] = model_advisories
        
        print(f"🧹 Clearing VRAM for next model...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # STEP 3: Save to CSV
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # Reorder columns for clean analysis later
    columns_to_save = [
        'custom_id', 'Token_Count', 'Full_Prompt', 
        'Reference_Thought', 'Reference_Advisory',
        'Base_Thought', 'Base_Advisory',
        'Distilled_Thought', 'Distilled_Advisory',
        'Teacher_Thought', 'Teacher_Advisory'
    ]
    
    # Filter to only include columns that actually exist (in case a model load failed)
    columns_to_save = [col for col in columns_to_save if col in df_eval.columns]
    final_df = df_eval[columns_to_save]
    
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Dataset generated successfully! Saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()