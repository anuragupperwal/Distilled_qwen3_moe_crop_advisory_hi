# import os
# import torch
# import pandas as pd
# import evaluate
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch.nn.functional as F

# from tqdm import tqdm
# from litgpt.model import GPT
# from litgpt.config import Config
# from litgpt.tokenizer import Tokenizer
# from agri_data import AgriDataset

# # --- CONFIGURATION ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TEST_DATA_PATH = "data/test_data/test_part_003.parquet"
# NUM_SAMPLES = 1
# MAX_NEW_TOKENS = 3072

# MODELS = {
#     "1. Teacher (8B)": (
#         "checkpoints/Qwen/Qwen3-8B/lit_model.pth", 
#         "Qwen3-8B"
#     ),
#     "2. Student (Dense Base)": (
#         "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth", 
#         "Qwen3-0.6B"
#     ),
#     "3. Student (MoE Init)": (
#         "checkpoints/Qwen/Qwen3-0.6B-moe-init/lit_model.pth", 
#         "Qwen3-0.6B-MoE"
#     ),
#     "4. Student (Distilled - Mid)": (
#         "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/1_run_test_51751D/step-1500.pth", 
#         "Qwen3-0.6B-MoE"
#     ),
#     "5. Student (Distilled - Final)": (
#         "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_02_run_test_39k_full_run/lit_model.pth", 
#         "Qwen3-0.6B-MoE"
#     )
# }

# print("⏳ Loading Metrics...")
# bleu = evaluate.load("bleu")
# rouge = evaluate.load("rouge")
# # bertscore = evaluate.load("bertscore") # Slow, uncomment if needed

# def get_config(model_name, config_name):
#     config = Config.from_name(config_name)
#     if "8B" in config_name:
#         config.n_embd = 4096 
#         config.n_head = 32
#         config.intermediate_size = 12288
#     return config

# def generate(model, idx, max_new_tokens, temperature=0.1, top_k=1, tokenizer=None, repetition_penalty=1.2):
#     """
#     Robust Generation with Type Fixes.
#     """
#     B, T = idx.shape
#     T_new = T + max_new_tokens
#     max_seq_length = min(T_new, model.config.block_size)
    
#     model.set_kv_cache(batch_size=B, max_seq_length=max_seq_length, device=DEVICE)
#     input_pos = torch.arange(0, T, device=DEVICE)
#     logits = model(idx, input_pos=input_pos)
#     logits = logits[:, -1, :]

#     stop_tokens = ["<|end|>", "<|im_end|>", "<|user|>", "<|system|>"]

#     for i in range(max_new_tokens):
#         # 1. Repetition Penalty
#         if repetition_penalty > 1.0:
#             current_context = idx[:, -200:].long() 
#             score = torch.gather(logits, 1, current_context)
#             score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
#             logits.scatter_(1, current_context, score)

#         # 2. Temperature
#         if temperature > 0:
#             logits = logits / temperature
        
#         # 3. Top-K
#         if top_k is not None:
#             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             logits[logits < v[:, [-1]]] = -float('Inf')

#         # 4. Sample
#         probs = F.softmax(logits, dim=-1)
#         idx_next = torch.multinomial(probs, num_samples=1)

#         # 5. Check Stop Tokens (FIXED)
#         if tokenizer is not None:
#             # Fix: Use idx_next[0] to pass a 1D tensor to decode
#             token_str = tokenizer.decode(idx_next[0])
            
#             if any(s in token_str for s in stop_tokens):
#                 break
#             # Also standard EOS ID check
#             if idx_next.item() == tokenizer.eos_id:
#                 break

#         idx = torch.cat((idx, idx_next), dim=1)
#         input_pos = torch.tensor([T + i], device=DEVICE, dtype=torch.long)
#         logits = model(idx_next, input_pos=input_pos)
#         logits = logits[:, -1, :]

#     return idx


# def extract_advisory(text):
#     """
#     Robust extraction that handles ALL the tag variations we saw in the logs.
#     """
#     # 1. Normalize ALL observed tags to a standard <|assistant|>
#     text = text.replace("<|start_of_answer|>", "<|assistant|>")  # Teacher 8B
#     text = text.replace("<|assistant_response|>", "<|assistant|>") # Student Mid
#     text = text.replace("<|end_of_response|>", "<|assistant|>")    # Student Final
#     text = text.replace("<|response|>", "<|assistant|>")           # Base Model
#     text = text.replace("<|output|>", "<|assistant|>")             # JSON hallucination
    
#     # 2. Extract after the standardized tag
#     if "<|assistant|>" in text:
#         advisory = text.split("<|assistant|>")[1]
        
#         # 3. Clean up trailing garbage tags
#         for stop_tag in ["<|user|>", "<|end|>", "<|system|>", "<|thought|>", "<|end_of_thought|>"]:
#             if stop_tag in advisory:
#                 advisory = advisory.split(stop_tag)[0]
#         return advisory.strip()
    
#     # 4. Fallback: If no tag found, but <|thought|> exists, take what's after
#     elif "<|thought|>" in text:
#         parts = text.split("<|thought|>")
#         if len(parts) > 1:
#             # Assume the second half contains the answer if no explicit tag
#             return parts[1].split("\n")[-1] # Heuristic: Take the last paragraph
            
#     # 5. Last Resort: Return everything (Better to get low score than 0)
#     return text.strip()



# def plot_metrics(df, save_path="outputs/model_comparison.png"):
#     sns.set_theme(style="whitegrid")
#     # Melt for grouped bar chart
#     df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
#     plt.figure(figsize=(12, 6))
#     chart = sns.barplot(
#         data=df_melted, x="Metric", y="Score", hue="Model",
#         palette="viridis", edgecolor="black"
#     )
    
#     for container in chart.containers:
#         chart.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

#     plt.title("Model Performance: Dense vs MoE Distillation", fontsize=16)
#     plt.ylim(0, 1.0) # Scores are 0-1
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     print(f"📊 Plot saved to {save_path}")
#     plt.close()

# def run_evaluation():
#     # Use base tokenizer
#     tokenizer_path = "checkpoints/Qwen/Qwen3-0.6B"
#     if not os.path.exists(tokenizer_path):
#         tokenizer_path = "checkpoints/Qwen/Qwen3-0.6B-moe-init"
#     tokenizer = Tokenizer(tokenizer_path)

#     df = pd.read_parquet(TEST_DATA_PATH)
#     if len(df) > NUM_SAMPLES:
#         df = df.sample(NUM_SAMPLES, random_state=42)
    
#     dummy_dataset = AgriDataset(TEST_DATA_PATH, tokenizer, max_seq_length=128)
    
#     results = []

#     for model_name, (ckpt_path, config_name) in MODELS.items():
#         print(f"\n🧠 Evaluating: {model_name}")
        
#         if not os.path.exists(ckpt_path):
#             print(f"   ❌ Checkpoint not found. Skipping.")
#             continue
            
#         config = get_config(model_name, config_name)
#         model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
        
#         try:
#             model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True), strict=False)
#         except Exception as e:
#             print(f"   ⚠️ Load Error: {e}")
#             continue
            
#         model.eval()
        
#         predictions = []
#         references = []
        
#         pbar = tqdm(df.iterrows(), total=len(df), desc="Generating")
#         for _, row in pbar:
#             scenario = dummy_dataset.format_scenario(row['prompt'])
            
#             # TEACHER/STUDENT PROMPTING DIFFERENCE
#             # Student (MoE) was trained to think first: <|thought|>
#             # Teacher (8B) might just answer. 
#             # To be fair, we use the Student Format for everyone.
#             prompt = f"<|system|>\nYou are an agricultural expert.\n<|user|>\n{scenario}\n<|thought|>\n"
            
#             input_ids = tokenizer.encode(prompt, bos=False, eos=False).to(DEVICE)
            
#             with torch.no_grad():
#                 output_ids = generate(
#                     model, 
#                     input_ids.unsqueeze(0), 
#                     max_new_tokens=MAX_NEW_TOKENS,
#                     repetition_penalty=1.2,
#                     tokenizer=tokenizer # Pass tokenizer for stop string check
#                 )
            
#             full_text = tokenizer.decode(output_ids[0])
#             print(full_text)
            
#             # IMPORTANT: We compare Advisory vs Advisory
#             generated_advisory = extract_advisory(full_text)
#             print("raw for: ", model_name)
            
#             predictions.append(generated_advisory)
#             references.append(row['advisory'])

#         print(f"   📉 Computing Metrics...")
#         try:
#             b_score = bleu.compute(predictions=predictions, references=references)
#             r_score = rouge.compute(predictions=predictions, references=references)
            
#             scores = {
#                 "Model": model_name,
#                 "BLEU": b_score['bleu'],
#                 "ROUGE-L": r_score['rougeL'],
#             }
#             results.append(scores)
#             print(f"   ✅ {model_name}: BLEU={scores['BLEU']:.4f}, ROUGE-L={scores['ROUGE-L']:.4f}")
            
#         except Exception as e:
#             print(f"   ❌ Metric Calculation Failed: {e}")

#         del model
#         torch.cuda.empty_cache()

#     if results:
#         results_df = pd.DataFrame(results)
#         print("\n🏆 Final Evaluation Results:")
#         print(results_df.to_markdown(index=False))
        
#         os.makedirs("outputs", exist_ok=True)
#         results_df.to_csv("outputs/model_comparison_metrics.csv", index=False)
#         plot_metrics(results_df, save_path="outputs/model_comparison_plot.png")
#     else:
#         print("No valid results computed.")

# if __name__ == "__main__":
#     run_evaluation()








import os
import torch
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from tqdm import tqdm
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_PATH = "data/test_data/test_part_003.parquet"
NUM_SAMPLES = 1  
MAX_NEW_TOKENS = 2048

MODELS = {
    "1. Teacher (8B)": ("checkpoints/Qwen/Qwen3-8B/lit_model.pth", "Qwen3-8B", "standard"),
    # "2. Student (Dense)": ("checkpoints/Qwen/Qwen3-0.6B/lit_model.pth", "Qwen3-0.6B", "standard"),
    # "3. Student (MoE Init)": ("checkpoints/Qwen/Qwen3-0.6B-moe-init/lit_model.pth", "Qwen3-0.6B-MoE", "standard"),
    "4. Student (Distilled Final)": ("checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_02_run_test_39k_full_run/lit_model.pth", "Qwen3-0.6B-MoE", "two_phase")
}

print("⏳ Loading Metrics...")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore") # Crucial for Semantic Evaluation

def get_config(model_name, config_name):
    config = Config.from_name(config_name)
    if "8B" in config_name:
        config.n_embd = 4096 
        config.n_head = 32
        config.intermediate_size = 12288
    return config

# --- ROBUST GENERATOR ---
def generate_stepwise(model, idx, max_new_tokens, temperature=0.3, top_k=40, top_p=0.9, stop_tokens_ids=None, repetition_penalty=1.2):
    B, T = idx.shape
    model.set_kv_cache(batch_size=B, max_seq_length=T + max_new_tokens, device=DEVICE)
    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)[:, -1, :]
    
    generated = []
    for i in range(max_new_tokens):
        if repetition_penalty > 1.0 and idx.size(1) > 1:
            context_len = 200
            start_idx = max(0, idx.size(1) - context_len)
            current_context = idx[:, start_idx:].long()
            score = torch.gather(logits, 1, current_context)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, current_context, score)

        logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # EXACT ID MATCHING
        if stop_tokens_ids is not None and idx_next.item() in stop_tokens_ids:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)
        input_pos = torch.tensor([T + i], device=DEVICE, dtype=torch.long)
        logits = model(idx_next, input_pos=input_pos)[:, -1, :]

    if len(generated) > 0:
        return torch.cat(generated, dim=1)
    return torch.tensor([[]], device=DEVICE, dtype=torch.long)

def plot_metrics(df, save_path):
    sns.set_theme(style="whitegrid")
    df_melted = df.melt(id_vars="Model", value_vars=["BLEU", "ROUGE-L", "BERTScore"], var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis", edgecolor="black")
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.title("Model Performance: Semantic vs Lexical Metrics", fontsize=16)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_evaluation():
    tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B-moe-init")
    eos_id = tokenizer.eos_id
    
    # Safe encoding of tags
    assistant_tag_ids = tokenizer.encode("<|assistant|>", bos=False, eos=False).to(DEVICE)
    assistant_id = assistant_tag_ids[0, 0].item() if assistant_tag_ids.ndim > 1 else assistant_tag_ids[0].item()

    df = pd.read_parquet(TEST_DATA_PATH)
    if len(df) > NUM_SAMPLES:
        df = df.sample(NUM_SAMPLES, random_state=42)
    
    results = []
    qualitative_log = []

    for model_name, (ckpt_path, config_name, strategy) in MODELS.items():
        print(f"\n🧠 Evaluating: {model_name}")
        
        if not os.path.exists(ckpt_path):
            print(f"   ❌ Checkpoint not found. Skipping.")
            continue
            
        config = get_config(model_name, config_name)
        model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True), strict=False)
        model.eval()
        
        predictions = []
        references = []
        
        pbar = tqdm(df.iterrows(), total=len(df), desc="Generating")
        for _, row in pbar:
            # Reconstruct scenario
            scenario = f"फसल: {row.get('crop','')} | क्षेत्र: {row.get('region','')} | चरण: {row.get('stage','')}"
            query = row['prompt']
            target_advisory = row['advisory']
            
            with torch.no_grad():
                if strategy == "two_phase":
                    # --- MOE DISTILLED LOGIC ---
                    prompt = f"<|system|>\nYou are an agricultural expert.\n<|user|>\nस्थिति:\n{scenario}\nसमस्या: {query}\n<|thought|>\n"
                    input_ids = tokenizer.encode(prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)
                    
                    thought_ids = generate_stepwise(model, input_ids, max_new_tokens=300, temperature=0.3, stop_tokens_ids={assistant_id, eos_id})
                    thought_text = tokenizer.decode(thought_ids[0]).replace("<|assistant|>", "").strip()
                    
                    clean_thought_ids = tokenizer.encode(thought_text, bos=False, eos=False).to(DEVICE).unsqueeze(0)
                    phase2_input = torch.cat([input_ids, clean_thought_ids, assistant_tag_ids.unsqueeze(0) if assistant_tag_ids.ndim==1 else assistant_tag_ids], dim=1)
                    
                    adv_ids = generate_stepwise(model, phase2_input, max_new_tokens=800, temperature=0.4, stop_tokens_ids={eos_id})
                    final_advisory = tokenizer.decode(adv_ids[0]).strip()
                    print("if: ", final_advisory)

                else:
                    # --- TEACHER / BASE LOGIC ---
                    prompt = f"<|system|>\nYou are an agricultural expert.\n<|user|>\nस्थिति:\n{scenario}\nसमस्या: {query}\n<|assistant|>\n"
                    input_ids = tokenizer.encode(prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)
                    
                    adv_ids = generate_stepwise(model, input_ids, max_new_tokens=800, temperature=0.4, stop_tokens_ids={eos_id})
                    final_advisory = tokenizer.decode(adv_ids[0]).strip()
                    print("else: ", final_advisory)

            # --- CRITICAL FIX: PREVENT DIVISION BY ZERO IN BLEU ---
            if not final_advisory.strip():
                print(f"\n   ⚠️ [Warning] {model_name} generated an empty string. Inserting placeholder.")
                final_advisory = "<empty_prediction>"

            predictions.append(final_advisory)
            references.append(target_advisory)
            
            
            # Save for manual inspection
            qualitative_log.append({
                "Model": model_name,
                "Query": query,
                "Ground_Truth": target_advisory,
                "Prediction": final_advisory
            })

        print(f"   📉 Computing Metrics...")
        try:
            if len(predictions) == 0:
                raise ValueError("No predictions were generated to evaluate.")

            b_score = bleu.compute(predictions=predictions, references=references)
            r_score = rouge.compute(predictions=predictions, references=references)
            
            bert_res = bertscore.compute(predictions=predictions, references=references, lang="hi")
            
            # CRITICAL FIX: Safe mean calculation (No 0.00001 hacks)
            f1_list = bert_res['f1']
            mean_bert = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0.0

            scores = {
                "Model": model_name,
                "BLEU": b_score['bleu'],
                "ROUGE-L": r_score['rougeL'],
                "BERTScore": mean_bert
            }
            results.append(scores)
            print(f"   ✅ BLEU={scores['BLEU']:.4f} | ROUGE-L={scores['ROUGE-L']:.4f} | BERTScore={scores['BERTScore']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Metric Calculation Failed: {e}")

        del model
        torch.cuda.empty_cache()

    os.makedirs("outputs", exist_ok=True)
    if results:
        results_df = pd.DataFrame(results)
        print("\n🏆 Final Evaluation Results:")
        print(results_df.to_markdown(index=False))
        results_df.to_csv("outputs/evaluation_metrics.csv", index=False)
        plot_metrics(results_df, save_path="outputs/evaluation_plot.png")
        
    if qualitative_log:
        pd.DataFrame(qualitative_log).to_csv("outputs/qualitative_predictions.csv", index=False)
        print("📁 Saved generated texts to outputs/qualitative_predictions.csv")

if __name__ == "__main__":
    run_evaluation()