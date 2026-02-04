import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from torcheval.metrics.text import Perplexity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# 1. SETUP
device = torch.device("cuda")
tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B-moe-initial")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Load Test Data (Ensure you have a separate test set or split your train)
df = pd.read_parquet("data/agri_hi_train.parquet").sample(100, random_state=42)  # Using 100 samples for fast eval

def load_model(path, config_name, dtype=torch.bfloat16):
    print(f"Loading {config_name} from {path}...")
    model = GPT(Config.from_name(config_name)).to(device, dtype=dtype)
    # Allow loading with strict=False to handle potential missing keys in MoE
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def calculate_metrics(model, df):
    ppl_metric = Perplexity()
    rouge_l_scores = []
    bleu_scores = []
    
    print("Running Generation & Scoring...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Format Input
        prompt = f"स्थिति (Scenario):\n{row['prompt']}\n\n"
        input_ids = tokenizer.encode(prompt, bos=True, eos=False).to(device)
        
        # 1. Perplexity Calculation (on the Ground Truth Answer)
        target_text = f"विचार (Thinking):\n{row['thoughts']}\n\nपरामर्श (Advisory):\n{row['advisory']}"
        target_ids = tokenizer.encode(target_text, bos=False, eos=True).to(device)
        
        # Feed full sequence to get logits
        full_seq = torch.cat([input_ids, target_ids])
        with torch.no_grad():
            logits = model(full_seq.unsqueeze(0))
        
        # Shift logits and labels for PPL (Standard Auto-regressive loss)
        shift_logits = logits[0, :-1, :].unsqueeze(0)
        shift_labels = full_seq[1:].unsqueeze(0)
        ppl_metric.update(shift_logits, shift_labels)
        
        # 2. Generation Quality (BLEU / ROUGE)
        # Generate strictly: Max 200 tokens
        gen_ids = model.generate(input_ids.unsqueeze(0), max_new_tokens=200, temperature=0.7, top_k=50)
        gen_text = tokenizer.decode(gen_ids[0])
        
        # Extract just the advisory part for scoring (Simplified)
        try:
            generated_advisory = gen_text.split("परामर्श (Advisory):")[1].strip()
            reference_advisory = row['advisory'].strip()
            
            # ROUGE
            scores = scorer.score(reference_advisory, generated_advisory)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
            
            # BLEU
            bleu_scores.append(sentence_bleu([reference_advisory.split()], generated_advisory.split()))
        except IndexError:
            # Model failed to follow format
            rouge_l_scores.append(0.0)
            bleu_scores.append(0.0)

    return {
        "Perplexity": ppl_metric.compute().item(),
        "Avg_ROUGE_L": sum(rouge_l_scores) / len(rouge_l_scores),
        "Avg_BLEU": sum(bleu_scores) / len(bleu_scores)
    }

# --- EVALUATION LOOP ---
models_to_test = {
    "Teacher (14B)": ("checkpoints/Qwen/Qwen3-14B/lit_model.pth", "Qwen3-14B"),
    "Base Student": ("checkpoints/Qwen/Qwen3-0.6B-moe-initial/lit_model.pth", "Qwen3-0.6B-MoE"),
    "Distilled Student": ("checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/lit_model.pth", "Qwen3-0.6B-MoE")
}

results = []
for name, (path, config) in models_to_test.items():
    try:
        model = load_model(path, config)
        metrics = calculate_metrics(model, df)
        metrics["Model"] = name
        results.append(metrics)
        del model # Free VRAM
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Skipping {name}: {e}")

# Save Report
results_df = pd.DataFrame(results)
print("\n--- 🏆 Final Performance Comparison ---")
print(results_df)
results_df.to_csv("outputs/model_comparison_report.csv", index=False)