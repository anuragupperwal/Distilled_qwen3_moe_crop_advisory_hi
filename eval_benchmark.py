import os
import torch
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns  

from tqdm import tqdm
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from agri_data import AgriDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_PATH = "data/test_part_003.parquet"
NUM_SAMPLES = 10 
MAX_NEW_TOKENS = 1024

MODELS = {
    "1. Teacher (8B)":     ("checkpoints/Qwen/Qwen3-8B/lit_model.pth", "Qwen3-8B", "teacher"),
    "2. Student (Base)":   ("checkpoints/Qwen/Qwen3-0.6B-moe-initial/lit_model.pth", "Qwen3-0.6B-moe-initial", "student"),
    "3. Student (Mid)":    ("checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_434D09/step-500.pth", "Qwen3-0.6B-Agri-Distilled", "student"),
    "4. Student (Final)":  ("checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_434D09/lit_model.pth", "Qwen3-0.6B-Agri-Distilled", "student")
}

# Load Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")

def generate_response(model, tokenizer, prompt, device):
    encoded = tokenizer.encode(prompt, device=device)
    prompt_len = encoded.size(0)
    generated = model.generate(
        encoded.unsqueeze(0), 
        max_new_tokens=MAX_NEW_TOKENS, 
        temperature=0.1, 
        top_k=1, 
        eos_id=tokenizer.eos_id
    )
    output_ids = generated[0][prompt_len:]
    return tokenizer.decode(output_ids)

def plot_metrics(df, save_path="outputs/model_comparison.png"):
    """
    Generates a grouped bar chart comparing models across all metrics.
    """
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Transform data for plotting (Melt the dataframe)
    # We want columns: [Model, Metric, Score]
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    
    # Create Grouped Bar Chart
    chart = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette="viridis", # Good color scheme for research
        edgecolor="black"
    )
    
    # Add values on top of bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    plt.title("Model Performance Comparison (Distillation)", fontsize=16, pad=20)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.1) # Assuming scores are 0-1 (except maybe PPL)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

def run_evaluation():
    try:
        tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B-moe-initial")
    except:
        tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B") 
        
    df = pd.read_parquet(TEST_DATA_PATH).sample(NUM_SAMPLES, random_state=42)
    
    results = []

    dummy_dataset = AgriDataset(TEST_DATA_PATH, tokenizer, max_seq_length=128)

    # Iterate through models
    for model_name, (ckpt_path, config_name, arch_type) in MODELS.items():
        print(f"\nEvaluating: {model_name}...")
        
        # Load Model Logic 
        config = Config.from_name(config_name)
        if "8B" in config_name:
            config.n_embd = 4096 
            config.n_head = 32
            config.intermediate_size = 12288

        model = GPT(config)
        # Ensure strict=False handles the loading
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True), strict=False)
        model.to(DEVICE).eval()
        
        predictions = []
        references = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            formatted_scenario = dummy_dataset.format_scenario(row['prompt'])
            prompt = f"<|system|>\n{row['system_instruction']}\n<|user|>\n{formatted_scenario.strip()}\n<|thought|>\n"
            reference = row['advisory']
            
            try:
                pred = generate_response(model, tokenizer, prompt, DEVICE)
            except Exception as e:
                print(f"Error: {e}")
                pred = ""
                
            predictions.append(pred)
            references.append(reference)
            
        print(f"Computing scores for {model_name}...")
        
        b_score = bleu.compute(predictions=predictions, references=references)
        r_score = rouge.compute(predictions=predictions, references=references)
        bs_score = bertscore.compute(predictions=predictions, references=references, lang="hi")
        m_score = meteor.compute(predictions=predictions, references=references)

        scores = {
            "Model": model_name,
            "BLEU": b_score['bleu'],
            "ROUGE-1": r_score['rouge1'],
            "ROUGE-L": r_score['rougeL'],
            "BERTScore_F1": sum(bs_score['f1']) / len(bs_score['f1']),
            "METEOR": m_score['meteor']
        }
        results.append(scores)
        
        del model
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    
    print("\nFinal Evaluation Results ")
    print(results_df.to_markdown(index=False))
    
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/model_comparison_metrics.csv", index=False)
    print("Saved to outputs/model_comparison_metrics.csv")
    
    plot_metrics(results_df, save_path="outputs/model_comparison_plot.png")

if __name__ == "__main__":
    run_evaluation()




