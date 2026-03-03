import os
import pandas as pd
import numpy as np
import sacrebleu
import google.generativeai as genai
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer, util
from math import pi
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 2.0 Flash (Fastest and most cost-effective for JSON parsing)
judge_model = genai.GenerativeModel('gemini-2.5-flash')

# Create Results Directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"📁 Results will be saved to: {RESULTS_DIR}")

# ==========================================
# 1. MATH METRICS (chrF++ & LaBSE)
# ==========================================
def calculate_chrf(predictions, references):
    """Calculates chrF++ for grammar and character-level overlap."""
    refs_formatted = [references] 
    chrf = sacrebleu.corpus_chrf(predictions, refs_formatted)
    return chrf.score

def calculate_labse(predictions, references):
    """Calculates LaBSE for semantic meaning similarity."""
    print("   -> Loading LaBSE embeddings...")
    labse = SentenceTransformer('sentence-transformers/LaBSE')
    
    ref_embs = labse.encode(references, convert_to_tensor=True)
    pred_embs = labse.encode(predictions, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(pred_embs, ref_embs)
    scores = torch.diag(cosine_scores).cpu().numpy()
    return np.mean(scores)

# ==========================================
# 2. LLM-AS-A-JUDGE (Gemini API)
# ==========================================
def get_gemini_score(scenario, reference, prediction):
    """Calls Gemini to grade the response using strict JSON mode."""
    prompt = f"""You are an expert Indian Agricultural Scientist evaluating an AI assistant.
    
    SCENARIO: {scenario}
    GOLDEN REFERENCE: {reference}
    MODEL PREDICTION: {prediction}
    
    Grade the prediction strictly on a scale of 1 to 5 for the following criteria:
    1. Factual_Accuracy: Does it suggest the correct treatment/inputs?
    2. Tone: Is it respectful and culturally appropriate for an Indian farmer?
    3. Safety: Does it strictly adhere to constraints (e.g., no chemicals if organic)?
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Force the API to return a strict JSON object
            response = judge_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                )
            )
            scores = json.loads(response.text)
            
            # Map 5-point scale to a 10-point scale for the main table
            avg_5_point = (scores.get("Factual_Accuracy", 0) + scores.get("Tone", 0) + scores.get("Safety", 0)) / 3
            scores["Overall_10_point"] = avg_5_point * 2 
            
            return scores
            
        except Exception as e:
            print(f"   -> API Error (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(4) 
            
    return {"Factual_Accuracy": 0, "Tone": 0, "Safety": 0, "Overall_10_point": 0}

# ==========================================
# 3. PLOTTING & VISUALIZATION
# ==========================================
def plot_radar_chart(results_dict, save_path):
    """Generates an industry-standard Radar Chart for qualitative metrics."""
    categories = ['Factual_Accuracy', 'Tone', 'Safety']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], ['Factual Accuracy', 'Tone', 'Safety'], size=12)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
    plt.ylim(0, 5)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] 
    for i, (model_name, scores) in enumerate(results_dict.items()):
        values = [scores['Factual_Accuracy'], scores['Tone'], scores['Safety']]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    plt.title("Qualitative Assessment (LLM Judge)", size=16, y=1.1, family='serif')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_quantitative_bars(results_df, save_path):
    """Generates a side-by-side bar chart for chrF++ and LaBSE."""
    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    models = results_df.index
    
    sns.barplot(x=models, y=results_df['chrF++'], ax=ax1, palette=colors)
    ax1.set_title('Grammar & Fluency (chrF++)', fontsize=14)
    ax1.set_ylabel('Score (0-100)')
    ax1.set_ylim(0, 100)
    
    sns.barplot(x=models, y=results_df['LaBSE'], ax=ax2, palette=colors)
    ax2.set_title('Semantic Similarity (LaBSE)', fontsize=14)
    ax2.set_ylabel('Score (0-1)')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_table_as_image(df, caption, save_path):
    """Renders a Pandas DataFrame in a strict Academic/LaTeX Booktabs style."""
    # Force serif font for the academic look
    plt.rcParams["font.family"] = "serif"
    
    # Calculate figure height dynamically based on rows
    fig, ax = plt.subplots(figsize=(8, len(df)*0.5 + 1.2))
    ax.axis('off')
    
    # Reset index so 'Model' becomes a standard column for formatting
    df_reset = df.reset_index()
    df_reset.rename(columns={'index': 'Models'}, inplace=True)
    
    # Create the table
    # We restrict the bbox height so there is room at the bottom for the caption
    table = ax.table(
        cellText=df_reset.round(3).values,
        colLabels=df_reset.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 1, 0.8] 
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Apply styling: Strip borders, align columns
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0) # Remove all default grid lines
        
        # Left-align the first column (Models), center the rest
        if col == 0:
            cell._loc = 'left'
        else:
            cell._loc = 'center'
            
        # Make header row text bold
        if row == 0:
            cell.set_text_props(weight='bold')

    # Draw the strict academic lines (toprule, midrule, bottomrule)
    # y=1.0 is top, y=0.2 is bottom of the table bbox
    num_rows = len(df_reset) + 1 # +1 for header
    row_height = 0.8 / num_rows
    
    y_top = 1.0
    y_mid = 1.0 - row_height
    y_bot = 0.2
    
    # Plot the literal lines over the axes
    ax.plot([0, 1], [y_top, y_top], color='black', lw=1.5, transform=ax.transAxes, clip_on=False) # Toprule
    ax.plot([0, 1], [y_mid, y_mid], color='black', lw=0.75, transform=ax.transAxes, clip_on=False) # Midrule
    ax.plot([0, 1], [y_bot, y_bot], color='black', lw=1.5, transform=ax.transAxes, clip_on=False) # Bottomrule
    
    # Add the caption below the table, left-aligned
    plt.figtext(0.05, 0.05, caption, ha='left', fontsize=11, family='serif', wrap=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Load your actual data here: df = pd.read_csv("my_test_results.csv")
    data = {
        "Scenario": ["मक्का | फूल आना | जैविक खेती | तुलासिता रोग"],
        "Reference": ["जैविक खेती के लिए 50 ग्राम ट्राइकोडर्मा को 10 लीटर पानी में मिलाकर स्प्रे करें। खेत में जलभराव न होने दें।"],
        "Base_Pred": ["खेत में 500ml नीम का तेल डालो और मरे हुए पौधे मिट्टी में दबा दो।"],
        "Distilled_Pred": ["राम राम! आपकी जैविक मक्का के लिए, 10 लीटर पानी में 50 ग्राम ट्राइकोडर्मा विरिडी मिलाएं और छिड़काव करें।"],
        "Teacher_Pred": ["किसान भाई, तुलासिता रोग से बचाव हेतु 50g ट्राइकोडर्मा प्रति 10L पानी में मिलाकर छिड़कें। खेत साफ रखें।"]
    }
    df = pd.DataFrame(data)
    
    models = ["Qwen3-0.6B (Base Student)", "Qwen3-0.6B-MoE (Distilled Student)", "Qwen3-8B (Teacher)"]
    final_results = {}
    qualitative_results = {}

    print("\n🚀 Starting Evaluation Pipeline...\n")
    
    for model in models:
        print(f"📊 Evaluating {model}...")
        preds = df[model].tolist()
        refs = df["Reference"].tolist()
        
        # 1. Math Metrics
        chrf_score = calculate_chrf(preds, refs)
        labse_score = calculate_labse(preds, refs)
        
        # 2. Gemini LLM Judge
        print("   -> Querying Gemini API...")
        gemini_scores_list = []
        for idx, row in df.iterrows():
            scores = get_gemini_score(row["Scenario"], row["Reference"], row[model])
            gemini_scores_list.append(scores)
            time.sleep(2) 
            
        avg_factual = np.mean([s["Factual_Accuracy"] for s in gemini_scores_list])
        avg_tone = np.mean([s["Tone"] for s in gemini_scores_list])
        avg_safety = np.mean([s["Safety"] for s in gemini_scores_list])
        avg_overall = np.mean([s["Overall_10_point"] for s in gemini_scores_list])
        
        final_results[model] = {
            "chrF++": round(chrf_score, 1),
            "LaBSE": round(labse_score, 3),
            "LLM Judge (/10)": round(avg_overall, 1)
        }
        qualitative_results[model] = {
            "Factual_Accuracy": avg_factual,
            "Tone": avg_tone,
            "Safety": avg_safety
        }

    # Convert to Pandas DataFrames
    results_df = pd.DataFrame(final_results).T
    qual_df = pd.DataFrame(qualitative_results).T

    # ------------------------------------------
    # SAVE ALL ARTIFACTS
    # ------------------------------------------
    print("\n💾 Saving Artifacts to /results/...")
    
    # 1. Save CSVs
    results_df.to_csv(os.path.join(RESULTS_DIR, "table1_quantitative_results.csv"))
    qual_df.to_csv(os.path.join(RESULTS_DIR, "table2_qualitative_results.csv"))
    
    # 2. Save LaTeX
    with open(os.path.join(RESULTS_DIR, "table1_latex.tex"), "w") as f:
        f.write(results_df.to_latex())
    with open(os.path.join(RESULTS_DIR, "table2_latex.tex"), "w") as f:
        f.write(qual_df.to_latex())

    # 3. Save Academic Table Images
    save_table_as_image(
        results_df, 
        "Table 1: Main quantitative and LLM-as-a-Judge evaluation results.", 
        os.path.join(RESULTS_DIR, "table1_image.png")
    )
    save_table_as_image(
        qual_df, 
        "Table 2: Qualitative breakdown of LLM-as-a-Judge scoring criteria.", 
        os.path.join(RESULTS_DIR, "table2_image.png")
    )

    # 4. Save Plots
    plot_radar_chart(qualitative_results, os.path.join(RESULTS_DIR, "plot_radar_qualitative.png"))
    plot_quantitative_bars(results_df, os.path.join(RESULTS_DIR, "plot_bars_quantitative.png"))

    print("\n✅ Evaluation Complete! Artifacts generated:")
    print(" - table1_quantitative_results.csv / .tex / .png")
    print(" - table2_qualitative_results.csv / .tex / .png")
    print(" - plot_radar_qualitative.png")
    print(" - plot_bars_quantitative.png")

if __name__ == "__main__":
    main()