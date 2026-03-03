import os
import pandas as pd
from datasets import load_dataset
from sklearn.utils import shuffle

# --- CONFIGURATION ---
# Replace this with the actual path to your 39k custom Hindi dataset
LOCAL_AGRI_HINDI_PATH = "data/train_agri_data_65k.parquet" 
OUTPUT_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"

# Target sizes for the mix
#agri_hi = 65k
NUM_EN_AGRI = 8000 #10%
NUM_HI_CHAT = 5000 #6.5%
NUM_EN_CHAT = 5000 #6.5%

# --- FORMATTING FUNCTION ---
def create_prompt(instruction, input_text, output_text, domain="general"):
    """
    Formats the raw data into your model's exact expected token structure.
    """
    if domain == "agri":
        system_msg = "You are an intelligent agricultural advisor. Answer accurately."
    else:
        system_msg = "You are a helpful, intelligent assistant."

    # Combine instruction and input (if input exists)
    user_msg = str(instruction).strip()
    if pd.notna(input_text) and str(input_text).strip() != "":
        user_msg += f"\n{str(input_text).strip()}"

    # We omit <|thought|> for general chat to teach the model rapid response,
    # but keep the structure identical to your inference pipeline.
    prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n{str(output_text).strip()}\n<|end|>"
    return prompt

def main():
    print("🚀 Starting Data Mixing Pipeline...\n")
    all_formatted_data = []

    # ==========================================
    # 1. LOAD LOCAL CUSTOM HINDI AGRI DATA
    # ==========================================
    if os.path.exists(LOCAL_AGRI_HINDI_PATH):
        print(f"📦 Loading local Hindi Agri data from {LOCAL_AGRI_HINDI_PATH}...")
        df_local = pd.read_parquet(LOCAL_AGRI_HINDI_PATH)
        
        print("   ✅ Formatting custom Agri columns into Two-Phase structure...")
        local_texts = []
        for _, row in df_local.iterrows():
            # Extract exactly the columns we know exist
            sys_msg = str(row.get('system_instruction', "You are an agricultural advisor.")).strip()
            user_msg = str(row.get('prompt', '')).strip()
            thoughts = str(row.get('thoughts', '')).strip()
            advisory = str(row.get('advisory', '')).strip()
            
            # Format exactly as the Two-Phase MoE requires
            full_prompt = (
                f"<|system|>\n{sys_msg}\n"
                f"<|user|>\n{user_msg}\n"
                f"<|thought|>\n{thoughts}\n"
                f"<|assistant|>\n{advisory}\n"
                f"<|end|>"
            )
            local_texts.append(full_prompt)
        
        all_formatted_data.extend(local_texts)
        print(f"   ✅ Added {len(local_texts)} custom Hindi Agri rows.\n")
    else:
        print(f"❌ ERROR: Local file {LOCAL_AGRI_HINDI_PATH} not found. Please check the path.")
        return

    # ==========================================
    # 2. LOAD ENGLISH AGRI DATA
    # ==========================================
    print("🌾 Downloading English Agriculture Data (KisanVaani)...")
    ds_agri_en = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
    ds_agri_en = ds_agri_en.shuffle(seed=42).select(range(min(NUM_EN_AGRI, len(ds_agri_en))))
    
    en_agri_texts = [
        create_prompt(row['question'], "", row['answers'], domain="agri") 
        for row in ds_agri_en
    ]
    all_formatted_data.extend(en_agri_texts)
    print(f"   ✅ Added {len(en_agri_texts)} English Agri rows.\n")

    # ==========================================
    # 3. LOAD HINDI CONVERSATIONAL DATA
    # ==========================================
    print("💬 Downloading Hindi Conversational Data (Alpaca-Hindi)...")
    ds_chat_hi = load_dataset("FreedomIntelligence/alpaca-gpt4-hindi", split="train")
    ds_chat_hi = ds_chat_hi.shuffle(seed=42).select(range(min(NUM_HI_CHAT, len(ds_chat_hi))))
    
    hi_chat_texts = []
    for row in ds_chat_hi:
        # Check if the dataset uses the ShareGPT 'conversations' format
        if 'conversations' in row:
            user_text = row['conversations'][0]['value']
            bot_text = row['conversations'][1]['value']
            hi_chat_texts.append(create_prompt(user_text, "", bot_text, domain="general"))
        
        # Fallback for the standard Alpaca 'instruction' format
        elif 'instruction' in row:
            hi_chat_texts.append(create_prompt(row['instruction'], row.get('input', ''), row['output'], domain="general"))
            
    all_formatted_data.extend(hi_chat_texts)
    print(f"   ✅ Added {len(hi_chat_texts)} Hindi Conversational rows.\n")

    # ==========================================
    # 4. LOAD ENGLISH CONVERSATIONAL DATA
    # ==========================================
    print("🧠 Downloading English Conversational Data (Alpaca)...")
    ds_chat_en = load_dataset("tatsu-lab/alpaca", split="train")
    ds_chat_en = ds_chat_en.shuffle(seed=42).select(range(min(NUM_EN_CHAT, len(ds_chat_en))))
    
    en_chat_texts = [
        create_prompt(row['instruction'], row.get('input', ''), row['output'], domain="general") 
        for row in ds_chat_en
    ]
    all_formatted_data.extend(en_chat_texts)
    print(f"   ✅ Added {len(en_chat_texts)} English Conversational rows.\n")

    # ==========================================
    # 5. MERGE, SHUFFLE, AND SAVE
    # ==========================================
    print("🔀 Shuffling the final dataset to ensure even distribution...")
    df_final = pd.DataFrame({'text': all_formatted_data})
    
    # Shuffle the dataframe thoroughly
    df_final = shuffle(df_final, random_state=42).reset_index(drop=True)
    
    print(f"💾 Saving {len(df_final)} total rows to {OUTPUT_PATH}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False)
    
    print("🎉 Done! Your robust, bilingual distillation dataset is ready.")

if __name__ == "__main__":
    main()