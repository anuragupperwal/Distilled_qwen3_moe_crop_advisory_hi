import torch
import os
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

#  CONFIGURATION 
checkpoint_path = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/lit_model.pth"
tokenizer_dir = "checkpoints/Qwen/Qwen3-0.6B-moe-initial" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
    """
    Standalone generation function for LitGPT models.
    Handles KV Caching for fast inference.
    """
    B, T = idx.shape
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)
    
    # 1. Setup KV Cache (Critical for speed)
    model.set_kv_cache(batch_size=B, max_seq_length=max_seq_length, device=device)
    
    # 2. Prefill Phase: Process the prompt
    # We create a position tensor [0, 1, ... T-1]
    input_pos = torch.arange(0, T, device=device)
    logits = model(idx, input_pos=input_pos)
    
    # Take the last token's logits
    logits = logits[:, -1, :]

    # 3. Decoding Phase: Generate token by token
    for i in range(max_new_tokens):
        # Apply Temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply Top-K Sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample from probabilities
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Stop if EOS token is generated
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

        # 4. Forward pass for the NEXT token only (using Cache)
        # Position is now T + i
        input_pos = torch.tensor([T + i], device=device, dtype=torch.long)
        logits = model(idx_next, input_pos=input_pos)
        logits = logits[:, -1, :]

    return idx


# INPUT FORMATTER (Structured Context) 
def build_agri_prompt(metadata, query):
    """
    Combines structured metadata with the user query into a single Hindi Scenario.
    """
    # Map English keys to Hindi display for the model
    context_str = []
    if "crop" in metadata: context_str.append(f"फसल: {metadata['crop']}")
    if "region" in metadata: context_str.append(f"क्षेत्र: {metadata['region']}")
    if "soil" in metadata: context_str.append(f"मिट्टी: {metadata['soil']}")
    if "stage" in metadata: context_str.append(f"चरण: {metadata['stage']}")
    
    # Join context with proper punctuation
    context_text = " | ".join(context_str)
    
    # Construct the final prompt matching the training format
    # We embed the context into the Scenario block
    prompt = f"""स्थिति (Scenario):
{context_text}
समस्या विस्तार: {query}

"""
    return prompt




#  OUTPUT PARSER (Separates Thinking from Answer) 
def parse_and_display(raw_output):
    """
    Splits the raw output into Thinking and Advisory blocks.
    """
    # Clean up the output to remove the prompt part if needed
    # (Assuming raw_output contains the prompt, we find the first tag)
    
    thinking_tag = "विचार (Thinking):"
    advisory_tag = "परामर्श (Advisory):"
    
    thinking_content = "Not generated."
    advisory_content = raw_output
    
    # Split logic
    if thinking_tag in raw_output:
        parts = raw_output.split(thinking_tag)
        # parts[0] is prompt, parts[1] is thinking + advisory
        if len(parts) > 1:
            remainder = parts[1]
            if advisory_tag in remainder:
                think_split = remainder.split(advisory_tag)
                thinking_content = think_split[0].strip()
                advisory_content = think_split[1].strip()
            else:
                thinking_content = remainder.strip()
    
    # --- DISPLAY UI ---
    print("\n" + "="*50)
    print("THINKING PROCESS (Backend Log)")
    print("="*50)
    print(f"\033[90m{thinking_content}\033[0m") # Print in Gray color
    
    print("\n" + "="*50)
    print("FINAL ADVISORY (User View)")
    print("="*50)
    print(f"\033[92m{advisory_content}\033[0m") # Print in Green color
    print("="*50 + "\n")





def run_inference():
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model not found at {checkpoint_path}")
        return

    print(f"Loading distilled model from {checkpoint_path}...")
    
    # 1. Load Model
    config = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(config).to(device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    # 2. Load Tokenizer
    tokenizer = Tokenizer(tokenizer_dir)

    metadata = {
        "crop": "गन्ना (Sugarcane)",
        "region": "पंजाब",
        "soil": "दोमट (Loam)",
        "stage": "पुष्पन (Flowering)"
    }
    user_query = "मेरी फसल में लाल रंग की सड़न दिखाई दे रही है और पत्तियां सूख रही हैं।"
    
    full_prompt = build_agri_prompt(metadata, user_query)
    input_ids = tokenizer.encode(full_prompt, bos=True, eos=False).to(device)

    print("\nGenerating Response (Thinking + Advisory) ")
    
    # 4. Call our Custom Generate Function
    with torch.no_grad():
        output_ids = generate(
            model, 
            input_ids.unsqueeze(0), 
            max_new_tokens=3072, 
            temperature=0.5, 
            top_k=40,
            eos_id=tokenizer.eos_id
        )

    # 5. Decode
    output_text = tokenizer.decode(output_ids[0])
    print(output_text)

if __name__ == "__main__":
    run_inference()



