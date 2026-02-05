# import torch
# import os
# import torch.nn.functional as F
# from litgpt.model import GPT
# from litgpt.config import Config
# from litgpt.tokenizer import Tokenizer

# #  CONFIGURATION 
# # checkpoint_path = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/lit_model.pth"
# checkpoint_path = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_405DC3/step-500.pth"
# tokenizer_dir = "checkpoints/Qwen/Qwen3-0.6B-moe-initial" 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
#     """
#     Standalone generation function for LitGPT models.
#     Handles KV Caching for fast inference.
#     """
#     B, T = idx.shape
#     T_new = T + max_new_tokens
#     max_seq_length = min(T_new, model.config.block_size)
    
#     # 1. Setup KV Cache (Critical for speed)
#     model.set_kv_cache(batch_size=B, max_seq_length=max_seq_length, device=device)
    
#     # 2. Prefill Phase: Process the prompt
#     # We create a position tensor [0, 1, ... T-1]
#     input_pos = torch.arange(0, T, device=device)
#     logits = model(idx, input_pos=input_pos)
    
#     # Take the last token's logits
#     logits = logits[:, -1, :]

#     # 3. Decoding Phase: Generate token by token
#     for i in range(max_new_tokens):
#         # Apply Temperature
#         if temperature > 0:
#             logits = logits / temperature
        
#         # Apply Top-K Sampling
#         if top_k is not None:
#             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             logits[logits < v[:, [-1]]] = -float('Inf')

#         # Sample from probabilities
#         probs = F.softmax(logits, dim=-1)
#         idx_next = torch.multinomial(probs, num_samples=1)

#         # Stop if EOS token is generated
#         if eos_id is not None and idx_next.item() == eos_id:
#             break

#         # Append new token to the sequence
#         idx = torch.cat((idx, idx_next), dim=1)

#         # 4. Forward pass for the NEXT token only (using Cache)
#         # Position is now T + i
#         input_pos = torch.tensor([T + i], device=device, dtype=torch.long)
#         logits = model(idx_next, input_pos=input_pos)
#         logits = logits[:, -1, :]

#     return idx


# # INPUT FORMATTER (Structured Context) 
# def build_agri_prompt(metadata, query):
#     """
#     Combines structured metadata with the user query into a single Hindi Scenario.
#     """
#     # Map English keys to Hindi display for the model
#     context_str = []
#     if "crop" in metadata: context_str.append(f"फसल: {metadata['crop']}")
#     if "region" in metadata: context_str.append(f"क्षेत्र: {metadata['region']}")
#     if "soil" in metadata: context_str.append(f"मिट्टी: {metadata['soil']}")
#     if "stage" in metadata: context_str.append(f"चरण: {metadata['stage']}")
    
#     # Join context with proper punctuation
#     context_text = " | ".join(context_str)
    
#     # Construct the final prompt matching the training format
#     # We embed the context into the Scenario block
#     prompt = f"""<|system|>
# You are an agricultural expert.
# <|user|>
# स्थिति (Scenario):
# {context_text}
# समस्या विस्तार: {query}
# <|thought|>
# """

#     return prompt




# #  OUTPUT PARSER (Separates Thinking from Answer) 
# def parse_and_display(raw_output):
#     """
#     Splits the output based on the special tokens <|thought|> and <|assistant|>
#     """
#     # Define the special tags used in training
#     thought_start = "<|thought|>"
#     assistant_start = "<|assistant|>"
    
#     thinking_content = "Not found."
#     advisory_content = raw_output

#     # Logic to split the string based on tags
#     if thought_start in raw_output:
#         # Split everything after <|thought|>
#         after_thought = raw_output.split(thought_start)[1]
        
#         if assistant_start in after_thought:
#             parts = after_thought.split(assistant_start)
#             thinking_content = parts[0].strip()
#             advisory_content = parts[1].strip()
#         else:
#             # If no assistant tag, everything is thought (unlikely if model finished)
#             thinking_content = after_thought.strip()
#             advisory_content = ""

#     # --- UI DISPLAY ---
#     print("\n" + "="*50)
#     print("THINKING PROCESS (Backend Log)")
#     print("="*50)
#     print(f"\033[90m{thinking_content}\033[0m") # Gray
    
#     print("\n" + "="*50)
#     print("FINAL ADVISORY (User View)")
#     print("="*50)
#     print(f"\033[92m{advisory_content}\033[0m") # Green
#     print("="*50 + "\n")





# def run_inference():
#     if not os.path.exists(checkpoint_path):
#         print(f"Error: Model not found at {checkpoint_path}")
#         return

#     print(f"Loading distilled model from {checkpoint_path}...")
    
#     # 1. Load Model
#     config = Config.from_name("Qwen3-0.6B-MoE")
#     model = GPT(config).to(device, dtype=torch.bfloat16)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
#     model.eval()

#     # 2. Load Tokenizer
#     tokenizer = Tokenizer(tokenizer_dir)

#     metadata = {
#         "crop": "गन्ना (Sugarcane)",
#         "region": "पंजाब",
#         "soil": "दोमट (Loam)",
#         "stage": "पुष्पन (Flowering)"
#     }
#     user_query = "मेरी फसल में लाल रंग की सड़न दिखाई दे रही है और पत्तियां सूख रही हैं।"
    
#     full_prompt = build_agri_prompt(metadata, user_query)
#     input_ids = tokenizer.encode(full_prompt, bos=True, eos=False).to(device)

#     print("\nGenerating Response (Thinking + Advisory) ")
    
#     # 4. Call our Custom Generate Function
#     with torch.no_grad():
#         output_ids = generate(
#             model, 
#             input_ids.unsqueeze(0), 
#             max_new_tokens=3072, 
#             temperature=0.5, 
#             top_k=40,
#             eos_id=tokenizer.eos_id
#         )

#     # 5. Decode
#     output_text = tokenizer.decode(output_ids[0])
#     # Pass the full text to the parser
#     print(output_text)
#     parse_and_display(output_text)



# if __name__ == "__main__":
#     run_inference()






import torch
import os
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# --- CONFIGURATION ---
checkpoint_path = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_405DC3/step-1000.pth" # Ensure this path is correct
tokenizer_dir = "checkpoints/Qwen/Qwen3-0.6B-moe-initial" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, tokenizer=None):
    """
    Generation with "Stop String" support.
    Stops if the model outputs specific text patterns like <|end|> or <|assistant|>.
    """
    B, T = idx.shape
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)
    
    model.set_kv_cache(batch_size=B, max_seq_length=max_seq_length, device=device)
    input_pos = torch.arange(0, T, device=device)
    logits = model(idx, input_pos=input_pos)
    logits = logits[:, -1, :]

    # Buffer to hold recent tokens to check for multi-token words like "<|end|>"
    generated_tokens = [] 

    for i in range(max_new_tokens):
        if temperature > 0:
            logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append to our list
        token_id = idx_next.item()
        generated_tokens.append(token_id)
        
        # 1. Check Standard EOS ID
        if tokenizer is not None and token_id == tokenizer.eos_id:
            break

        # 2. Check "Stop Strings"
        # We decode the last 10 tokens to see if they form a stop phrase
        if tokenizer is not None:
            # Decode last few tokens
            recent_text = tokenizer.decode(torch.tensor(generated_tokens[-10:]))
            
            # Check for the tags your model likes to spam
            if "<|end|>" in recent_text or "<|im_end|>" in recent_text:
                break
        

        idx = torch.cat((idx, idx_next), dim=1)
        input_pos = torch.tensor([T + i], device=device, dtype=torch.long)
        logits = model(idx_next, input_pos=input_pos)
        logits = logits[:, -1, :]

    return idx

def build_agri_prompt(metadata, query):
    context_str = []
    if "crop" in metadata: context_str.append(f"फसल: {metadata['crop']}")
    if "region" in metadata: context_str.append(f"क्षेत्र: {metadata['region']}")
    if "soil" in metadata: context_str.append(f"मिट्टी: {metadata['soil']}")
    if "stage" in metadata: context_str.append(f"चरण: {metadata['stage']}")
    context_text = " | ".join(context_str)
    
    # Exact Training Format
    prompt = f"""<|system|>
You are an agricultural expert.
<|user|>
स्थिति (Scenario):
{context_text}
समस्या विस्तार: {query}
<|thought|>
"""
    return prompt

def run_inference():
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model not found at {checkpoint_path}")
        return

    print(f"Loading distilled model from {checkpoint_path}...")
    
    config = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(config).to(device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    tokenizer = Tokenizer(tokenizer_dir)

    metadata = {
        "crop": "गन्ना (Sugarcane)",
        "region": "पंजाब",
        "soil": "दोमट (Loam)",
        "stage": "पुष्पन (Flowering)"
    }
    user_query = "मेरी फसल में लाल रंग की सड़न दिखाई दे रही है और पत्तियां सूख रही हैं।"
    
    full_prompt = build_agri_prompt(metadata, user_query)
    
    # OPTION: Try forcing the first word of the thought to "kickstart" the model
    # Uncomment the next line if the empty output persists
    # full_prompt += "विचार (Thinking): " 

    input_ids = tokenizer.encode(full_prompt, bos=False, eos=False).to(device)

    print("\nGenerating Response... ")
    
    with torch.no_grad():
        output_ids = generate(
            model, 
            input_ids.unsqueeze(0), 
            max_new_tokens=1500, # Short run for debug
            temperature=0.6,    # Slightly higher temp to break loops
            top_k=50,
            tokenizer=tokenizer # Pass tokenizer for debug decoding
        )

    output_text = tokenizer.decode(output_ids[0])
    print("\n--- FINAL RAW OUTPUT ---")
    print(output_text)

if __name__ == "__main__":
    run_inference()