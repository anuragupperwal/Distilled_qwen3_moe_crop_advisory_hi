import json
import torch
from torch.utils.data import Dataset
import pandas as pd

class AgriDataset(Dataset):
    """
    Handles CoT (Chain-of-Thought) Distillation from Parquet files.
    Formats: Scenario -> Thinking -> Advisory.
    """
    def __init__(self, data_path, tokenizer, max_seq_length=4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load Parquet directly into a DataFrame
        self.df = pd.read_parquet(data_path)
        print(f"Loaded {len(self.df)} samples from {data_path}")
        self.pad_id = tokenizer.eos_id


    def __len__(self):
        return len(self.df)

    def format_scenario(self, raw_prompt_str):
        """
        Converts the raw 'Growth Stage: ...' string into the Hindi format 
        we decided on for inference.
        """
        # Simple parsing logic (You might need to adjust based on your exact string format)
        lines = raw_prompt_str.split('\n')
        mapping = {}
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                mapping[key.strip()] = val.strip()

        # Construct Hindi Context
        # Matches the build_agri_prompt function from inference
        context_parts = []
        if 'Crop' in mapping: context_parts.append(f"फसल: {mapping['Crop']}")
        if 'Region' in mapping: context_parts.append(f"क्षेत्र: {mapping['Region']}")
        if 'Soil Type' in mapping: context_parts.append(f"मिट्टी: {mapping['Soil Type']}")
        if 'Growth Stage' in mapping: context_parts.append(f"चरण: {mapping['Growth Stage']}")
        
        context_str = " | ".join(context_parts)
        stress = mapping.get('Stress', '')
        
        return f"स्थिति (Scenario):\n{context_str}\nसमस्या विस्तार: {stress}\n"


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Format strings based on your column names
        # Using Hindi labels helps the model learn the semantic structure
        sys_text = f"<|system|>\n{row['system_instruction']}\n"
        user_text = f"<|user|>\n{self.format_scenario(row['prompt'])}"
        thought_text = f"<|thought|>\n{row['thoughts']}\n"
        # Ensure advisory ends with EOS
        assistant_text = f"<|assistant|>\n{row['advisory']}" 

        # Note: We do this separately to calculate lengths for masking
        sys_ids = self.tokenizer.encode(sys_text, bos=False, eos=False)
        user_ids = self.tokenizer.encode(user_text, bos=False, eos=False)
        thought_ids = self.tokenizer.encode(thought_text, bos=False, eos=False)
        assistant_ids = self.tokenizer.encode(assistant_text, bos=False, eos=True) # Add EOS here!

        # Concatenate
        input_ids = torch.cat([sys_ids, user_ids, thought_ids, assistant_ids])

        # Create Mask (IGNORE System + User, LEARN Thought + Assistant)
        # 1 = Calculate Loss, 0 = Ignore
        # For Distillation, we usually want the student to mimic the teacher on EVERYTHING 
        # EXCEPT the prompt.
        mask = torch.ones_like(input_ids)
        mask[:len(sys_ids) + len(user_ids)] = 0  # Mask out System + User

        #Padding / Truncation
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            mask = mask[:self.max_seq_length]
        else:
            # Pad to max_length (standard for batching)
            pad_len = self.max_seq_length - len(input_ids)
            pad_tensor = torch.full((pad_len,), self.pad_id, dtype=torch.long, device=input_ids.device)
            mask_pad = torch.zeros((pad_len,), dtype=torch.long, device=mask.device) 
            
            input_ids = torch.cat([input_ids, pad_tensor])
            mask = torch.cat([mask, mask_pad])

            
        return input_ids, mask
    










