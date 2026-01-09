# Distilled-Qwen3-MoE: Upcycled Sparse Mixture-of-Experts for Hindi Crop Advisory

This project focuses on the development of a high-efficiency, reasoning-capable Language Model specifically optimized for the Indian agricultural sector. By "upcycling" the **Qwen3-0.6B** dense model into a **Sparse Mixture-of-Experts (MoE)** architecture, we achieve a model that maintains a small deployment footprint while significantly increasing its specialized capacity for Hindi-language crop diagnostics and advisory.

---

## Technical Architecture: From Dense to Sparse

The core of this project is the transformation of a standard dense transformer into a sparse MoE via **Sparse Upcycling**.

### 1. The Architecture Shift

Instead of training a MoE from scratch, we perform "weight surgery" on the Qwen3-0.6B model:

* **Frozen Attention:** We retain the pre-trained Attention and LayerNorm weights, as these already capture the linguistic structure of Hindi and English.
* **MLP Proliferation:** We duplicate the original Feed-Forward Network (FFN) layers into  identical "experts."
* **Router Initialization:** We introduce a Top- gating mechanism (the Router) that learns to dispatch tokens to the most relevant experts.
* **CKA Distillation:** To ensure the upcycled model retains the base model's knowledge, we use **Centered Kernel Alignment (CKA)** loss during initial training to minimize the representational shift between the dense teacher and the sparse student.

### 2. The "Thinking Token" Extension

To improve the model's agricultural reasoning (e.g., "If  pest is present and the weather is , then apply "), we experiment with **Reasoning Distillation**:

* **Vocab Expansion:** Adding `<thought>` tokens to the vocabulary.
* **Latent Reasoning:** Training the model to generate internal rationales before providing a final answer in Hindi.
* **Masked Distillation:** Using a teacher model to guide the final output while allowing the student MoE to utilize its "thinking" tokens to bridge the gap.

---

## Application: Dynamic Crop Advisory (Hindi)

While the architecture is cutting-edge, the objective is practical: providing real-time, context-aware agricultural support to farmers in their native language.

### Key Capabilities:

* **Localized Diagnostics:** Farmers can describe crop symptoms in Hindi (e.g., "गेहूं के पत्तों पर पीले धब्बे," *Yellow spots on wheat leaves*), and the model identifies potential diseases.
* **Dynamic Advisory:** Unlike static lookup tables, the model synthesizes advice based on variables like soil health, crop age, and current weather conditions.
* **Bilingual Reasoning:** While the input and output are in Hindi, the model leverages its English pre-training for technical botanical knowledge, translating complex agricultural science into actionable Hindi instructions.

---


## Project Structure

```text
qwen3-moe-project/
├── src/
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── configuration_qwen3_moe.py   # Hyperparams (num_experts, etc.)
│   │   ├── modular_qwen3_moe.py         # THE SOURCE: Your custom logic
│   │   └── modeling_qwen3_moe.py        # GENERATED: Standalone for production
│   ├── surgery/
│   │   ├── __init__.py
│   │   ├── upcycle.py                   # Script to convert 0.6B Dense -> MoE
│   │   └── vocab_extender.py            # Logic for adding <thought> tokens
│   └── utils/
│       ├── __init__.py
│       ├── cka_loss.py                  # Centered Kernel Alignment implementation
│       └── distil_utils.py              # Custom loss functions for distillation
├── scripts/
│   ├── run_upcycle.sh                   # Bash script to trigger weight surgery
│   ├── train_distill.py                 # Main training/distillation loop
│   └── eval_cka.py                      # Validation script for structural parity
├── configs/
│   ├── base_upsample.yaml               # Training params for CKA phase
│   └── thought_distill.yaml             # Training params for Thinking phase
├── data/                                # Local samples for initial testing
├── tests/                               # Unit tests for router logic
└── requirements.txt
```



#### 1. `src/modeling/` (The Architecture)

* **`modular_qwen3_moe.py`**: This is where you write your code. Define your `Qwen3MoeSparseMoeBlock` here.
* **`modeling_qwen3_moe.py`**: You will use a script (or manual copy-paste) to flatten the modular code into this file so it can be used with `AutoModel`.

#### 2. `src/surgery/` (The Weight Mapping)

This is the most critical part for your specific project.

* **`upcycle.py`**: This script loads the **Qwen3-0.6B** (Dense) state dict, creates a new **Qwen3-MoE** (Sparse) model, and maps the weights.
* *Logic:* It takes `layer.i.mlp.gate_proj` and maps it to `layer.i.mlp.experts.gate_up_proj` across all  experts.



#### 3. `src/utils/` (The Math)

* **`cka_loss.py`**: Put your CKA implementation here. You will import this into your training script to ensure the "Upsampled" model still behaves like the "Base" model.
* **`distil_utils.py`**: Since you want to use "Thinking Tokens," you'll need a custom loss that masks out the thought tokens when comparing the Student (MoE) to the Teacher (Dense).

#### 4. `scripts/train_distill.py` (The Execution)

This is your entry point for training. It should:

1. Load the Teacher (Qwen3-0.6B).
2. Load the Student (Your Upcycled MoE).
3. Load the data.
4. Run the forward passes and calculate `Loss = Alpha * Task_Loss + Beta * CKA_Loss`.


---

### Contact & Contribution

This project is currently in the **Architecture Validation** phase. If you are interested in the intersection of MoE upcycling and Indic-language LLMs for social good, please reach out.
