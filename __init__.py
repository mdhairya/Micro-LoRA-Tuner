import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

def get_quantization_config() -> BitsAndBytesConfig:
    """Returns optimal 4-bit NormalFloat configuration for memory-efficient training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

def get_lora_config(r: int = 16, alpha: int = 32) -> LoraConfig:
    """Returns LoRA config targeting both Attention and MLP layers."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", # Attention
            "gate_proj", "up_proj", "down_proj"     # MLP
        ]
    )
