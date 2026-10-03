import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced QLoRA SFT Pipeline")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_id", type=str, default="philschmid/dolly-15k-oai-style")
    parser.add_argument("--output_dir", type=str, default="./tuned-model-lora")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    print(f"Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fixes strange behaviors with fp16 training

    # 1. Advanced 4-Bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat 4 - mathematically optimal for weights
        bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 to maintain speed/stability
        bnb_4bit_use_double_quant=True       # Secondary quantization to save extra VRAM
    )

    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Prepare for memory-efficient training
    model.config.use_cache = False # Required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration (Targeting Attention + MLP)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", # Attention
            "gate_proj", "up_proj", "down_proj"     # MLP
        ]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    print(f"Loading Dataset {args.dataset_id}...")
    dataset = load_dataset(args.dataset_id, split="train")

    def format_chat_template(example):
        # Applies the model's native chat template (e.g., ChatML) to the raw messages
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = dataset.map(format_chat_template, num_proc=os.cpu_count())

    # 5. Advanced Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,       # Effective batch size = 16
        gradient_checkpointing=True,         # Trades compute for VRAM 
        optim="paged_adamw_8bit",            # Pages optimizer states to CPU RAM if GPU runs out
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        num_train_epochs=3,
        bf16=True,                           # Faster than fp16 on Ampere+ GPUs
        logging_steps=10,
        save_strategy="epoch",
        neftune_noise_alpha=5.0,             # Adds noise to embeddings, improving instruction adherence
        report_to="none"                     # Set to "wandb" for real tracking
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving Final Adapter...")
    trainer.model.save_pretrained(f"{args.output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/final_adapter")

if __name__ == "__main__":
    main()
