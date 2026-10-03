import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Import from our custom module
from micro_lora.configs import get_quantization_config, get_lora_config
from micro_lora.data import prepare_chat_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run QLoRA Supervised Fine-Tuning")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_id", type=str, default="philschmid/dolly-15k-oai-style")
    parser.add_argument("--output_dir", type=str, default="./tuned-model-lora")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Setup Model with 4-bit Quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=get_quantization_config(),
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)

    # 3. Apply LoRA Adapters
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    # 4. Prepare Data
    dataset = prepare_chat_dataset(args.dataset_id, tokenizer)

    # 5. Define Training Hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        neftune_noise_alpha=5.0,
        report_to="none" 
    )

    # 6. Initialize and Run Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=get_lora_config(),
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting SFT Training...")
    trainer.train()
    
    print(f"Saving finalized adapter to {args.output_dir}/final_adapter")
    trainer.model.save_pretrained(f"{args.output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/final_adapter")

if __name__ == "__main__":
    main()
