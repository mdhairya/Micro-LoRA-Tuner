import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="./tuned-model-lora/final_adapter")
    parser.add_argument("--output_dir", type=str, default="./merged-local-model")
    args = parser.parse_args()

    print("Loading base model in bfloat16 for clean mathematical merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading and applying LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging AxB matrices into base linear layers...")
    model = model.merge_and_unload()

    print(f"Saving standalone model to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Merge complete. Ready for local inference deployment.")

if __name__ == "__main__":
    main()
