import os
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def prepare_chat_dataset(dataset_id: str, tokenizer: PreTrainedTokenizer, split: str = "train"):
    """
    Loads a dataset and applies the model's native chat template to the messages.
    """
    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, split=split)

    def format_chat_template(example):
        # Converts [{"role": "user", "content": "..."}] into the model's specific string format
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    # Process in parallel for speed
    processed_dataset = dataset.map(format_chat_template, num_proc=os.cpu_count())
    return processed_dataset
