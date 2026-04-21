import json
from datasets import load_dataset
import os

def prepare_dataset():
    dataset = load_dataset("deccan-ai/insuranceQA-v2")
    
    os.makedirs("data", exist_ok=True)
    
    for split in dataset.keys():
        output_file = f"data/{split}.jsonl"
        print(f"Processing {split} split...")
        with open(output_file, "w") as f:
            for item in dataset[split]:
                # Map 'input' to 'instruction' and 'output' to 'response'
                formatted_item = {
                    "instruction": item["input"].strip(),
                    "response": item["output"].strip()
                }
                f.write(json.dumps(formatted_item) + "\n")
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    prepare_dataset()
