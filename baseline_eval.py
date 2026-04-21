import json
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm

def baseline_eval(model_id, data_path, output_path, num_samples=50):
    print(f"Loading processor and model from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    # On CPU, avoid half precision for stability if not supported
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu"
    )
    
    print(f"Loading test data from {data_path}...")
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    
    subset = dataset[:num_samples]
    results = []
    
    print(f"Running evaluation on {num_samples} samples...")
    for item in tqdm(subset):
        instruction = item["instruction"]
        messages = [{"role": "user", "content": instruction}]
        
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(text=prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        response = processor.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        results.append({
            "instruction": instruction,
            "expected": item["response"],
            "generated": response
        })
        
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    model_id = "models/gemma-4-E2B-it"
    data_path = "data/test.jsonl"
    output_path = "results/baseline_results.json"
    os.makedirs("results", exist_ok=True)
    baseline_eval(model_id, data_path, output_path, num_samples=10) # 10 samples for quick check on CPU
