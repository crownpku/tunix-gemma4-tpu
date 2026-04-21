import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys

def run_eval(model_id, output_file, num_samples=5):
    print(f"Loading model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    test_data = "data/test.jsonl"
    results = []
    
    print(f"Running evaluation on {num_samples} samples...")
    with open(test_data, "r") as f:
        lines = f.readlines()
        
    for line in lines[:num_samples]:
        example = json.loads(line)
        instruction = example["instruction"]
        
        # Format for Gemma 4 Chat Template
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        results.append({
            "instruction": instruction,
            "expected": example["response"],
            "generated": response
        })
        print(f"Q: {instruction}\nA: {response}\n")
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_comparison_hf.py <model_path> <output_file>")
        sys.exit(1)
    run_eval(sys.argv[1], sys.argv[2])
