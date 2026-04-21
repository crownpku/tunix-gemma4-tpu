import os
# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import json
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors
import sys

def manual_generate(model, tokenizer, prompt, max_len=64):
    token_ids = tokenizer.encode(prompt).ids
    input_ids = jnp.array([token_ids])
    
    generated = []
    # Simple non-cached generation for logic simplicity
    for _ in range(max_len):
        # We need attention mask and positions for Gemma 4
        seq_len = input_ids.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        # Causal mask: [B, 1, L, L]
        mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
        
        logits = model(input_ids, positions=positions, attention_mask=mask)
        next_token = jnp.argmax(logits[0, -1, :])
        
        if next_token == tokenizer.token_to_id("<eos>"):
            break
            
        generated.append(int(next_token))
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        
    return tokenizer.decode(generated)

def run_eval(model_dir, output_file, num_samples=5):
    print(f"Loading model from {model_dir}...")
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = params_safetensors.create_model_from_safe_tensors(
        file_dir=model_dir,
        config=config,
        dtype=jnp.bfloat16
    )
    
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    
    print(f"Running manual evaluation on {num_samples} samples...")
    test_data = "data/test.jsonl"
    results = []
    
    with open(test_data, "r") as f:
        lines = f.readlines()
        
    for line in lines[:num_samples]:
        example = json.loads(line)
        instruction = example["instruction"]
        prompt = f"<bos><|turn|>user\n{instruction}<turn|>\n<|turn|>model\n"
        
        response = manual_generate(model, tokenizer, prompt)
            
        results.append({
            "instruction": instruction,
            "expected": example["response"],
            "generated": response
        })
        print(f"Q: {instruction}\nA: {response}\n")
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python manual_eval.py <model_dir> <output_file>")
        sys.exit(1)
    run_eval(sys.argv[1], sys.argv[2])
